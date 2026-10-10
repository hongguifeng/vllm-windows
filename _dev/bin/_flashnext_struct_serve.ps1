# Launcher for Qwen3.8-Flash-Next on the PLE-free view and in production.
#
# This is deliberately NOT start_server.ps1: that one pins the 27B's knobs
# (marlin int8 inputs, --mamba-ssm-cache-dtype float16, the DFlash2 drafter,
# the WSL batch-3 patch). None of those belong to qwen4_exp.
#
# Without -KvGiB this lets vLLM profile VRAM and report how much room the
# weights actually leave, which is what the structural probe wants. With -KvGiB
# it pins the KV budget outright and skips that profiling, which is what the
# measurement arms want; _dev\bin\_serve_bg.ps1 defaults to this script, so
# every campaign launch so far went through here.
#
#   & D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1              # GPU1, LM only, eager
#   & D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1 -WithVision  # vision tower + max_pixels cap
#   & ...\_flashnext_struct_serve.ps1 -WithVision -MaxPixels 0               # uncapped images
#   & ...\_flashnext_struct_serve.ps1 -WithVision -MaxImages 0               # no per-prompt image cap
#   & ...\_flashnext_struct_serve.ps1 -WithVision -MaxImages 8               # cap at 8 images per request
#   & D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1 -Graphs      # drop --enforce-eager
#   & D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1 -DryRun      # print argv only
#   & D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1 -ThinkingOnDefault  # template default (thinking on)
#   & ...\_flashnext_struct_serve.ps1 -NoPromptTokensDetails  # usage without prompt_tokens_details
#   & D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1 -NoTools -NoReasoningParser  # probe arms
#   & D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1 -FullWeights  # the 143 GiB checkpoint
#   & D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1 -FullWeights -PleSsd
#
# Through bash, quote every Windows path: an unquoted -Venv D:\code\... arrives
# with its backslashes stripped and every path check then fails.
#
# It runs against the PLE-free view built by _dev\probe\_flashnext_trim.py, on
# the card that is free (GPU1 by default; GPU0 belongs to the WSL container).
#
# The installed tree carries one probe-only patch (see
# _dev\out\wsl_flash_next\probe_inc_moe.patch): INC now recognises RoutedExperts
# MoE layers and, when VLLM_PROBE_INC_HUMMING_LOWBIT=1, sends CUDA 2/3/5/6/7-bit
# weights to humming instead of dying in MoeWNA16Method. Revert with
#   Copy-Item D:\code\vllm-windows\_dev\out\installed_backups\inc\*.orig ... (see doc)

[CmdletBinding()]
param(
    [string]$Model = 'D:\models\Qwen3.8-Flash-Next-struct',
    [string]$ServedName = 'qwen3.8-flash-next-struct',
    [string]$Venv = 'D:\code\vllm-windows',
    [int]$Port = 8111,
    [int]$Gpu = 1,
    [int]$MaxLen = 4096,
    [int]$MaxSeqs = 4,
    [int]$BatchedTokens = 2048,
    # Fair chunked prefill (both default 0 = upstream scheduling). A positive
    # -PrefillChunkWithDecodes caps the prefill tokens issued per step whenever
    # something is decoding, which is what keeps co-resident decode latency down
    # while a long prompt is being prefilled.
    [int]$PrefillChunkWithDecodes = 0,
    [int]$MaxNumPartialPrefills = 0,
    [double]$MemUtil = 0.96,
    [string]$LoadStrategy = '',
    [long]$KvGiB = 0,
    [switch]$FullWeights,
    [switch]$PleSsd,
    [int]$PleDepth = 256,
    [int]$PleCacheMb = 128,
    [int]$PleWorkers = 8,
    [int]$PlePrefetchTokens = 0,
    [int]$OmpThreads = 0,
    [switch]$FusedAlign,
    [switch]$IterDetails,
    [switch]$PreparedLaunch,
    [switch]$PleTrace,
    [switch]$PleIdsPoll,
    [switch]$PleWitness,
    [int]$StepTimingWindow = 0,
    [switch]$PhaseEvents,
    [int]$PhaseEventsLogEvery = 2000,
    [switch]$AttnBuildTiming,
    [int]$AttnBuildLogEvery = 200,
    [switch]$Profiler,
    [int]$ProfilerActiveIterations = 30,
    [switch]$NSys,
    [switch]$AsyncSched,
    [switch]$WithVision,      # vision tower instead of --language-model-only
    [int]$MaxPixels = 1310720,
    # 0 (default) = no per-prompt image cap beyond the context window; a
    # positive value caps how many images one request may carry.
    [int]$MaxImages = 0,
    [switch]$NoAllocHeal,
    [switch]$Graphs,
    # Empty (default) lets vLLM derive capture sizes itself. Pass a bracketed
    # list to reproduce the WSL arm's explicit capture coverage, e.g.
    #   -CaptureSizes '[1,2,3,...,2048]'
    [string]$CaptureSizes = '',
    [switch]$Mtp,
    [int]$MtpTokens = 1,
    # -AdaptiveK turns on the acceptance-adaptive draft count: the scheduler
    # picks this step's verification width from the running requests' recent
    # acceptance instead of always using -MtpTokens. Value is the maximum count
    # (must be <= -MtpTokens). Default -1 means "follow -MtpTokens", i.e. the
    # feature is on; pass 0 to keep speculative decoding byte-identical to a
    # fixed draft count.
    [int]$AdaptiveK = -1,
    [string]$ToolParser = 'qwen3_xml',
    [switch]$NoTools,
    [switch]$NoReasoningParser,
    [switch]$ThinkingOnDefault,
    # usage.prompt_tokens_details.cached_tokens is ON unless this is passed.
    # Upstream ships the switch off (vllm/entrypoints/launchers/cli_args.py:139).
    [switch]$NoPromptTokensDetails,
    # -Force kills whatever holds the engine handshake port; without it the
    # script fails fast instead (see the port guard below).
    [switch]$Force,
    # Engine handshake port. 0 = derive one from -Port (see the guard below).
    [int]$DpRpcPort = 0,
    [switch]$DryRun
)

$ErrorActionPreference = 'Continue'
$REPO = 'D:\code\vllm-windows'
$DEV  = "$REPO\_dev"
$py   = "$Venv\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath "$Model\config.json")) {
    Write-Host "ERROR: no config.json under $Model -- run _dev\probe\_flashnext_trim.py first" -ForegroundColor Red
    exit 1
}

# The engine handshake socket is a machine-global singleton by default: its
# port is `data_parallel_rpc_port`'s config default (29550) and on Windows
# launch_core_engines forces handshake_local_only=False, so the address is that
# fixed TCP one rather than a per-process uuid. Two consequences, both seen on
# this rig: a crashed instance (the API server outlives a failed EngineCore)
# keeps 29550 bound and the next start dies a minute in with
# `zmq.error.ZMQError: Address in use`, and a second instance can only come up
# by sharing that one port, which lets a frontend handshake with the wrong
# EngineCore. Give every instance its own port instead: -Force is then only
# about this instance's leftovers.
if ($DpRpcPort -le 0) { $DpRpcPort = 29550 + ($Port % 1000) }
foreach ($p in $DpRpcPort, $Port) {
    $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $pids = ($conn | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
        if ($DryRun) {
            # -DryRun starts nothing and must not touch a single process.
            Write-Host "port $p held by PID $pids (-DryRun: reporting only, nothing killed)" -ForegroundColor DarkGray
        } elseif ($Force) {
            Write-Host "port $p held by PID $pids - killing (-Force)" -ForegroundColor Yellow
            $pids -split ', ' | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 2
        } else {
            Write-Host "ERROR: port $p is already in use by PID $pids" -ForegroundColor Red
            Write-Host "       (a leftover EngineCore? re-run with -Force to kill it)" -ForegroundColor Yellow
            exit 1
        }
    }
}

# -FullWeights moves to the real 143 GiB checkpoint unless -Model was given.
if ($FullWeights) {
    if ($Model -eq 'D:\models\Qwen3.8-Flash-Next-struct') {
        $Model = 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP'
    }
    if ($ServedName -eq 'qwen3.8-flash-next-struct') {
        $ServedName = 'qwen3.8-flash-next-full'
    }
}

if ($PleSsd) {
    $env:VLLM_ALLOW_EXTERNAL_PLE_SSD_LIBRARY = '1'
}

# humming JITs its MoE kernels through nvrtc_compile.exe and patches the cubins
# through libcubinpatch.dll; all three of its native helpers are prebuilt here so
# that starting the server costs nothing. Without them the first MoE forward dies.
$hummingNative = "$Venv\.venv\Lib\site-packages\humming\_native\x86_64"
$needed = @('_device_info.pyd', 'nvrtc_compile.exe', 'libcubinpatch.dll', 'libhumming_launcher.so', 'manifest.json')
$missing = @($needed | Where-Object { -not (Test-Path -LiteralPath "$hummingNative\$_") })
if ($missing) {
    Write-Host ("ERROR: humming native artifacts missing: " + ($missing -join ', ')) -ForegroundColor Red
    Write-Host 'run these three, in this order:' -ForegroundColor Yellow
    Write-Host '  & D:\code\vllm-windows\_dev\bin\_humming_device_info.ps1'
    Write-Host '  & D:\code\vllm-windows\_dev\bin\_humming_nvrtc.ps1'
    Write-Host '  & D:\code\vllm-windows\_dev\bin\_humming_cubinpatch.ps1'
    Write-Host '  & D:\code\vllm-windows\_dev\bin\_humming_launcher.ps1'
    exit 1
}

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$OutputEncoding = [System.Text.Encoding]::UTF8

function EscJson([string]$Json) {
    # PowerShell 5.1 throws away embedded double quotes when it hands an argument
    # to a native program, so '{"a": 1}' arrives as {a: 1} and argparse rejects
    # it. Backslash-escaping the quotes survives the trip.
    return ($Json -replace '"', '\"')
}

$log = "$DEV\out\logs\flashnext_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss')
if (-not (Test-Path "$DEV\out\logs")) {
    New-Item -ItemType Directory -Path "$DEV\out\logs" -Force | Out-Null
}

$env:CUDA_VISIBLE_DEVICES      = "$Gpu"
$env:VLLM_ENABLE_V1_MULTIPROCESSING = '0'
$env:VLLM_USE_FLASHINFER_SAMPLER    = '0'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED = '1'
# Long-context scratch buffers stay cached after a request finishes, so free
# VRAM settles near zero and later allocations are served by host paging -- the
# state in which 8K prefill grows from 2 s to 7 s and only a restart helps.
# WSL runs this healer; hand it the same thresholds here.
if ($NoAllocHeal) {
    $env:QSA_ALLOC_HEAL = '0'
} else {
    $env:QSA_ALLOC_HEAL = '1'
    if (-not $env:QSA_ALLOC_HEAL_FREE_MB) { $env:QSA_ALLOC_HEAL_FREE_MB = '384' }
    if (-not $env:QSA_ALLOC_HEAL_CACHED_MB) { $env:QSA_ALLOC_HEAL_CACHED_MB = '256' }
    if (-not $env:QSA_ALLOC_HEAL_COOLDOWN_S) { $env:QSA_ALLOC_HEAL_COOLDOWN_S = '1.0' }
}
if ($OmpThreads -gt 0) {
    $env:OMP_NUM_THREADS = "$OmpThreads"
    # The reference engine pins both; leaving MKL free lets BLAS spawn a second
    # set of threads that compete with the step thread for the same cores.
    $env:MKL_NUM_THREADS = "$OmpThreads"
}
if ($FusedAlign) {
    $env:VLLM_MAMBA_ALIGN_FUSED = '1'
}
if ($PreparedLaunch) {
    $env:VLLM_TRITON_PREPARED_LAUNCH = '1'
}
if ($PleTrace) {
    $env:VLLM_PLE_SSD_TRACE = '1'
}
if ($PleIdsPoll) {
    $env:VLLM_PLE_SSD_IDS_POLL = '1'
}
if ($PleWitness) {
    $env:VLLM_PLE_SNAPSHOT_WITNESS = '1'
}
if ($StepTimingWindow -gt 0) {
    $env:VLLM_STEP_TIMING_WINDOW = "$StepTimingWindow"
}
if ($PhaseEvents) {
    $env:VLLM_PHASE_EVENTS = '1'
    $env:VLLM_PHASE_EVENTS_LOG_EVERY = "$PhaseEventsLogEvery"
}
if ($AttnBuildTiming) {
    $env:VLLM_ATTN_BUILD_TIMING = '1'
    $env:VLLM_ATTN_BUILD_LOG_EVERY = "$AttnBuildLogEvery"
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8       = '1'
$env:TORCH_CUDA_ARCH_LIST = '8.0'
$env:VLLM_TARGET_DEVICE   = 'cuda'
# humming finds nvrtc64_*.dll by walking CUDA_PATH; without it the JIT path
# reports "Could not locate libnvrtc.so in CUDA path" and the MoE never builds.
$env:CUDA_PATH  = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$env:CUDA_HOME  = $env:CUDA_PATH
$env:CUDA_ROOT  = $env:CUDA_PATH
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:False'
# Without this the 2/3-bit experts fall through to MoeWNA16Method, which only
# knows int4/int8 and raises before a single layer is built.
$env:VLLM_PROBE_INC_HUMMING_LOWBIT = '1'

$argv = @(
    '-u', '-m', 'vllm.entrypoints.openai.api_server',
    '--model', $Model,
    '--served-model-name', $ServedName,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--data-parallel-rpc-port', "$DpRpcPort",
    '--dtype', 'bfloat16',
    '--max-model-len', "$MaxLen",
    '--max-num-seqs', "$MaxSeqs",
    '--max-num-batched-tokens', "$BatchedTokens",
    '--gpu-memory-utilization', "$MemUtil",
    '--mamba-cache-mode', 'align',
    '--language-model-only'
)
if ($WithVision) {
    # --language-model-only was appended above; drop it by rebuilding argv.
    $argv = $argv | Where-Object { $_ -ne '--language-model-only' }
    # -MaxImages 0 leaves the cap at vLLM's own default (999), so the context
    # window is the only thing that binds; a positive value sets the cap.
    $imageCap = if ($MaxImages -gt 0) { $MaxImages } else { 999 }
    $argv += @('--limit-mm-per-prompt', (EscJson ('{"image":{"count":' + $imageCap + '}}')))
    # One token is 32x32 px (patch 16 x merge 2), so max_pixels 1310720 caps an
    # image at 1280 tokens. Without it the checkpoint allows 4096x4096, i.e.
    # 16384 tokens per image, so an uncapped request could eat the whole context.
    if ($MaxPixels -gt 0) {
        $argv += @('--mm-processor-kwargs', (EscJson ('{"max_pixels": ' + $MaxPixels + '}')))
    }
}
if (-not $Graphs) { $argv += '--enforce-eager' }
if ($CaptureSizes) {
    # Only the size list differs from the WSL arm; cudagraph_mode already matches
    # (FULL_AND_PIECEWISE is what the Windows engine resolves today).
    $argv += @('--compilation-config', (EscJson (
        '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": ' +
        $CaptureSizes + '}')))
}
if ($LoadStrategy) {
    $argv += @('--safetensors-load-strategy', $LoadStrategy)
}
if ($PrefillChunkWithDecodes -gt 0) {
    $argv += @('--prefill-chunk-with-decodes', "$PrefillChunkWithDecodes")
}
if ($MaxNumPartialPrefills -gt 0) {
    $argv += @('--max-num-partial-prefills', "$MaxNumPartialPrefills")
}
if ($KvGiB -gt 0) {
    $kvBytes = [long]($KvGiB * 1GB)
    $argv += @('--kv-cache-memory-bytes', "$kvBytes")
}
if ($IterDetails) {
    $argv += @('--enable-logging-iteration-details')
}
if ($PleSsd) {
    $pleDll = 'D:/code/vllm-windows/_dev/out/ple_ssd_io/ple_ssd_io_win.dll'
    if (-not (Test-Path -LiteralPath "D:\code\vllm-windows\_dev\out\ple_ssd_io\ple_ssd_io_win.dll")) {
        Write-Host 'ERROR: ple_ssd_io_win.dll missing -- run _dev\bin\_ple_ssd_io_win_build.ps1' -ForegroundColor Red
        exit 1
    }
    $pleCfg = '{"ple_ssd_offload": true, "ple_ssd_native_library": "' +
        $pleDll + '", "ple_ssd_io_depth": ' + $PleDepth +
        ', "ple_ssd_cache_mb": ' + $PleCacheMb +
        ', "ple_ssd_workers": ' + $PleWorkers
    if ($PlePrefetchTokens -gt 0) {
        $pleCfg += ', "ple_ssd_prefetch_tokens": ' + $PlePrefetchTokens
    }
    $argv += @('--additional-config', (EscJson ($pleCfg + '}')))
}
if ($Profiler) {
    # A profiled block is an attribution screen, not a throughput baseline: the
    # extras stay off and only a few dozen steps go active.
    $profDir = 'D:/code/vllm-windows/_dev/out/profile'
    New-Item -ItemType Directory -Force -Path $profDir | Out-Null
    $argv += @(
        '--profiler-config.profiler', 'torch',
        '--profiler-config.torch_profiler_dir', $profDir,
        '--profiler-config.torch_profiler_with_stack', 'false',
        '--profiler-config.torch_profiler_record_shapes', 'false',
        '--profiler-config.torch_profiler_with_memory', 'false',
        '--profiler-config.torch_profiler_with_flops', 'false',
        '--profiler-config.ignore_frontend', 'true',
        '--profiler-config.wait_iterations', '5',
        '--profiler-config.warmup_iterations', '3',
        '--profiler-config.active_iterations', $ProfilerActiveIterations
    )
}
if ($NSys) {
    # nsys only traces the process it starts, so the engine has to live in this
    # one process, and the capture opens on cudaProfilerStart from the endpoint.
    $env:VLLM_ENABLE_V1_MULTIPROCESSING = '0'
    $argv += @('--profiler-config.profiler', 'cuda')
}
if ($Mtp) {
    if ($AdaptiveK -lt 0) { $AdaptiveK = $MtpTokens }
    $spec = '{"method": "mtp", "num_speculative_tokens": ' + $MtpTokens + '}'
    if ($AdaptiveK -gt 0) {
        if ($AdaptiveK -gt $MtpTokens) {
            throw "-AdaptiveK ($AdaptiveK) cannot exceed -MtpTokens ($MtpTokens)"
        }
        $spec = '{"method": "mtp", "num_speculative_tokens": ' + $MtpTokens +
            ', "adaptive_k": {"min": 1, "max": ' + $AdaptiveK +
            ', "log_interval": 200}}'
    }
    $argv += @('--speculative-config', (EscJson $spec))
}
if ($AsyncSched) {
    # Overlap the CPU side of a step with the GPU side; worth it when a step is
    # half CPU dispatch rather than half compute.
    $argv += '--async-scheduling'
}

# Tool calling needs BOTH flags. Without them every request carrying `tools`
# returns HTTP 400 -- including one with no `tool_choice`, which defaults to
# "auto" (vllm/entrypoints/openai/chat_completion/protocol.py:915).
#
# The parser has to match the format the chat template asks for. Qwen3.8 asks
# for the XML envelope (a tool_call tag wrapping function=NAME and
# parameter=KEY), not the JSON body `hermes` reads. Picking wrong raises
# nothing: the call comes back as ordinary `content`, the client sees no
# `tool_calls`, and it reads as a model problem instead of a server one.
# qwen3_xml, qwen3_coder and mimo are three aliases for the same parser.
if (-not $NoTools) {
    $argv += @('--enable-auto-tool-choice', '--tool-call-parser', $ToolParser)
}
# Lifts the thinking block out of `content` into `reasoning` -- vLLM 0.29 names
# that field `reasoning`, not DeepSeek's `reasoning_content`. With this on and
# thinking left enabled, probes that read `content` get None: send
# enable_thinking false per request, or read `reasoning`.
if (-not $NoReasoningParser) {
    $argv += @('--reasoning-parser', 'qwen3')
}
# Thinking stays off unless a request asks for it. The template treats an
# undefined enable_thinking as true (chat_template.jinja:46), so without this
# every request pays a ~40 token reasoning-instruction prompt and the answer
# lands in `reasoning` instead of `content`. Per-request chat_template_kwargs
# still win (vllm/renderers/params.py:116-128), so -ThinkingOnDefault only
# moves the default, it never locks the knob.
if (-not $ThinkingOnDefault) {
    $argv += @('--default-chat-template-kwargs', (EscJson '{"enable_thinking": false}'))
}
# Lift the prefix-cache hit out of the black box: with this the usage block of
# /v1/chat/completions and /v1/completions carries
# prompt_tokens_details.cached_tokens (RequestOutput.num_cached_tokens) plus
# created_cache_tokens, so a probe can tell "prefilled" from "served from cache"
# without reading the server log. Upstream defaults it off, hence the on-by-default
# here; -NoPromptTokensDetails restores the upstream shape for A/B arms.
if (-not $NoPromptTokensDetails) {
    $argv += '--enable-prompt-tokens-details'
}

Write-Host ''
Write-Host '==== flash-next structural smoke ====' -ForegroundColor Cyan
Write-Host ("python : $py")
# Measured, not asserted: which card this run lands on and who is already on it.
# GPU0 used to belong to the WSL container by convention, but that is a runtime
# state, not a property of the card, so read it instead of reciting it.
$gpuLine = "gpu    : CUDA_VISIBLE_DEVICES=$Gpu"
try {
    $gcsv = & nvidia-smi --id="$Gpu" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
    $gf   = ($gcsv | Select-Object -First 1) -split ','
    $gpuLine += ("  ({0} GiB used / {1} total, {2} GiB free)" -f
        [math]::Round([double]$gf[1] / 1024, 1), [math]::Round([double]$gf[0] / 1024, 1),
        [math]::Round(([double]$gf[0] - [double]$gf[1]) / 1024, 1))
    $holders = @(nvidia-smi --id="$Gpu" --query-compute-apps=pid --format=csv,noheader)
    if ($holders) {
        $gpuLine += "  holders: $($holders -join ', ')"
        if ($Gpu -ne 0) {
            $gpuLine += "  (another engine on this card -- they share one NVMe, so never bench both)"
        }
    }
} catch { $gpuLine += '  (nvidia-smi unavailable)' }
Write-Host $gpuLine
Write-Host ("model  : $Model")
Write-Host ("mode   : " + $(if ($Graphs) { 'CUDA graphs' } else { 'enforce-eager' }) +
            $(if ($WithVision) {
                if ($MaxPixels -gt 0) { " + vision tower  max_pixels=$MaxPixels (~$([math]::Floor($MaxPixels/1024)) tok/image)" }
                else { ' + vision tower  max_pixels=checkpoint (~16384 tok/image)' }
            } else { '  (--language-model-only)' }) +
            $(if ($WithVision) {
                if ($MaxImages -gt 0) { "  images<=${MaxImages}/prompt" } else { '  images unlimited (context-bound)' }
            } else { '' }) +
            $(if ($FullWeights) { '  full weights' } else { '  PLE-free view' }) +
            $(if ($PleSsd) { "  + PLE SSD (ask $PleDepth, $PleCacheMb MiB cache, $PleWorkers workers" + $(if ($PlePrefetchTokens -gt 0) { ", prefetch $PlePrefetchTokens" } else { '' }) + ')' } else { '' }) +
            $(if ($Mtp) { "  + MTP x$MtpTokens" } else { '' }) +
            $(if ($AsyncSched) { '  + async scheduling' } else { '' }) +
            $(if ($NoTools) { '   tools OFF' } else { "  + tools ($ToolParser)" }) +
            $(if ($NoReasoningParser) { '  no reasoning parser' } else { '  + reasoning qwen3' }) +
            $(if ($ThinkingOnDefault) { '  thinking on by default' } else { '  thinking off by default' }) +
            $(if ($NoPromptTokensDetails) { '  usage.cached_tokens OFF' } else { '  usage.cached_tokens ON' }))
Write-Host ("limits : $MaxLen ctx, $MaxSeqs seqs, $BatchedTokens batched tokens, util $MemUtil" +
    $(if ($PrefillChunkWithDecodes -gt 0) { ", prefill chunk w/ decodes $PrefillChunkWithDecodes" } else { '' }) +
    $(if ($MaxNumPartialPrefills -gt 0) { ", partial prefills $MaxNumPartialPrefills" } else { '' }) +
    $(if ($Mtp -and $AdaptiveK -gt 0) { ", adaptive k<=$AdaptiveK" } else { ', fixed k' }) +
            $(if ($KvGiB -gt 0) { ", KV $KvGiB GiB fixed" } else { '' }) +
            $(if ($LoadStrategy) { ", load $LoadStrategy" } else { '' }) +
            $(if ($NoAllocHeal) { ', alloc-heal OFF' } else { ", alloc-heal free<$($env:QSA_ALLOC_HEAL_FREE_MB)MiB cached>$($env:QSA_ALLOC_HEAL_CACHED_MB)MiB" }))
Write-Host ("log    : $log")
Write-Host ''

if ($DryRun) {
    Write-Host ($py + ' ' + ($argv -join ' '))
    exit 0
}

# Convention: never start vLLM inside the repo -- the checked-out vllm\ tree
# shadows the installed package and its built extensions.
Push-Location 'C:\Users\hong'
if ($NSys) {
    $nsysExe = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe'
    if (-not (Test-Path -LiteralPath $nsysExe)) {
        Write-Host "ERROR: nsys missing: $nsysExe" -ForegroundColor Red
        exit 1
    }
    $nsysDir = 'D:\code\vllm-windows\_dev\out\nsys'
    New-Item -ItemType Directory -Force -Path $nsysDir | Out-Null
    $nsysArgs = @(
        'profile',
        '--trace', 'cuda,nvtx',
        '--cuda-graph-trace', 'node',
        '--capture-range', 'cudaProfilerApi',
        '--capture-range-end', 'repeat',
        '--force-overwrite', 'true',
        '--output', "$nsysDir\stepspan"
    )
    Write-Host "nsys: capturing only between start_profile and stop_profile" -ForegroundColor Yellow
    try {
        & $nsysExe @nsysArgs $py @argv *>&1 | Tee-Object -FilePath $log -Append
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} else {
    try {
        & $py @argv *>&1 | Tee-Object -FilePath $log -Append
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
}
Write-Host ''
Write-Host "exited with $code" -ForegroundColor Yellow
Write-Host "log: $log" -ForegroundColor DarkGray
exit $code
