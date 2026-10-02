# Start the OpenAI-compatible vLLM server with the configuration that measured
# best in the WSL<->Windows A/B (see _dev/docs/AB_WSL_VS_WINDOWS.md).
#
# Lives in the repo ROOT (moved out of _dev\bin on 2026-09-22 so it can be run
# without cd-ing first), and every path inside is absolute, so it works from any
# cwd.  Nothing else moved: _env.ps1 stays in _dev\bin, the run logs stay in
# _dev\out\logs.
#
#   & D:\code\vllm-windows\start_server.ps1  # 27B W4A16 + DFlash2 + vision, 71,680 ctx
#   .\start_server.ps1 -TextOnly             # drop the vision tower (a bit more KV)
#   .\start_server.ps1 -Spec mtp             # MTP instead of DFlash2 (better at high
#                                            # concurrency, worse for a single stream)
#   .\start_server.ps1 -Spec none            # no speculative decoding
#   .\start_server.ps1 -Port 8080
#   .\start_server.ps1 -Gpu 1             # pin the card; -Gpu '' (default) inherits the
#                                          # caller's CUDA_VISIBLE_DEVICES, -Gpu '0,1' for TP2.
#                                          # Stop that same service with: stop_vllm.ps1 -Gpu 1
#   .\start_server.ps1 -NoAsync              # back to --no-async-scheduling. async is
#                                            # the default since 2026-09-22 (~10%
#                                            # faster decode); -Async is kept as an
#                                            # alias for the old opt-in.
#   .\start_server.ps1 -DryRun               # print argv, start nothing (and kill
#                                            # nothing either -- not even with -Force)
#   .\start_server.ps1 -Force                # kill whatever holds 29550 / the port.
#                                            # Note it only kills the *listener*: if
#                                            # you kill a live API server this way its
#                                            # EngineCore survives holding the card,
#                                            # so clean that up too.
#   .\start_server.ps1 -NoLog                # do not write a run log
#   .\start_server.ps1 -LogFile D:\x.log     # or -LogDir <dir> to move the whole set
#
# Every run tees to _dev\out\logs\serve_<yyyyMMdd_HHmmss>.log (UTF-8, no BOM), headed by
# the exact invocation, the repo HEAD and the resolved argv. The 2026-09-21 hang
# was only diagnosable because the terminal scrollback still existed -- the
# config that caused it was nowhere on disk, and killing the process destroyed
# the evidence. Follow a run from another terminal with
#     Get-Content D:\code\vllm-windows\_dev\out\logs\serve_*.log -Wait -Tail 40
#
# Known flaky at startup: on 2026-09-20/21 about one launch in three died during
# DFlash2's CUDA graph capture with
#     torch.AcceleratorError: CUDA error: device-side assert triggered
#     speculator.py:147 -> cudagraph.py:123 -> torch.cuda.graph()
# It never happened twice in a row and a plain re-run came up fine (160 s to
# healthy). The 2026-09-22 root cause below plausibly covers it too -- capture
# executes the graph once, so it walks the same top-k path -- but that is a
# hypothesis, not a measurement. Either way, if you see it, re-run the script --
# after making sure no stale EngineCore is holding port 29550, which the
# pre-flight below checks for you.
#
# What this pins down, and why each piece is load-bearing:
#
#   * python patches are applied first, idempotently:
#       port_wsl_patches.py --batch 3   restores the "pad the drafter's 5
#                                       sliding-window layers" bucketing that
#                                       0.29 dropped. Without it 71,680 ctx
#                                       needs 6.37 GiB of KV and does not fit
#                                       the 5.4 GiB pin; with it the server
#                                       reports 72,785 tokens at 71,680 ctx
#                                       (WSL 0.28 reports 75,181 -- its pin is
#                                       bigger, and a bigger pin is not free:
#                                       see -KvBytes below).
#       fix_winloop_import.py           upstream imports uvloop unconditionally.
#   * --kv-cache-memory-bytes is a HARD cap and skips VRAM profiling, so
#     --gpu-memory-utilization is nominal. The pin plus ~15.8 GiB of weights
#     must still fit; the pre-flight below checks that.
#   * INT8 activations (VLLM_MARLIN_INPUT_DTYPE) come straight from the WSL
#     .env: +95% prefill / -4% decode on the 3090, at ~36-143 s more init.
#
#   * Vision + 71,680 under --async-scheduling used to crash. Measured
#     2026-09-20: with the tower enabled both attempts died with
#     `CUDA error: device-side assert triggered` as soon as two streams ran
#     concurrently, while --language-model-only at the same context and the tower
#     at 60,928 both ran C1-C8 clean; memory ledgers were identical, so not OOM.
#     ROOT CAUSE FOUND AND FIXED 2026-09-22: flashinfer's top_k returns garbage
#     indices when its graph is replayed, and the drafter captures the
#     candidate-selection graph -- the bad id then hit the codebook gather at
#     qwen3_dflash2.py:181. It is off by default now (torch.topk); see
#     fix_flashinfer_topk_graph_replay.py. Re-verified on the fixed tree:
#     vision + 71,680 + async ran the C1-C8 ladder 4x with zero asserts and zero
#     [clamp] hits, same boot fingerprint as the no-async path.
#   * The context guard on the vision path is kept, but re-scoped to the thing it
#     actually guarded: it now fires only when DFLASH2_TOPK_IMPL=flashinfer turns
#     the buggy path back on. -AllowUnstable still overrides it.
#
# Two more combinations are known-bad and the script refuses them rather than
# starting a server that silently runs at a quarter speed:
#   - MTP with a long context at the default pin (the batch-3 patch is a no-op
#     under MTP, so 71,680 does not fit; raising the pin to 7e9 leaves 196 MiB
#     free and ITL goes 28.7 -> 128.8 ms with no error in the log).
#   - --mamba-ssm-cache-dtype float16 removed while asking for 71,680: that
#     enables the fused GDN kernel but shrinks the ceiling to 52,416.

[CmdletBinding()]
param(
    [string]$Model  = 'D:\models\Qwen3.8-27B-W4A16-AutoRound-fast',
    [string]$Drafter = 'D:\models\Qwen3.8-27B-DFlash2-W4A16',
    [string]$ServedName = 'qwen3.8-27b',
    [ValidateSet('dflash', 'mtp', 'none')]
    [string]$Spec = 'dflash',
    [int]$MaxLen = 71680,
    # 5.8e9, not 6.0e9. Both fit on paper (75,181 vs 72,785 tokens) but only this
    # one stays fast: at 6.0e9 the extra 195 MiB of pre-reserved KV makes WDDM
    # leave part of the *weights* in host memory -- `Local` still reports the
    # same figure, so the counters look fine -- and each decode step faults
    # ~100 MB back over PCIe. Measured 2026-09-22 20:37: p50 85.4 ms/step
    # (rx 312-1,208 MB/s sustained) vs 26.7 ms here (rx -> 0). See §16.8c.
    [long]$KvBytes = 5800000000,
    [int]$Port = 8000,
    [int]$MaxSeqs = 8,
    # Tensor-parallel size. 1 (default) is the only mode measured on this rig.
    # >1 flips VLLM_ENABLE_V1_MULTIPROCESSING back to 1 (TP needs worker
    # subprocesses; the in-process EngineCore this script normally runs cannot
    # host them) and expects a working NCCL: on Windows that means the
    # SystemPanic/nccl-windows build reachable via VLLM_NCCL_SO_PATH --
    # pynccl ignores is_nccl_available() and dlopens the dll directly.
    # NEVER measured on the 2x CMP 170HX rig; treat as experimental.
    [int]$TP = 1,
    # Card selection. '' (default) inherits whatever the caller exported, which is
    # how the 170HX launcher pins a card today; '0'/'1' overrides it, '0,1' exposes
    # both for -TP 2. Also decides which card the VRAM pre-flight reads.
    [string]$Gpu = '',
    [int]$BatchedTokens = 2048,
    [double]$MemUtil = 0.93,
    [int]$KeepAlive = 30,
    # CUDA-graph capture ceiling, on by default as of 2026-09-22 because it
    # bought 0.22 GiB (1.30 -> 1.08 GiB, measured at the 5-6 s mark).
    # 0 = derive from -MaxSeqs, which is what you want: the graph-length list is
    # [1,2,4] + range(8,256,8) + range(256,N+1,16), so with -MaxSeqs 8 nothing
    # above a decode batch of 8 is reachable and sizes 16-64 are dead weight.
    # Passing a number explicitly still wins (e.g. an A/B that forces 64).
    [int]$MaxGraphCapture = 0,
    # Size of the CPU KV offload pool in GiB. 0 (default) disables offloading.
    #
    # Old context is moved to pinned host memory on eviction and copied back on
    # a prefix hit, so a multi-turn / multi-agent workload stops re-prefilling
    # the whole conversation. Windows cannot use the stock backend: its
    # CPUOffloadingSpec mmaps /dev/shm (Linux-only) and moves blocks with
    # cuMemcpyBatchAsync, which fails with CUDA_ERROR_INVALID_VALUE under WDDM.
    # On this platform OffloadingConnector is therefore routed to
    # PinnedCPUOffloadingSpec (vllm/v1/kv_offload/cpu/pinned_spec.py), which
    # allocates plain pinned tensors and copies with non_blocking=True on a
    # dedicated stream (~1.5 GB/s measured on this card).
    #
    # Offloaded blocks live in host RAM, not VRAM, so this does not shrink
    # -KvBytes -- it trades system memory for avoided prefill. Transfers run on
    # a low-priority side stream and stores are deferred to the next engine
    # step, so steady-state decode should not pay for them; measure ITL/TTFT
    # against a run with -KvOffloadGiB 0 before trusting that.
    [double]$KvOffloadGiB = 0.0,
    # Release the vision tower's device copy right after each encode instead of
    # leaving it in the caching allocator for the life of the process
    # (patches/fix_offload_release.py). Measured 0.86 GiB of the card never
    # comes back without it, which is what made vision + long context a
    # trade-off at all -- so this is ON by default as of 2026-09-22.
    # Turn it off with -ReleaseOffloadCopy:$false for an A/B against upstream.
    [switch]$ReleaseOffloadCopy = $true,
    # KV cache dtype. bfloat16 is the measured baseline; the *_per_token_head /
    # turboquant_* values shrink the pool so a longer -MaxLen fits beside a
    # vision tower that lands ~1.3 GiB on the card. They are NOT free: see the
    # -KvDtype note above $argv.
    [ValidateSet('bfloat16', 'int8_per_token_head', 'int4_per_token_head',
                 'fp8_per_token_head', 'turboquant_k3v4_nc', 'turboquant_4bit_nc')]
    [string]$KvDtype = 'bfloat16',
    # Leave empty to pair it automatically with -KvDtype. Only set this to run an
    # A/B (e.g. bfloat16 + TRITON_ATTN to isolate backend from dtype).
    [ValidateSet('', 'FLASH_ATTN', 'TRITON_ATTN', 'TURBOQUANT')]
    [string]$AttnBackend = '',
    [switch]$TextOnly,
    [switch]$NoVisionOffload,
    [switch]$FusedGdn,
    [switch]$NoInt8,
    [switch]$NoPatches,
    [switch]$Async,
    [switch]$NoAsync,
    [switch]$AllowUnstable,
    [switch]$Force,
    [string]$LogDir  = '',
    [string]$LogFile = '',
    [switch]$NoLog,
    [switch]$DryRun
)

$ErrorActionPreference = 'Continue'
$REPO = 'D:\code\vllm-windows'
$DEV  = "$REPO\_dev"
. "$DEV\bin\_env.ps1"

$py = "$REPO\.venv\Scripts\python.exe"
$SITEVLLM = "$REPO\.venv\Lib\site-packages\vllm"
$PATCHES = "$DEV\patches"
$KvGiB = [math]::Round($KvBytes / 1GB, 2)

# ------------------------------------------------------------------ run log --
# PowerShell decodes a native command's stdout using [Console]::OutputEncoding.
# On a zh-CN console that is GBK, so vLLM's UTF-8 banner reaches the console --
# and anything teed to a file -- as `鈻堚枅` (measured: 266 bytes of mojibake
# against a byte-exact 247). Force UTF-8 before the first byte is produced; it
# also fixes the banner's rendering on the console itself. $OutputEncoding
# defaults to ASCII in 5.1 and is the belt to that pair of braces -- it is also
# the only one of the two that can be set when there is no console attached.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$OutputEncoding = [System.Text.Encoding]::UTF8

$script:LogPath = ''
if (-not $NoLog -and -not $DryRun) {
    if ($LogFile) {
        $script:LogPath = $LogFile
    } else {
        if (-not $LogDir) { $LogDir = "$DEV\out\logs" }
        $script:LogPath = Join-Path $LogDir ("serve_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    }
    $parent = Split-Path -Parent $script:LogPath
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

# Tee-Object rather than Out-File/Add-Content: it is the one that writes UTF-8
# with no BOM *and* streams line by line, so the file can be followed during the
# ~160 s startup. It also passes its input through, so never assign a call's
# output -- the exit code comes back in $script:LastRc instead.
function Write-Log([object[]]$Lines) {
    if (-not $script:LogPath) { return }
    $Lines | ForEach-Object { "$_" } | Tee-Object -FilePath $script:LogPath -Append | Out-Null
}

function Invoke-Logged {
    param([string]$Exe, [string[]]$Arguments)
    if ($script:LogPath) {
        & $Exe @Arguments *>&1 | Tee-Object -FilePath $script:LogPath -Append
    } else {
        & $Exe @Arguments
    }
    $script:LastRc = $LASTEXITCODE
}

if ($script:LogPath) {
    $bound = ($PSBoundParameters.GetEnumerator() | Sort-Object Key |
              ForEach-Object { "-$($_.Key) $($_.Value)" }) -join ' '
    $head = @(
        '==== vllm-windows serve ===='
        "started_at  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        "host        : $env:COMPUTERNAME  (powershell pid $PID)"
        "shell       : $([System.Diagnostics.Process]::GetCurrentProcess().ProcessName) $($PSVersionTable.PSVersion)"
        "invocation  : .\start_server.ps1 $bound"
        "log         : $script:LogPath"
    )
    try {
        $h = & git -C $REPO rev-parse --short HEAD 2>$null
        if ($h) { $head += "repo HEAD   : $h" }
    } catch { }
    $head += ''
    Write-Log $head
}

function Fail([string]$msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    Write-Log @("ERROR: $msg")
    exit 1
}

# ---------------------------------------------------------------- pre-flight --
foreach ($d in @($Model)) {
    if (-not (Test-Path -LiteralPath $d -PathType Container)) { Fail "model dir not found: $d" }
}
if ($Spec -eq 'dflash' -and -not (Test-Path -LiteralPath $Drafter -PathType Container)) {
    Fail "drafter dir not found: $Drafter"
}

# A killed EngineCore keeps 127.0.0.1:29550 bound and the next start dies with
# `zmq.error.ZMQError: Address in use`. Catch it here, not 60 s into startup.
foreach ($p in 29550, $Port) {
    $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $pids = ($conn | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
        if ($DryRun) {
            # -DryRun is documented as "print argv, start nothing" and must not
            # touch a single process.  It used to fall through to the -Force
            # branch below: `-Force -DryRun` killed the live API server and left
            # its EngineCore orphaned with 23 GiB of the card (2026-09-22).
            Write-Host "port $p held by PID $pids (-DryRun: reporting only, nothing killed)" -ForegroundColor DarkGray
        } elseif ($Force) {
            Write-Host "port $p held by PID $pids - killing (-Force)" -ForegroundColor Yellow
            $pids -split ', ' | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 2
        } else {
            Fail "port $p is already in use by PID $pids`n       (a leftover EngineCore? re-run with -Force to kill it)"
        }
    }
}
# -Force kills the LISTENER, and for vLLM the listener is the API server while
# the EngineCore (`spawn_main`, its child) survives -- still holding the whole
# KV pool. The next start then comes up in the paging state (85 ms/step, rx
# streaming) with no error anywhere. Report it rather than killing silently:
# it is a separate process and could belong to another server.
if ($Force -and -not $DryRun) {
    $stray = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'spawn_main|vllm\.entrypoints\.openai\.api_server' } |
        Select-Object -ExpandProperty ProcessId)
    if ($stray.Count) {
        Write-Host "WARNING: vLLM process(es) survived the port kill: $($stray -join ', ')" -ForegroundColor Yellow
        Write-Host "         the orphaned EngineCore keeps the KV pool on the card, so the next" -ForegroundColor Yellow
        Write-Host "         start would come up paging. Clear them first:" -ForegroundColor Yellow
        Write-Host "           Stop-Process -Id $($stray -join ',') -Force" -ForegroundColor Yellow
    }
}

# --kv-cache-memory-bytes skips profiling, so nothing in vLLM will tell you the
# card is too full -- it just thrashes the allocator. Check it ourselves.
#
# 2026-09-22: the original `15.8 + KvGiB + 0.7` under-counted, and the whole
# check was comparing MiB against GiB so it never fired either (see below).  Card
# -side non-KV cost, measured with the tower offloaded to RAM:
#     ~17.8 GiB  weights + CUDA graphs + workspace, tower not yet on the card
#     +0.9 GiB   the tower's upload, once the first image has been encoded
#                (1/2/8 full-budget images land within 0.5 GiB of each other, so
#                 it saturates rather than growing with the image count)
#     +~0.4 GiB  the vision encoder's own workspace, same first-image trigger
# The last two are what turned a 4.2 vs 6.0 GiB pin into 26.6 vs 104.6 ms/step.
# ------------------------------------------------------- which card? --
# The VRAM check used to read GPU0 no matter where the service was actually going,
# so a run pinned to the busy card could pass a check made against the empty one.
if ($Gpu) { $env:CUDA_VISIBLE_DEVICES = $Gpu }
$visible = if ($env:CUDA_VISIBLE_DEVICES) {
    @($env:CUDA_VISIBLE_DEVICES -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
} else { @('0') }
$checkId = $visible[0]
if ($TP -gt 1 -and $Gpu -and $visible.Count -lt $TP) {
    Fail "-Gpu '$Gpu' exposes $($visible.Count) card(s) but -TP $TP needs $TP.`n       Pass a list instead: -Gpu '0,1'."
}

try {
    $csv  = & nvidia-smi --id="$checkId" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
    $f    = ($csv | Select-Object -First 1) -split ','
    $free = [double]$f[0] - [double]$f[1]        # MiB -- nvidia-smi --nounits
    # UNITS (fixed 2026-09-22): $free is MiB while $need below is GiB, and the
    # two used to be compared directly -- 24000 < 25.3 is always false, so this
    # guard never once fired. The 6 GiB / 71,680 run that thrashed the allocator
    # therefore started with no warning at all.
    #
    # Card-side demand only: the ~1.4 GiB of `Non Local` in the GPU-process
    # counters is host-pinned memory, not card memory (WSL shows 16.5 GiB of it
    # for a 16 GiB /dev/shm KV tier and still runs at 26.5 ms/step).
    # Base = W4 weights (15,060 MiB main + 1,221 drafter) + CUDA graphs
    # (1,080 at the derived ceiling, 1,330 if you force 64) + workspace. The
    # tower's upload adds 0.88 GiB;
    # -ReleaseOffloadCopy makes that transient, but it is on the card while an
    # image encodes either way, so it is still counted here.
    $baseGiB   = 17.8
    $visionGiB = if ($TextOnly) { 0.0 } else { 0.9 }
    $need   = $baseGiB + ($(if ($ReleaseOffloadCopy) { 0.0 } else { $visionGiB })) + $KvGiB
    $needPeak = $baseGiB + $visionGiB + $KvGiB
    $freeGiB = $free / 1024
    if ($freeGiB -lt $need) {
        Write-Host "WARNING: only $([math]::Round($freeGiB,1)) GiB free, this config wants ~$([math]::Round($need,1)) GiB." -ForegroundColor Yellow
        Write-Host "         Lower -KvBytes or -MaxLen, or free the card first." -ForegroundColor Yellow
    }
    if ($ReleaseOffloadCopy -and -not $TextOnly -and $freeGiB -lt $needPeak) {
        Write-Host "         the tower's copy is released after each encode, so only the steady" -ForegroundColor DarkGray
        Write-Host "         figure has to fit; the encode-time peak is ~$([math]::Round($needPeak,1)) GiB (Local 24,439 with the" -ForegroundColor DarkGray
        Write-Host "         tower resident measured fine at p50 26.75 ms -- see skill 16.9)." -ForegroundColor DarkGray
    }
} catch { Write-Host "nvidia-smi unavailable, skipping the VRAM check" -ForegroundColor Yellow }

# --------------------------------------------------- async scheduling default --
# async is the DEFAULT since 2026-09-22. The crash that forced the old
# --no-async-scheduling default is fixed -- see the 2026-09-22 notes at the top
# of this file -- and was re-verified with the tower at 71,680 (C1-C8 x4,
# 0 asserts, 0 [clamp] hits). Healthy decode is ~10% faster than the
# --no-async-scheduling path (29 vs 27 ms/step); the boot fingerprint is the
# same either way (fingerprint 2026-09-22: 72,785 tokens / 1.02x, 2-5 s / 1.08 GiB).
#   -NoAsync            -> WSL's own path (--no-async-scheduling, ASYNC_SCHED=0)
#   -Async              -> kept so older invocations and repro scripts still work
#   -Async:$false       -> same as -NoAsync
if ($NoAsync) {
    $Async = $false
} elseif (-not $PSBoundParameters.ContainsKey('Async')) {
    $Async = $true
}

# -------------------------------------------------- CUDA-graph capture ceiling --
# Resolve the derived default now that -MaxSeqs is known. The capture ladder's
# step is 8 up to 256, then 16, so round MaxSeqs up onto it; never go below 8
# (1,2,4 are always captured) and cap at 64, the old hard default, because
# anything larger is a deliberate choice and should be passed explicitly.
if ($MaxGraphCapture -le 0) {
    $MaxGraphCapture = [int][math]::Min(64, [math]::Max(8, [math]::Ceiling($MaxSeqs / 8.0) * 8))
}

# --------------------------------------------------- context/pin interaction --
# The guard that used to gate -Async is kept, but re-scoped to the bug it was
# actually guarding instead of to the flag: it fires only when the fixed top-k
# is bypassed on purpose (DFLASH2_TOPK_IMPL=flashinfer, for A/B work).
$topkImpl = if ($env:DFLASH2_TOPK_IMPL) { $env:DFLASH2_TOPK_IMPL } else { 'torch.topk' }
if (-not $TextOnly -and $Async -and $topkImpl -eq 'flashinfer') {
    if ($MaxLen -gt 60928 -and -not $AllowUnstable) {
        if ($PSBoundParameters.ContainsKey('MaxLen')) {
            Fail "DFLASH2_TOPK_IMPL=flashinfer re-enables the 2.2c crash, so vision + $MaxLen is refused under --async-scheduling.`n       Options: drop that env var (torch.topk is the fixed default), -MaxLen 60928, -TextOnly, or -AllowUnstable to try anyway."
        }
        Write-Host "vision + Async + flashinfer top-k: defaulting context to 60,928 (not needed on the fixed torch.topk path)" -ForegroundColor Yellow
        $MaxLen = 60928
    }
}
# The batch-3 patch returns early under MTP, so it buys nothing there.
if ($Spec -eq 'mtp') {
    if ($MaxLen -gt 60928 -and -not $PSBoundParameters.ContainsKey('KvBytes')) {
        if ($PSBoundParameters.ContainsKey('MaxLen')) {
            Fail "MTP + $MaxLen ctx needs a bigger pin than the default 5.4 GiB.`n       Either drop to -MaxLen 60928, or pass -KvBytes explicitly -- but note a bigger`n       pin is not just a capacity knob: 6.0e9 at the same context measured 85.4 ms/step`n       (paging, rx 1,208 MB/s) where 5.8e9 measured 26.7 ms. See the -KvBytes note."
        }
        Write-Host "MTP: patch is a no-op here, lowering context to 60,928" -ForegroundColor Yellow
        $MaxLen = 60928
    }
}
# Dropping float16 enables the fused GDN kernel but caps the context at 52,416.
if ($FusedGdn -and -not $PSBoundParameters.ContainsKey('MaxLen')) {
    Write-Host "FusedGdn: context ceiling drops to 52,416" -ForegroundColor Yellow
    $MaxLen = 52416
}

# ------------------------------------------------------------------- patches --
if (-not $NoPatches -and -not $DryRun) {
    Write-Host "applying python patches (idempotent)..."
    Write-Log @('', '--- python patches ---')
    # Both fix_*.py take the vllm package dirs as arguments and exit 2 without
    # them -- the bare call here silently no-op'd (and tripped the warning).
    # Every step's rc is checked now, not just the last one's.
    $rcs = @()
    Invoke-Logged $py -Arguments @("$PATCHES\port_wsl_patches.py", '--batch', '3'); $rcs += $script:LastRc
    Invoke-Logged $py -Arguments @("$PATCHES\fix_winloop_import.py", $SITEVLLM, "$REPO\vllm"); $rcs += $script:LastRc
    Invoke-Logged $py -Arguments @("$PATCHES\fix_responses_toolcall_none_name.py", $SITEVLLM, "$REPO\vllm"); $rcs += $script:LastRc
    # Runtime behaviour is gated on VLLM_OFFLOAD_RELEASE_AFTER_FORWARD, so the
    # patch itself is inert until the env var below is set (it is, by default).
    Invoke-Logged $py -Arguments @("$PATCHES\fix_offload_release.py", $SITEVLLM, "$REPO\vllm"); $rcs += $script:LastRc
    $bad = $rcs | Where-Object { $_ -ne 0 }
    if ($bad) { Write-Host "WARNING: a patch step returned $($bad -join ', ')" -ForegroundColor Yellow }
}

# ----------------------------------------------------------------- env vars --
$env:PATH = "$REPO\.venv\Scripts;$env:PATH"
# TP>1 needs worker subprocesses, so the in-process EngineCore default goes.
$env:VLLM_ENABLE_V1_MULTIPROCESSING = if ($TP -gt 1) { '1' } else { '0' }
$env:VLLM_USE_FLASHINFER_SAMPLER    = '0'   # no flashinfer JIT sampler
$env:HF_HUB_DISABLE_SYMLINKS_WARNING= '1'
$env:PYTHONUNBUFFERED = '1'
# GBK console makes vLLM's banner throw inside the logging handler.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8       = '1'
$env:TORCH_CUDA_ARCH_LIST = '8.0'
$env:VLLM_TARGET_DEVICE   = 'cuda'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:False'
$env:VLLM_SPEC_DECODE_ATTN  = '1'
# Set by default (see -ReleaseOffloadCopy). Read with os.environ at call time,
# so it is a plain on/off switch for the patched offloader -- and deleting the
# variable restores stock behaviour without re-patching anything.
if ($ReleaseOffloadCopy) {
    $env:VLLM_OFFLOAD_RELEASE_AFTER_FORWARD = '1'
} else {
    Remove-Item Env:VLLM_OFFLOAD_RELEASE_AFTER_FORWARD -ErrorAction SilentlyContinue
}
if (-not $NoInt8) {
    $env:VLLM_MARLIN_INPUT_DTYPE    = 'int8'
    $env:VLLM_MARLIN_INT8_INCLUDE_RE = 'mlp'
}

# --------------------------------------------------------------- serve argv --
# `python -m vllm.entrypoints.openai.api_server` is deprecated in 0.29 ("use
# `vllm server` instead") but it is the entrypoint every measurement here used,
# so it stays until `vllm server` is re-validated.
# --- KV dtype -> attention backend -------------------------------------------
# These two are not independent. The per-token-head int8/int4/fp8 KV quantizers
# live in TRITON_ATTN (`v1/attention/backends/triton_attn.py`, gated on
# `_is_per_token_head_quant`) and the turboquant packers in
# TurboQuantAttentionBackend. Pairing the wrong two either refuses to boot or
# silently runs an unquantized kernel.
#
# Cost of the int8 tier, measured on WSL with this exact model (gotcha 40):
# 136,429 tokens of pool in a 5.2 GiB pin against bf16's 69,758 -- 2x the
# context -- but -34% decode / -44% prefill at 60k, and +2.5% end-to-end at
# chat length. That is the trade that buys 71680 + vision on a 24 GiB card.
if (-not $AttnBackend) {
    $AttnBackend = switch -Wildcard ($KvDtype) {
        'bfloat16'         { 'FLASH_ATTN' }
        '*_per_token_head' { 'TRITON_ATTN' }
        'turboquant_*'     { 'TURBOQUANT' }
        default            { 'FLASH_ATTN' }
    }
}

if ($KvDtype -ne 'bfloat16') {
    Write-Host "kv dtype       : $KvDtype  ->  --attention-backend $AttnBackend" -ForegroundColor Cyan
    Write-Host "  Quantized KV pays off only if -KvBytes comes down with it: the pin is" -ForegroundColor DarkGray
    Write-Host "  sized in bytes, so leaving it at 6e9 spends the same VRAM for 2x the context." -ForegroundColor DarkGray
}

$argv = @(
    '-u', '-m', 'vllm.entrypoints.openai.api_server',
    '--model', $Model,
    '--served-model-name', $ServedName,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--trust-remote-code',
    '--gpu-memory-utilization', "$MemUtil",
    '--max-model-len', "$MaxLen",
    '--max-num-seqs', "$MaxSeqs",
    '--max-num-batched-tokens', "$BatchedTokens",
    '--attention-backend', $AttnBackend,
    '--kv-cache-dtype', $KvDtype,
    '--enable-prefix-caching',
    '--mamba-cache-mode', 'align',
    '--enable-prompt-tokens-details',
    '--reasoning-parser', 'qwen3',
    '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
    '--default-chat-template-kwargs', '{"enable_thinking": false}',
    # Double-quoted on purpose (and the same idiom as --speculative-config
    # below): the capture ceiling is a parameter now, so the JSON has to be
    # built rather than written out.
    '--compilation-config', "{""max_cudagraph_capture_size"":$MaxGraphCapture,""custom_ops"":[""+rms_norm"",""+silu_and_mul""]}",
    '--kv-cache-memory-bytes', "$KvBytes",
    '--sse-keep-alive-interval', "$KeepAlive"
    # SSE comments so proxies do not drop a long prefill. 30 is what the A/B
    # ran; that value made *our bench client* abort requests with TTFT > 30 s,
    # which is client-side, not server. Set 0 to disable, or raise it if a
    # full-length prefill still times out.
)
# CPU KV offloading (see the -KvOffloadGiB note). The backend is "native" so
# that CacheConfig post-init resolves the connector; on Windows that resolves to
# OffloadingConnector + PinnedCPUOffloadingSpec via vllm/config/vllm.py.
if ($KvOffloadGiB -gt 0) {
    $argv += @('--kv-offloading-size', "$KvOffloadGiB",
               '--kv-offloading-backend', 'native')
}
# async by default since 2026-09-22: ~10% faster healthy decode, and the crash
# that forced the --no-async-scheduling default is fixed (see the header notes).
# -NoAsync restores WSL's ASYNC_SCHED=0 path.
if ($Async) { $argv += '--async-scheduling' } else { $argv += '--no-async-scheduling' }

if ($TP -gt 1) { $argv += @('--tensor-parallel-size', "$TP") }

if ($TextOnly) {
    $argv += '--language-model-only'
} else {
    $argv += @('--limit-mm-per-prompt', '{"image":{"count":32}}')
    $argv += @('--mm-processor-kwargs', '{"size":{"shortest_edge":65536,"longest_edge":1048576}}')
}

# float16 turns OFF the fused CUDA GDN decode kernel (Triton fallback), but it
# is what keeps 71,680 in reach -- the fused kernel's cache costs more per token.
if (-not $FusedGdn) { $argv += @('--mamba-ssm-cache-dtype', 'float16') }

switch ($Spec) {
    'dflash' {
        # Paths go into a JSON string, so backslashes have to become forward
        # slashes first -- `D:\models\...` makes json.loads() reject the whole
        # --speculative-config with "cannot be converted to <function loads>".
        $draftJson = $Drafter -replace '\\', '/'
        $argv += @('--speculative-config',
            "{""method"": ""dflash"", ""model"": ""$draftJson"", ""num_speculative_tokens"": 7, ""draft_sample_method"": ""probabilistic""}")
    }
    'mtp' {
        # single-quoted on purpose: a double-quoted PS string would need "" for
        # each literal quote, and getting that wrong ships doubled quotes to vLLM
        $argv += @('--speculative-config', '{"method": "mtp", "num_speculative_tokens": 1}')
    }
}

# 0.29's upstream equivalent of the WSL vision-tower-cpu-offload patch:
# --cpu-offload-gb + --cpu-offload-params, with UVA disabled (bulk copy).
# Must stay last in argv: --cpu-offload-params is nargs='+'.
if (-not $TextOnly -and -not $NoVisionOffload) {
    $env:VLLM_WEIGHT_OFFLOADING_DISABLE_UVA = '1'
    $argv += @('--cpu-offload-gb', '1', '--cpu-offload-params', 'visual')
}

# Convention 5: never run from the repo root -- the checked-out vllm\ tree
# would shadow the installed package and the built extension modules.  The
# switch happens at the launch below, not here, so -DryRun leaves the caller's
# shell alone.

# Capacity is NOT proportional to the pin, and it also moves with -MaxLen.
# Five measured anchors (all with this model, bf16, -MaxSeqs 8):
#     4.2e9 / 48,000 -> 48,168      6.0e9 / 60,928 -> 72,899
#     6.0e9 / 71,680 -> 75,181      5.8e9 / 70,000 -> 72,193
#     5.8e9 / 71,680 -> 72,785   <- the 2026-09-22 default, vision verified
# Fitting tokens = (KvBytes - 1.091e9) / 81,859 + 0.2122 * MaxLen reproduces all
# five to within ~200 tokens, so use it instead of the old hard-coded 75,181 --
# which is only right for one config and made every other run look surprising.
# Do not read the pin as a capacity dial without reading the -KvBytes note: the
# biggest pin here is also the slowest, by 3.2x.
if ($KvDtype -ne 'bfloat16') {
    $expectKv = 'not comparable -- quantized KV packs differently (WSL measured 136,429 tokens in a 5.2 GiB pin)'
} elseif ($Spec -eq 'mtp') {
    $expectKv = '~61,303 (patch is a no-op under MTP; not measured)'
} elseif ($FusedGdn) {
    $expectKv = '52,416 (fused GDN cache costs more per token)'
} else {
    $estKv = [int][math]::Floor(($KvBytes - 1091000000) / 81859 + 0.2122 * $MaxLen)
    $expectKv = "$('{0:N0}' -f $estKv) tokens (fit to the 3 measured anchors; the pin is not linear and -MaxLen moves it too)"
}
# The same lines go to the console and into the run log: post-mortems need to
# know which effective config produced a run, and the guards above may have
# overridden what was asked for.
# 2026-09-21: DFLASH2_IDCHECK was invisible to post-mortems -- the env dump
# below only prints VLLM_/PYTHON/TORCH_/... prefixed names, so a repro run that
# never armed the probe looked exactly like a clean run (both show zero
# `[idcheck]` lines).  Report the probe + blocking state in the header, and add
# both to the env whitelist.
$idkState = if ($env:DFLASH2_IDCHECK -eq '1') {
    "on (every $(if ($env:DFLASH2_IDCHECK_EVERY) { $env:DFLASH2_IDCHECK_EVERY } else { 25 }) steps)"
} else { 'off' }
$info = @(
    ''
    "  model      $Model"
    "  spec       $Spec$(if ($Spec -eq 'dflash') { " -> $Drafter" })"
    "  context    $MaxLen tokens, KV pin $KvGiB GiB"
    "  vision     $(if ($TextOnly) { 'off (--language-model-only)' } else { "on$(if (-not $NoVisionOffload) { ', tower offloaded to RAM' })" })"
    "  int8 act   $(if ($NoInt8) { 'off' } else { 'on (mlp)' })   fused GDN $(if ($FusedGdn) { 'on' } else { 'off (float16 cache)' })"
    "  graphs     max_cudagraph_capture_size $MaxGraphCapture$(if ($MaxGraphCapture -lt 64) { "  (upstream default here is 64, but -MaxSeqs $MaxSeqs caps decode batches at $MaxSeqs)" })"
    "  vision copy $(if ($TextOnly) { 'n/a (text only)' } elseif ($NoVisionOffload) { 'tower stays on the card (~0.9 GiB resident)' } elseif ($ReleaseOffloadCopy) { 'released after each encode (default; -ReleaseOffloadCopy:$false to keep it)' } else { 'stays in the allocator -- 0.9 GiB gone for the life of the process' })"
    "  async      $(if ($Async) { 'on (--async-scheduling, default)' } else { 'off (--no-async-scheduling, -NoAsync)' })"
    "  tp         $TP$(if ($TP -gt 1) { '  (EXPERIMENTAL on this rig; V1 multiproc re-enabled)' })"
    "  gpu        CUDA_VISIBLE_DEVICES=$(if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { '(unset -> device 0)' })   VRAM check read card $checkId"
    "  probe      idcheck $idkState   CUDA_LAUNCH_BLOCKING $(if ($env:CUDA_LAUNCH_BLOCKING) { $env:CUDA_LAUNCH_BLOCKING } else { 'off' })   draft CG $(if ($env:DFLASH2_NO_DRAFT_CUDAGRAPH -eq '1') { 'off (eager, diagnostic)' } else { 'on' })   clamp $(if ($env:DFLASH2_CLAMP -eq '1') { 'ON (in-graph guard)' } else { 'off' })"
    "  topk       $(if ($topkImpl -eq 'flashinfer') { 'flashinfer radix (fast, UNSAFE in the drafter graph -- root cause of 2.2c; the context guard is armed)' } else { 'torch.topk (safe; see fix_flashinfer_topk_graph_replay.py)' })"
    "  endpoint   http://127.0.0.1:$Port/v1"
    "  log        $(if ($script:LogPath) { $script:LogPath } else { '(disabled)' })"
    ''
    "expect in the log:  'GPU KV cache size: $expectKv'"
)
if ($Spec -ne 'mtp') {
    $info += "                    'Sliding-window bucket(s) are the smallest'  <- patch active"
}
$info += "                    'Graph capturing finished in ~5-6 secs, took $(if ($MaxGraphCapture -lt 64) { "less than 1.30 GiB -- ceiling lowered to $MaxGraphCapture" } else { '1.30 GiB' })'"
$info += "  capture sanity: took >>10s, or GiB far below the figure above = silent partial capture;"
$info += "  decode will run 5-15x slow with NO error logged -> restart the server."
$info += ''
$info | ForEach-Object { Write-Host $_ }
Write-Log $info

if ($script:LogPath) {
    Write-Log @('--- argv ---', ($py + ' ' + ($argv -join ' ')))
    Write-Log @('', '--- env ---')
    Write-Log (Get-ChildItem env: |
        Where-Object { $_.Name -match '^(VLLM_|PYTHON|TORCH_|PYTORCH_|HF_|NCCL|CUDA_|DFLASH2_|VBENCH_OUT)' } |
        Sort-Object Name | ForEach-Object { "  $($_.Name)=$($_.Value)" })
    Write-Log @('', '--- server output ---')
}

if ($DryRun) {
    Write-Host 'argv:' -ForegroundColor Cyan
    Write-Host ($py + ' ' + ($argv -join ' '))
    exit 0
}

# Convention 5 again: the child inherits this shell's cwd, and vLLM must not
# start inside the repo.  Push/Pop instead of a bare Set-Location -- the latter
# outlives the script (the location is session state, not script scope), so the
# shell is left in C:\Users\hong and the next `.\start_server.ps1` dies with
# "not recognized as a name of a cmdlet".  Measured 2026-09-21.
#
# Windows PowerShell 5.1 mangles embedded double quotes when it hands argv to a
# native exe, so the JSON-valued flags (--default-chat-template-kwargs,
# --speculative-config, --compilation-config) arrive quote-stripped:
#     api_server.py: error: argument --default-chat-template-kwargs:
#       invalid loads value: '{enable_thinking: false}'
# Measured 2026-09-21 20:16 (server died in 17 s, rc=2).  pwsh 7.3+ passes them
# through, so refuse anything older.  DryRun is unaffected (it never execs
# python), hence this check sits after it.
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Fail "run this under PowerShell 7+ (pwsh), not Windows PowerShell $($PSVersionTable.PSVersion).`n       5.1 strips the quotes from --default-chat-template-kwargs and api_server exits 2.`n       Launch: pwsh -File .\start_server.ps1 ..."
}
# Report the blocking flag: the repro script (_repro_async_tower.ps1) sets it to
# get an accurate stack, and a log that does not say so is easy to misread --
# CUDA_LAUNCH_BLOCKING=1 makes every step 2-5x slower and is NOT the config any
# baseline here was measured with.
if ($env:CUDA_LAUNCH_BLOCKING) {
    Write-Host "NOTE: CUDA_LAUNCH_BLOCKING=$($env:CUDA_LAUNCH_BLOCKING) is set -- this run is for stack accuracy, not for timing." -ForegroundColor Yellow
    Write-Log @("NOTE: CUDA_LAUNCH_BLOCKING=$($env:CUDA_LAUNCH_BLOCKING) -- repro mode; decode is much slower than any baseline")
}
Push-Location 'C:\Users\hong'
try {
    Invoke-Logged $py -Arguments $argv
    $code = $script:LastRc
} finally {
    Pop-Location
}
Write-Log @('', "--- exited with $code at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ---")
if ($code -ne 0) {
    Write-Host ''
    Write-Host "exited with $code." -ForegroundColor Yellow
    Write-Host "If the log shows 'device-side assert triggered' while capturing dflash2" -ForegroundColor Yellow
    Write-Host "CUDA graphs, that step is flaky -- just re-run this script." -ForegroundColor Yellow
}
if ($script:LogPath) { Write-Host "log: $script:LogPath" -ForegroundColor DarkGray }
exit $code
