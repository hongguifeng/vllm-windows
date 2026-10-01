# Run vLLM on the CMP 170HX (GA100 / sm_80) on Windows.
#
# Usage:
# Lives in _dev\bin. Run it from anywhere -- every path below is absolute:
#
#   & D:\code\vllm-windows\_dev\bin\_run.ps1                  # end-to-end smoke test (0.5B)
#   .\ _run.ps1 -Model <hf_repo_id>          # smoke test against another model
#   .\ _run.ps1 -Model <local_dir> -TextOnly -MaxLen 16384
#                                            # local quantized checkpoint (27B script)
#   .\ _run.ps1 -Serve                       # start an OpenAI-compatible API server
#   .\ _run.ps1 -Eager                       # disable CUDA graphs / torch.compile
#   .\ _run.ps1 -W4A8                        # int8 activations on the Marlin GEMMs
#   .\ _run.ps1 -W4A8 -W4A8Layers mlp        # ...only on the MLP projections
#   .\ _run.ps1 -Script <path.py>            # run an arbitrary script with the same env
#
# W4A8 (the -W4A8 switch) is a prefill/decoding trade, measured on this card with
# Qwen3.8-27B-W4A16 at max_model_len=16384:
#
#                      prefill 8.8k   batch prefill   decode bs=1   decode bs=8   engine init
#     W4A16 (default)     2 197 t/s       2 266 t/s      66.9 t/s      462 t/s        69 s
#     W4A8 (all layers)   2 833 t/s       2 999 t/s      63.1 t/s      432 t/s       106 s
#     W4A8 (-W4A8Layers mlp) 2 602 t/s    2 735 t/s      64.8 t/s      445 t/s        85 s
#
# So: +29% prefill against -6% decode, plus a one-time ~36 s longer engine init
# (torch.compile +17 s, warmup +10 s, weight load +3 s) -- because the int8 graph
# is bigger to compile and the checkpoint gets an extra activation-scale pass.
# Decode loses because the int8 path adds a per-token activation quantization to
# every layer and an int8 GEMM has no advantage at M=1.  Pick -W4A8 when prompts
# dominate (RAG, document QA, long-context); stay on the default when generation
# dominates.  `lm_head` and `mtp` are excluded by default, matching upstream.
#
# Configuration that the build/bring-up actually needs -- every item matters:
#
#   * CUDA graphs are ON by default (i.e. enforce_eager=0).  They are worth it:
#     on Qwen3.8-27B-W4A16 this is ~73.5 tok/s versus ~15.7 tok/s eager.  Use
#     -Eager when a new model misbehaves and you want a smaller failure surface.
#   * VLLM_ENABLE_V1_MULTIPROCESSING=0
#       Skips the EngineCore ZMQ subprocess.  Without it, a run that dies
#       leaves a process bound to 127.0.0.1:<port> and the next run fails with
#       `zmq.error.ZMQError: Address in use`.
#   * VLLM_USE_FLASHINFER_SAMPLER=0
#       flashinfer JIT-compiles its top-k/top-p sampler on first use; that
#       build needs ninja + cl.exe and is fragile on Windows.  vLLM falls back
#       to the native PyTorch sampler, which is fine for inference.
#   * .venv\Scripts on PATH
#       Puts ninja/cmake in reach of the runtime JIT helpers.
#   * TORCH_CUDA_ARCH_LIST=8.0 / VLLM_TARGET_DEVICE=cuda
#       Must match how the wheel was built (sm_80 only).
#   * gpu_memory_utilization has to fit the *free* memory, not the total.
#       Default here is derived from whatever is free at launch, so this still
#       works while another process (e.g. gpu_burn.exe) holds most of the card.
#       But make sure the environment is *stable* while vLLM profiles: if some
#       other process releases VRAM mid-startup, vLLM aborts with
#       `AssertionError: Error in memory profiling. Initial free memory ...`.

[CmdletBinding()]
param(
    [string]$Model,
    [double]$MemUtil = 0,
    [switch]$Eager,
    [switch]$Serve,
    [switch]$TextOnly,
    [int]$MaxLen = 0,
    [int]$MaxSeqs = 4,
    [int]$BatchedTokens = 2048,
    [int]$Port = 8000,
    [switch]$NoTools,
    [switch]$NoReasoningParser,
    [switch]$W4A8,
    [string]$W4A8Layers = '',
    [switch]$MTP,
    [int]$SpecTokens = 1,
    [int]$MaxImages = 8,
    [int]$MaxPixels = 1310720,
    [int]$Tp = 1,
    [int]$Pp = 1,
    [string]$Script
)

$ErrorActionPreference = 'Continue'
. 'D:\code\vllm-windows\_dev\bin\_env.ps1'

$REPO = 'D:\code\vllm-windows'
$DEV  = "$REPO\_dev"
if (-not $Model) { $Model = $(if ($env:VMODEL) { $env:VMODEL } else { 'Qwen/Qwen2.5-0.5B-Instruct' }) }
$isLocal = Test-Path -LiteralPath $Model -PathType Container

# Never import from the repo root: the checked-out `vllm\` source tree would
# shadow the installed package and the built extension modules.
Set-Location 'C:\Users\hong'

# Runtime JIT helpers (flashinfer, torch.compile, ...) shell out to ninja/cmake.
$env:PATH = "$REPO\.venv\Scripts;$env:PATH"

$env:VMODEL                      = $Model
$env:VEAGER                      = $(if ($Eager) { '1' } else { '0' })
$env:VTEXTONLY                   = $(if ($TextOnly) { '1' } else { '0' })
$env:VMAXSEQS                    = "$MaxSeqs"
$env:VBATCHED                    = "$BatchedTokens"
if ($MaxLen -gt 0) { $env:VMAXLEN = "$MaxLen" }
$env:PYTHONUNBUFFERED            = '1'
# The Windows console defaults to GBK on a zh-CN host, which makes vLLM's
# ASCII-art banner raise UnicodeEncodeError inside the logging handler -- it is
# caught and printed as a bogus traceback into the log.  Force UTF-8 streams.
$env:PYTHONIOENCODING            = 'utf-8'
$env:PYTHONUTF8                  = '1'
$env:VLLM_ENABLE_V1_MULTIPROCESSING  = '0'   # avoid the ZMQ EngineCore subprocess
$env:VLLM_USE_FLASHINFER_SAMPLER     = '0'   # avoid flashinfer's JIT sampler
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

# W4A8: run the Marlin GEMMs with int8 activations (GA100 has int8 tensor cores).
# `lm_head|mtp` excluded by default.  -W4A8Layers <regex> narrows it further --
# `mlp` keeps the bulk of the prefill win for a smaller decode loss, because the
# MLP projections are ~63% of this model's parameters.
if ($W4A8) {
    $env:VLLM_MARLIN_INPUT_DTYPE = 'int8'
    if ($W4A8Layers) { $env:VLLM_MARLIN_INT8_INCLUDE_RE = $W4A8Layers }
    Write-Host "W4A8 enabled (int8 activations)$(if ($W4A8Layers) { " for layers matching '$W4A8Layers'" })"
}

# Derive a safe default utilization from currently-free VRAM.
if ($MemUtil -le 0) {
    if ($env:VMEM) {
        $MemUtil = [double]$env:VMEM
    } else {
        try {
            $csv  = & nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
            $line = ($csv | Select-Object -First 1) -split ','
            $free = [double]$line[0] - [double]$line[1]
            $headroom = if ($free -ge 16384) { 0.94 } else { 0.90 }
            $MemUtil = [math]::Round(($free / [double]$line[0]) * $headroom, 2)
        } catch {
            $MemUtil = 0.90
        }
    }
}
$env:VMEM = "$MemUtil"
Write-Host "gpu-memory-utilization = $MemUtil"

$py = "$REPO\.venv\Scripts\python.exe"

if ($Serve) {
    $serveArgs = @(
        '-m', 'vllm.entrypoints.openai.api_server',
        '--model', $Model,
        '--served-model-name', (Split-Path -Leaf $Model),
        '--host', '127.0.0.1',
        '--port', "$Port",
        '--max-model-len', "$(if ($MaxLen -gt 0) { $MaxLen } else { 8192 })",
        '--gpu-memory-utilization', "$MemUtil",
        '--max-num-seqs', "$MaxSeqs",
        '--max-num-batched-tokens', "$BatchedTokens",
        '--trust-remote-code'
    )
    # Tensor / pipeline parallelism.  Requires a locally built NCCL for Windows --
    # _env.ps1 auto-detects C:\nccl-windows\install\bin\nccl.dll and exports
    # VLLM_NCCL_SO_PATH.  Build it with _build_nccl.ps1.
    if ($Tp -gt 1 -or $Pp -gt 1) {
        if (-not $env:VLLM_NCCL_SO_PATH) {
            Write-Host "ERROR: -Tp/-Pp needs NCCL, but no nccl.dll was found." -ForegroundColor Red
            Write-Host "       Build it first:  & D:\code\vllm-windows\_dev\bin\_build_nccl.ps1" -ForegroundColor Red
            exit 1
        }
        $gpuCount = @(& nvidia-smi -L 2>$null | Where-Object { $_ -match '^GPU \d+:' }).Count
        if ($gpuCount -lt ($Tp * $Pp)) {
            Write-Host "WARNING: asked for $($Tp * $Pp) ranks but only $gpuCount GPU(s) are visible." -ForegroundColor Yellow
            Write-Host "         TP/PP spans physical GPUs -- install the second card first." -ForegroundColor Yellow
        }
        $serveArgs += @('--tensor-parallel-size', "$Tp")
        $serveArgs += @('--pipeline-parallel-size', "$Pp")
        Write-Host "parallelism: tp=$Tp pp=$Pp (nccl = $env:VLLM_NCCL_SO_PATH)" -ForegroundColor Cyan
        # Each rank must hold an equal shard, so the model is sized by the SMALLEST
        # card.  With a 64 GB GA100 + 24 GB GA102 pair that means the 24 GB card.
        Write-Host "note: shard size is bounded by the smallest GPU; keep -MemUtil conservative." -ForegroundColor Yellow
    }
    Write-Host "OpenAI-compatible endpoint: http://127.0.0.1:$Port/v1"
    Write-Host "model name: $(Split-Path -Leaf $Model)"
    # Do NOT add `--mamba-ssm-cache-dtype float16` here.  On qwen3_5-style hybrid
    # linear-attention models it disqualifies the fused CUDA GDN decode kernel
    # (which wants a BF16 convolution cache) and vLLM silently falls back to the
    # slower Triton path -- the log then reads `GDN decode kernel: triton`
    # instead of `: cuda`.  vLLM's own default keeps the fused kernel.
    # Multimodal is ON unless -TextOnly.  The checkpoint is a full
    # Qwen3_5ForConditionalGeneration and carries the whole vision tower
    # (333 `model.visual.*` tensors, 0.92 GB, kept BF16 because AutoRound's
    # quant ignore list covers every one of them), so stripping it only saves
    # that 0.92 GB -- worth it when you want the KV cache instead.
    #
    # When vision is on, both image knobs are set explicitly on purpose:
    # vLLM's own `--limit-mm-per-prompt` default is 999, and the checkpoint's
    # processor_config allows up to 4096x4096 px = 16384 tokens for ONE image
    # (tokens = pixels / 1024).  Left at the defaults, a handful of photos can
    # swallow a 200K context.
    if ($TextOnly) {
        $serveArgs += '--language-model-only'
    } else {
        $serveArgs += @('--limit-mm-per-prompt', "{""image"": $MaxImages}")
        if ($MaxPixels -gt 0) {
            $serveArgs += @('--mm-processor-kwargs', "{""max_pixels"": $MaxPixels}")
        }
    }

    # Tool calling needs BOTH of these flags.  Without them vLLM answers HTTP 400
    # to *every* request that carries `tools` -- including one with no
    # `tool_choice` at all, because the default is "auto":
    #     "auto" tool choice requires --enable-auto-tool-choice and
    #     --tool-call-parser to be set
    #
    # The parser has to match what the model's chat template asks for.  Qwen3.8
    # asks for XML -- <tool_call><function=NAME><parameter=K>V</parameter> -- and
    # NOT the JSON body that `hermes` (the usual answer for a Qwen model) reads.
    # Choosing wrong raises nothing: the call comes back as ordinary `content`,
    # the client sees no `tool_calls`, and it reads as the model being bad at
    # tools rather than as a misconfigured server.
    # qwen3_coder / qwen3_xml / mimo are three aliases for the same parser.
    if (-not $NoTools) {
        $serveArgs += @('--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder')
    }
    # Lifts <think>...</think> out of `content` into a separate field -- which in
    # vLLM 0.29 is named `reasoning`, not DeepSeek's `reasoning_content`.
    if (-not $NoReasoningParser) {
        $serveArgs += @('--reasoning-parser', 'qwen3')
    }
    # Native multi-token prediction (speculative decoding).  This checkpoint
    # ships one MTP head (`mtp_num_hidden_layers = 1`) inside
    # model_extra_tensors.safetensors, so method "mtp" resolves against the
    # target checkpoint itself -- no separate drafter model is needed.
    # num_speculative_tokens defaults to the MTP layer count (1); larger values
    # re-run the head autoregressively (must be a multiple of the layer count).
    if ($MTP) {
        $serveArgs += @(
            '--speculative-config',
            "{""method"": ""mtp"", ""num_speculative_tokens"": $SpecTokens}"
        )
        Write-Host "MTP speculative decoding enabled (num_speculative_tokens=$SpecTokens)"
    }

    & $py @serveArgs
    exit $LASTEXITCODE
}

if ($Script) {
    & $py -u $Script
    exit $LASTEXITCODE
}

# A local checkpoint that the 0.5B smoke test cannot cover (e.g. the W4A16
# AutoRound build) goes through the larger-model script; -Script overrides.
if ($isLocal) { $Script = "$DEV\bin\_verify_27b.py" } else { $Script = "$DEV\bin\_verify.py" }
& $py -u $Script
exit $LASTEXITCODE
