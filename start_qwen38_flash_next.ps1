# Front door for the Flash-Next service: builds the launcher arguments, starts it
# detached through _start_vllm_service.ps1, and waits for /health.
#
#   & D:\code\vllm-windows\start_qwen38_flash_next.ps1              # GPU1, :9393
#   & D:\code\vllm-windows\start_qwen38_flash_next.ps1 -Gpu 0       # the other card
#   & ...\start_qwen38_flash_next.ps1 -NoVision -MtpTokens 2        # text-only arm
#   & ...\start_qwen38_flash_next.ps1 -MaxPixels 0                  # uncapped images
# Vision is ON by default: the tower takes ~2.3 GiB out of the auto KV budget, which
# leaves MORE VRAM free than letting KV fill the whole 0.94 ceiling.
#   & ...\start_qwen38_flash_next.ps1 -CaptureSizes '[1,2,3,...,2048]'  # explicit graph coverage
#
# One card's service is stopped by D:\code\vllm-windows\stop_vllm.ps1 -Gpu 0 (or
# -Gpu 1 / -Port 9393); with no argument it stops both, as before.
# A service running inside WSL is not visible to that script -- stop it with
# `wsl.exe -e bash -lc "~/code/qwen3.8-flash-next-cmp170hx/bin/stop.sh --check"`.
[CmdletBinding()]
param(
    [string]$Model = 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP',
    [string]$ServedName = 'qwen3.8-flash-next-win',
    [int]$Port = 9393,
    [int]$Gpu = 1,
    [int]$MaxLen = 262144,
    [int]$MaxSeqs = 4,
    [int]$BatchedTokens = 2048,
    [double]$MemUtil = 0.94,
    [int]$KvGiB = 0,
    # Empty (default) = production behaviour unchanged. Bracketed list reproduces
    # the WSL arm's explicit CUDA-graph capture coverage.
    [string]$CaptureSizes = '',
    [int]$PleDepth = 256,
    [int]$PleCacheMb = 512,
    [int]$PleWorkers = 16,
    [int]$PlePrefetchTokens = 16384,
    [int]$MtpTokens = 3,
    [string]$Venv = 'D:\code\vllm-windows\.flashnext-root',
    [switch]$NoVision,
    [int]$MaxPixels = 1310720,
    [switch]$NoAllocHeal,
    [switch]$NoMtp,
    [switch]$NoGraphs,
    [switch]$SkipGuard,
    [switch]$DryRun,
    [switch]$Async,
    [switch]$NoAsync,
    [int]$TimeoutSec = 1800,
    [string]$ToolParser = 'qwen3_xml',
    [switch]$NoTools,
    [switch]$NoReasoningParser,
    [switch]$ThinkingOnDefault
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\code\vllm-windows'
$launcher = "$repo\_dev\bin\_flashnext_struct_serve.ps1"

if (-not (Test-Path -LiteralPath "$Model\config.json")) {
    Write-Host "ERROR: model config not found: $Model\config.json" -ForegroundColor Red
    exit 1
}

$args = @(
    '-Model', $Model,
    '-ServedName', $ServedName,
    '-Venv', $Venv,
    '-Port', "$Port",
    '-Gpu', "$Gpu",
    '-MaxLen', "$MaxLen",
    '-MaxSeqs', "$MaxSeqs",
    '-BatchedTokens', "$BatchedTokens",
    '-MemUtil', "$MemUtil",
    '-KvGiB', "$KvGiB",
    '-PleDepth', "$PleDepth",
    '-PleCacheMb', "$PleCacheMb",
    '-PleWorkers', "$PleWorkers",
    '-PlePrefetchTokens', "$PlePrefetchTokens",
    '-FullWeights',
    '-PleSsd'
)
if ($CaptureSizes) { $args += @('-CaptureSizes', $CaptureSizes) }
if (-not $NoGraphs) { $args += '-Graphs' }
if ($NoAllocHeal) { $args += '-NoAllocHeal' }
if ($DryRun) { $args += '-DryRun' }
# Vision is ON by default; -NoVision falls back to the inner script's text-only
# default (that is what the measurement arms use).
if (-not $NoVision) { $args += '-WithVision' }
$args += @('-MaxPixels', "$MaxPixels")
if (-not $NoMtp) { $args += @('-Mtp', '-MtpTokens', "$MtpTokens") }
if ($Async -and -not $NoAsync) { $args += '-AsyncSched' }
if ($NoTools) { $args += '-NoTools' }
if ($NoReasoningParser) { $args += '-NoReasoningParser' }
if ($ThinkingOnDefault) { $args += '-ThinkingOnDefault' }
if ($ToolParser) { $args += @('-ToolParser', $ToolParser) }

# -DryRun prints the composed command and returns. Do not go through the service
# wrapper, which waits on /health and would report an OLD server as "ready".
# A child powershell parses $args as named parameters; an in-process @args splat
# would bind them positionally.
if ($DryRun) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launcher @args
    exit $LASTEXITCODE
}

& "$repo\_start_vllm_service.ps1" `
    -ServerScript $launcher `
    -Name 'qwen38-flash-next-win' `
    -Port $Port `
    -Model $Model `
    -TimeoutSec $TimeoutSec `
    -GuardGpu $Gpu `
    -SkipGuard:$SkipGuard `
    -ServerArgs $args
exit $LASTEXITCODE
