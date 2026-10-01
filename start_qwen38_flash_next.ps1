[CmdletBinding()]
param(
    [string]$Model = 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP',
    [string]$ServedName = 'qwen3.8-flash-next-full',
    [int]$Port = 8111,
    [int]$Gpu = 1,
    [int]$MaxLen = 204800,
    [int]$MaxSeqs = 4,
    [int]$BatchedTokens = 4096,
    [double]$MemUtil = 0.95,
    [int]$KvGiB = 14,
    [int]$PleDepth = 256,
    [int]$PleCacheMb = 512,
    [int]$PleWorkers = 16,
    [int]$PlePrefetchTokens = 16384,
    [int]$MtpTokens = 2,
    [switch]$WithVision,
    [switch]$NoMtp,
    [switch]$NoGraphs,
    [switch]$NoAsync,
    [int]$TimeoutSec = 1800
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
    '-Venv', $repo,
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
if (-not $NoGraphs) { $args += '-Graphs' }
if ($WithVision) { $args += '-WithVision' }
if (-not $NoMtp) { $args += @('-Mtp', '-MtpTokens', "$MtpTokens") }
if (-not $NoAsync) { $args += '-AsyncSched' }

& "$repo\_start_vllm_service.ps1" `
    -ServerScript $launcher `
    -Name 'qwen38-flash-next' `
    -Port $Port `
    -Model $Model `
    -TimeoutSec $TimeoutSec `
    -ServerArgs $args
exit $LASTEXITCODE
