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
    [int]$PleDepth = 256,
    [int]$PleCacheMb = 512,
    [int]$PleWorkers = 16,
    [int]$PlePrefetchTokens = 16384,
    [int]$MtpTokens = 3,
    [string]$Venv = 'D:\code\vllm-windows\.flashnext-root',
    [switch]$WithVision,
    [switch]$NoMtp,
    [switch]$NoGraphs,
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
if (-not $NoGraphs) { $args += '-Graphs' }
if ($WithVision) { $args += '-WithVision' }
if (-not $NoMtp) { $args += @('-Mtp', '-MtpTokens', "$MtpTokens") }
if ($Async -and -not $NoAsync) { $args += '-AsyncSched' }
if ($NoTools) { $args += '-NoTools' }
if ($NoReasoningParser) { $args += '-NoReasoningParser' }
if ($ThinkingOnDefault) { $args += '-ThinkingOnDefault' }
if ($ToolParser) { $args += @('-ToolParser', $ToolParser) }

& "$repo\_start_vllm_service.ps1" `
    -ServerScript $launcher `
    -Name 'qwen38-flash-next-win' `
    -Port $Port `
    -Model $Model `
    -TimeoutSec $TimeoutSec `
    -ServerArgs $args
exit $LASTEXITCODE
