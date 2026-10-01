[CmdletBinding()]
param(
    [string]$Model = 'D:\models\Qwen3.8-27B-W4A16-AutoRound-fast',
    [string]$Drafter = 'D:\models\Qwen3.8-27B-DFlash2-W4A16',
    [ValidateSet('dflash', 'mtp', 'none')]
    [string]$Spec = 'dflash',
    [int]$Port = 8000,
    [int]$Gpu = 1,
    [int]$MaxLen = 0,
    [long]$KvBytes = 0,
    [int]$MaxSeqs = 8,
    [switch]$TextOnly,
    [switch]$VisionOffload,
    [switch]$NoAsync,
    [switch]$Force,
    [int]$TimeoutSec = 1800
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\code\vllm-windows'
$launcher = "$repo\start_server_170hx.ps1"

if (-not (Test-Path -LiteralPath "$Model\config.json")) {
    Write-Host "ERROR: model config not found: $Model\config.json" -ForegroundColor Red
    exit 1
}
if ($Spec -eq 'dflash' -and -not (Test-Path -LiteralPath "$Drafter\config.json")) {
    Write-Host "ERROR: DFlash2 drafter config not found: $Drafter\config.json" -ForegroundColor Red
    exit 1
}

$args = @(
    '-Model', $Model,
    '-Drafter', $Drafter,
    '-Spec', $Spec,
    '-Gpu', "$Gpu",
    '-Port', "$Port",
    '-MaxLen', "$MaxLen",
    '-KvBytes', "$KvBytes",
    '-MaxSeqs', "$MaxSeqs"
)
if ($TextOnly) { $args += '-TextOnly' }
if ($VisionOffload) { $args += '-VisionOffload' }
if ($NoAsync) { $args += '-NoAsync' }
if ($Force) { $args += '-Force' }

& "$repo\_start_vllm_service.ps1" `
    -ServerScript $launcher `
    -Name 'qwen38-27b' `
    -Port $Port `
    -Model $Model `
    -TimeoutSec $TimeoutSec `
    -ServerArgs $args
exit $LASTEXITCODE
