[CmdletBinding()]
param(
    [string]$Model = 'D:\models\Qwen3.8-27B-W4A16-AutoRound-fast',
    [int]$Port = 8000,
    [int]$Gpu = 1,
    [int]$MaxLen = 16384,
    [int]$MaxSeqs = 4,
    [int]$BatchedTokens = 4096,
    [switch]$TextOnly,
    [switch]$NoMtp,
    [switch]$Eager,
    [double]$MemUtil = 0,
    [int]$TimeoutSec = 900
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\code\vllm-windows'
$launcher = "$repo\_dev\bin\_run.ps1"

if (-not (Test-Path -LiteralPath "$Model\config.json")) {
    Write-Host "ERROR: model config not found: $Model\config.json" -ForegroundColor Red
    exit 1
}

# _run.ps1 uses the first visible CUDA device. GPU1 is the Windows card on this
# machine; pass -Gpu 0 when running on a machine where the target card is GPU0.
$env:CUDA_VISIBLE_DEVICES = "$Gpu"

$args = @(
    '-Model', $Model,
    '-Serve',
    '-Port', "$Port",
    '-MaxLen', "$MaxLen",
    '-MaxSeqs', "$MaxSeqs",
    '-BatchedTokens', "$BatchedTokens"
)
if ($MemUtil -gt 0) { $args += @('-MemUtil', "$MemUtil") }
if ($TextOnly) { $args += '-TextOnly' }
if (-not $NoMtp) { $args += '-MTP' }
if ($Eager) { $args += '-Eager' }

& "$repo\_start_vllm_service.ps1" `
    -ServerScript $launcher `
    -Name 'qwen38-27b' `
    -Port $Port `
    -Model $Model `
    -TimeoutSec $TimeoutSec `
    -ServerArgs $args
exit $LASTEXITCODE
