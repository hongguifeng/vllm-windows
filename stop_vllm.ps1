[CmdletBinding()]
param(
    [switch]$KeepGpuReport
)

$ErrorActionPreference = 'Continue'
$repo = 'D:\code\vllm-windows'
$stopper = "$repo\_dev\bin\_stop_vllm.ps1"

if (-not (Test-Path -LiteralPath $stopper)) {
    Write-Host "ERROR: stop helper not found: $stopper" -ForegroundColor Red
    exit 1
}

Write-Host '=== stopping vLLM services ===' -ForegroundColor Yellow
& $stopper
$code = $LASTEXITCODE

if ($KeepGpuReport) { exit $code }
exit $code
