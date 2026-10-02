# Stop a vLLM service started by these scripts, then report GPU memory.
#
#   & D:\code\vllm-windows\stop_vllm.ps1              # stop everything (old behaviour)
#   & D:\code\vllm-windows\stop_vllm.ps1 -Gpu 0       # only the service on GPU0
#   & D:\code\vllm-windows\stop_vllm.ps1 -Gpu 1       # only the service on GPU1
#   & D:\code\vllm-windows\stop_vllm.ps1 -Port 9393   # select by port instead of card
#   & D:\code\vllm-windows\stop_vllm.ps1 -List        # roster only, stop nothing
#   & D:\code\vllm-windows\stop_vllm.ps1 -Gpu 0 -DryRun
#
# Selection works by mapping the EngineCore pid nvidia-smi reports back to a card,
# then pulling in that process's whole tree (API server, resource tracker) -- killing
# only one half leaves the other holding the port or the KV pool.
#
# A service running inside WSL cannot be stopped from here: its python.exe is not a
# Windows process and its VRAM is charged to vmmem. Stop those inside WSL:
#   wsl.exe -e bash -lc "~/code/qwen3.8-flash-next-cmp170hx/bin/stop.sh --check"
[CmdletBinding()]
param(
    [switch]$KeepGpuReport,
    # -1 (default) = every vLLM process, whatever card it is on.
    [int]$Gpu = -1,
    [string]$Port = '',
    [switch]$DryRun,
    [switch]$List
)

$ErrorActionPreference = 'Continue'
$repo = 'D:\code\vllm-windows'
$stopper = "$repo\_dev\bin\_stop_vllm.ps1"

if (-not (Test-Path -LiteralPath $stopper)) {
    Write-Host "ERROR: stop helper not found: $stopper" -ForegroundColor Red
    exit 1
}

$scope = if ($Port) { "port $Port" } elseif ($Gpu -ge 0) { "GPU$Gpu" } else { 'every card' }
if ($DryRun) {
    Write-Host "=== dry run: would stop the vLLM service on $scope ===" -ForegroundColor Yellow
} elseif ($List) {
    Write-Host '=== listing vLLM services ===' -ForegroundColor Cyan
} else {
    Write-Host "=== stopping the vLLM service on $scope ===" -ForegroundColor Yellow
}
& $stopper -Gpu $Gpu -Port $Port -DryRun:$DryRun -List:$List
$code = $LASTEXITCODE

if ($KeepGpuReport) { exit $code }
exit $code
