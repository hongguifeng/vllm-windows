# Stop the EXL3 TP2 smoke launcher started by _exl3_tp2_smoke.ps1 and every
# process it spawned (EngineCore, TP workers), deepest first so no worker is
# orphaned while its orchestrator still holds GPU memory.
$ErrorActionPreference = 'Continue'

function Stop-ProcessTree([int]$root) {
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$root"
    foreach ($child in $children) { Stop-ProcessTree $child.ProcessId }
    Stop-Process -Id $root -Force -ErrorAction SilentlyContinue
    Write-Host "stopped $root"
}

$marker = 'D:\code\vllm-windows\_dev\out\logs\exl3_tp2.last'
if (Test-Path -LiteralPath $marker) {
    $line = Get-Content -LiteralPath $marker -First 1
    $parts = $line -split ' ', 2
    Stop-ProcessTree ([int]$parts[0])
}

# Orphans of an earlier stop: the API server is gone but EngineCore (a
# multiprocessing spawn whose parent no longer exists) still holds the cards.
Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -like 'python*' -and
        $_.CommandLine -like '*multiprocessing.spawn*' -and
        -not (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue)
    } |
    ForEach-Object { Stop-ProcessTree $_.ProcessId }
