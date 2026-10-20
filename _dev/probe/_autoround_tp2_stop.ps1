# Stop the AutoRound TP=2 test instance started by _autoround_tp2_smoke.ps1.
#
#   & _dev\probe\_autoround_tp2_stop.ps1
#   & _dev\probe\_autoround_tp2_stop.ps1 -DryRun
#
# Kills depth-first from the recorded launcher pid so the engine core and the
# worker children go with it, then reports free VRAM.
[CmdletBinding()]
param([switch]$DryRun)

$ErrorActionPreference = 'Continue'
$repo = 'D:\code\vllm-windows'
$marker = "$repo\_dev\out\autoround_tp2\last"

function Get-ChildPids([int]$Parent) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$Parent" -ErrorAction SilentlyContinue |
        ForEach-Object { [int]$_.ProcessId }
}

# Post-order: children first, so no worker survives its parent's exit.
function Stop-Tree([int]$Target) {
    foreach ($child in @(Get-ChildPids $Target)) { Stop-Tree -Target $child }
    $proc = Get-Process -Id $Target -ErrorAction SilentlyContinue
    if ($null -eq $proc) { return }
    Write-Host ("  kill {0} {1}" -f $Target, $proc.ProcessName)
    if ($DryRun) { return }
    try { $proc.Kill($true) } catch { Stop-Process -Id $Target -Force -ErrorAction SilentlyContinue }
    $proc.WaitForExit(15000) | Out-Null
}

$roots = @()
if (Test-Path -LiteralPath $marker) {
    $line = (Get-Content -LiteralPath $marker -TotalCount 1)
    if ($line) { $roots += [int]("$line".Trim().Split(' ')[0]) }
}
# Orphans keep the engine handshake port and VRAM without their launcher.
$roots += @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match 'vllm\.entrypoints\.openai\.api_server' -and
        $_.CommandLine -match 'Qwen3\.8-Flash-Next-AutoRound-3bpw-MTP' -and
        $_.CommandLine -match '--port 9401'
    } | ForEach-Object { [int]$_.ProcessId })
$roots = @($roots | Sort-Object -Unique)

if (-not $roots) {
    Write-Host 'no AutoRound TP2 instance found'
} else {
    Write-Host "stopping roots: $($roots -join ', ')"
    foreach ($root in $roots) { Stop-Tree -Target $root }
}

Start-Sleep -Seconds 4
Write-Host '--- gpu memory ---'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
