# Stop the experiment instance started by _flashnext_arm.ps1.
#
#   & _dev\probe\_flashnext_arm_stop.ps1                 # the port in the marker
#   & _dev\probe\_flashnext_arm_stop.ps1 -Port 9500      # any port, marker gone
#   & _dev\probe\_flashnext_arm_stop.ps1 -DryRun         # show what would die
#
# The instance is identified by its port, which _flashnext_arm.ps1 records in
# _dev\out\arms\last ("pid=... port=... log=..."), so any port works: pass -Port
# when the marker is gone or belongs to another instance. -ProtectPort (default
# 9393, the production service) is never touched, even through a stale marker or
# a pid tree that reaches it.
[CmdletBinding()]
param(
    [int]$Port = 0,
    [int[]]$ProtectPort = @(9393),
    [switch]$DryRun
)

$ErrorActionPreference = 'Continue'
$repo = 'D:\code\vllm-windows'
$marker = "$repo\_dev\out\arms\last"
$explicitPort = $PSBoundParameters.ContainsKey('Port')

function Get-ChildPids([int]$Parent) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$Parent" -ErrorAction SilentlyContinue |
        ForEach-Object { [int]$_.ProcessId }
}

function Get-ParentPid([int]$Child) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$Child" -ErrorAction SilentlyContinue
    if ($proc -and $proc.ParentProcessId) { [int]$proc.ParentProcessId } else { 0 }
}

function Get-PortsOf([string]$CommandLine) {
    @([regex]::Matches($CommandLine, '--port\s+(\d+)') | ForEach-Object { [int]$_.Groups[1].Value })
}

function Stop-Tree([int]$Target) {
    foreach ($child in @(Get-ChildPids $Target)) { Stop-Tree -Target $child }
    $proc = Get-Process -Id $Target -ErrorAction SilentlyContinue
    if ($null -eq $proc) { return }
    Write-Host ("  kill {0} {1}" -f $Target, $proc.ProcessName)
    if ($DryRun) { return }
    try { $proc.Kill($true) } catch { Stop-Process -Id $Target -Force -ErrorAction SilentlyContinue }
    $proc.WaitForExit(15000) | Out-Null
}

$launcherPid = 0
$markerPort = 0
if (Test-Path -LiteralPath $marker) {
    # "pid=<n> port=<n> log=<path>"; older markers were "<pid> <port> <log>" or
    # "<pid> <log>", so accept every shape.
    $line = "$(Get-Content -LiteralPath $marker -TotalCount 1)"
    if ($line -match 'pid=(\d+)') { $launcherPid = [int]$Matches[1] }
    elseif ($line -match '^\s*(\d+)') { $launcherPid = [int]$Matches[1] }
    if ($line -match 'port=(\d+)') { $markerPort = [int]$Matches[1] }
    elseif ($line -match '^\s*\d+\s+(\d{2,5})\s') { $markerPort = [int]$Matches[1] }
}
if (-not $explicitPort) { $Port = $markerPort }
# The marker only describes the instance it recorded; a different -Port means
# the caller is after another one, so its pid must not be reused.
if ($explicitPort -and $markerPort -ne $Port) { $launcherPid = 0 }

if ($Port -le 0 -and $launcherPid -le 0) {
    Write-Host 'no port to identify an experiment instance: pass -Port <n>'
    exit 0
}

$servers = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match 'vllm\.entrypoints\.openai\.api_server' -and
        @(Get-PortsOf $_.CommandLine | Where-Object { $ProtectPort -notcontains $_ }).Count -gt 0
    }
if ($Port -gt 0) {
    $servers = $servers | Where-Object { @(Get-PortsOf $_.CommandLine) -contains $Port }
}

$roots = @()
if ($launcherPid -gt 0) { $roots += $launcherPid }
$roots += @($servers | ForEach-Object { [int]$_.ProcessId })
$roots = @($roots | Sort-Object -Unique)
# A stale marker may name production's launcher; never take that down.
$roots = @($roots | Where-Object {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$_" -ErrorAction SilentlyContinue).CommandLine
    if (-not $cmd) { return $false }
    @(Get-PortsOf $cmd | Where-Object { $ProtectPort -contains $_ }).Count -eq 0
})
# Keep only the topmost roots: killing a parent already takes its children.
$roots = @($roots | Where-Object {
    $walker = [int]$_
    $isChild = $false
    for ($i = 0; $i -lt 16; $i++) {
        $walker = Get-ParentPid $walker
        if ($walker -le 0) { break }
        if ($roots -contains $walker) { $isChild = $true; break }
    }
    -not $isChild
})

if (-not $roots) {
    Write-Host "no experiment instance found for port $Port"
} else {
    Write-Host ("stopping port {0} (roots {1}; protected: {2})" -f $Port, ($roots -join ', '), ($ProtectPort -join ', '))
    foreach ($root in $roots) { Stop-Tree -Target $root }
}

if ($DryRun) { Write-Host '(dry run: nothing killed)' } else {
    Start-Sleep -Seconds 4
    Write-Host '--- gpu memory ---'
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}
