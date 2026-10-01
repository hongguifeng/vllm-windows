# Start the flash-next server detached, so it outlives the calling command.
#
#   & D:\code\vllm-windows\_dev\bin\_serve_bg.ps1 -ServeArgs @('-FullWeights','-PleSsd')
#
# The server itself logs to a timestamped file the launcher picks; this returns
# that path once the child is running so a caller can poll it.
[CmdletBinding()]
param(
    [string]$ServeScript = 'D:\code\vllm-windows\_dev\bin\_flashnext_struct_serve.ps1',
    [string[]]$ServeArgs = @()
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ServeScript)) {
    Write-Host "ERROR: no such server script: $ServeScript" -ForegroundColor Red
    exit 1
}

$before = @{}
$logDir = 'D:\code\vllm-windows\_dev\out\logs'
if (Test-Path $logDir) {
    foreach ($f in Get-ChildItem -LiteralPath $logDir -Filter '*.log') {
        $before[$f.Name] = $true
    }
}

$child = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $ServeScript) + $ServeArgs) `
    -WindowStyle Minimized -PassThru
Write-Host "server pid: $($child.Id)"

$log = $null
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    $fresh = Get-ChildItem -LiteralPath $logDir -Filter '*.log' |
        Where-Object { -not $before.ContainsKey($_.Name) } |
        Sort-Object LastWriteTime -Descending
    if ($fresh) {
        $log = $fresh[0].FullName
        break
    }
}

if ($null -eq $log) {
    Write-Host 'ERROR: no new log appeared within 30 s' -ForegroundColor Red
    exit 1
}
Write-Host "log: $log"
exit 0
