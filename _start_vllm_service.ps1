[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ServerScript,
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [int]$Port,
    [string]$Model = '',
    [int]$TimeoutSec = 1800,
    [string[]]$ServerArgs = @()
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\code\vllm-windows'
$logDir = "$repo\_dev\out\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $ServerScript)) {
    throw "server script not found: $ServerScript"
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutLog = "$logDir\${Name}_${stamp}.stdout.log"
$stderrLog = "$logDir\${Name}_${stamp}.stderr.log"

Write-Host "=== starting $Name ===" -ForegroundColor Cyan
if ($Model) { Write-Host "model : $Model" }
Write-Host "port  : $Port"
Write-Host "logs  : $stdoutLog"
Write-Host "        $stderrLog"
Write-Host ''

$child = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $ServerScript) + $ServerArgs) `
    -WorkingDirectory 'C:\Users\hong' `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Host "launcher pid: $($child.Id)"
Write-Host 'waiting for /health; startup output follows:' -ForegroundColor Yellow
Write-Host ''

$stdoutOffset = 0L
$stderrOffset = 0L
$started = Get-Date
$healthUri = "http://127.0.0.1:$Port/health"

function Show-LogDelta([string]$Path, [long]$Offset, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { return $Offset }
    try {
        $text = [System.IO.File]::ReadAllText($Path)
    } catch {
        return $Offset
    }
    if ($text.Length -le $Offset) { return $Offset }
    $delta = $text.Substring([int]$Offset)
    if ($delta.Trim().Length -gt 0) {
        Write-Host ("[{0}] {1}" -f $Label, $delta.TrimEnd())
    }
    return [long]$text.Length
}

while (((Get-Date) - $started).TotalSeconds -lt $TimeoutSec) {
    $stdoutOffset = Show-LogDelta $stdoutLog $stdoutOffset 'out'
    $stderrOffset = Show-LogDelta $stderrLog $stderrOffset 'err'

    try {
        $response = Invoke-WebRequest -Uri $healthUri -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $stdoutOffset = Show-LogDelta $stdoutLog $stdoutOffset 'out'
            $stderrOffset = Show-LogDelta $stderrLog $stderrOffset 'err'
            $elapsed = [int]((Get-Date) - $started).TotalSeconds
            Write-Host ''
            Write-Host "=== $Name is ready (${elapsed}s) ===" -ForegroundColor Green
            Write-Host "endpoint: http://127.0.0.1:$Port/v1"
            Write-Host "health  : $healthUri"
            Write-Host "metrics : http://127.0.0.1:$Port/metrics"
            if ($Model) { Write-Host "model   : $Model" }
            Write-Host "stdout  : $stdoutLog"
            Write-Host "stderr  : $stderrLog"
            Write-Host 'stop    : D:\code\vllm-windows\stop_vllm.ps1'
            exit 0
        }
    } catch {
        # The server is still loading, compiling, or has not bound the port yet.
    }

    if (-not (Get-Process -Id $child.Id -ErrorAction SilentlyContinue)) {
        $stdoutOffset = Show-LogDelta $stdoutLog $stdoutOffset 'out'
        $stderrOffset = Show-LogDelta $stderrLog $stderrOffset 'err'
        Write-Host ''
        Write-Host "ERROR: $Name launcher exited before /health became ready" -ForegroundColor Red
        Write-Host "stdout: $stdoutLog"
        Write-Host "stderr: $stderrLog"
        exit 1
    }
    Start-Sleep -Seconds 2
}

$stdoutOffset = Show-LogDelta $stdoutLog $stdoutOffset 'out'
$stderrOffset = Show-LogDelta $stderrLog $stderrOffset 'err'
Write-Host ''
Write-Host "ERROR: timed out after $TimeoutSec seconds waiting for $healthUri" -ForegroundColor Red
Write-Host "stdout: $stdoutLog"
Write-Host "stderr: $stderrLog"
Write-Host 'The process is still running; use stop_vllm.ps1 if it needs to be stopped.'
exit 1
