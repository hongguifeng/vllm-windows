# Build and run the pure-C unbuffered-IO test.
#
# Removes ctypes from the equation: if this exe also fails NO_BUFFERING, the
# rejection is real and not an artifact of how Python calls the Win32 API.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_nbtest_build_run.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_nbtest_build_run.ps1 -ExtraFile 'D:\some\file'

param(
    [string]$ExtraFile = ''
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # -> d:\code\vllm-windows
$envScript = Join-Path $PSScriptRoot '_flashtest_env.ps1'
if (-not (Test-Path $envScript)) { throw "missing $envScript" }
. $envScript

$src = Join-Path $Root '_dev\probe\_nbtest.c'
$outDir = 'C:\Users\hong\AppData\Local\Temp\psprobe'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$exe = Join-Path $outDir 'nbtest.exe'
$obj = Join-Path $outDir 'nbtest.obj'

Push-Location $outDir
try {
    $fo = '/Fo:' + $outDir + '\'
$fe = '/Fe:' + $exe
& cl.exe /nologo /W3 /O2 $fo $fe $src
    if (-not (Test-Path $exe)) { throw "compile produced no exe" }
} finally {
    Pop-Location
}

# a file we created ourselves, so no AV/EDR history is involved
$fresh = Join-Path $outDir 'nbtest_fresh_c.bin'
$bytes = New-Object byte[] (1MB)
[IO.File]::WriteAllBytes($fresh, $bytes)

$ple = 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP\model-00001-of-00011.safetensors'
foreach ($f in @($fresh, $ple, $ExtraFile)) {
    if (-not $f -or -not (Test-Path $f)) { continue }
    Write-Host "=== $f ($([int](Get-Item $f).Length) B) ===" -ForegroundColor Cyan
    & $exe $f | Out-String -Width 110 | Write-Host
}
