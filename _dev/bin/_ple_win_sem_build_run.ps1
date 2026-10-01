# Build and run the CPU-only Win32 semantics probe for the PLE/SSD reader.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_ple_win_sem_build_run.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_ple_win_sem_build_run.ps1 -Dll 'D:\some\ple_ssd_io_win.dll'

param(
    [string]$Dll = 'D:\code\vllm-windows\_dev\out\ple_ssd_io\ple_ssd_io_win.dll'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # -> d:\code\vllm-windows
. (Join-Path $PSScriptRoot '_flashtest_env.ps1') -Repo 'D:\code\vllm-windows'

$src = Join-Path $Root '_dev\probe\_ple_win_sem.c'
if (-not (Test-Path $src)) { throw "missing $src" }
if (-not (Test-Path $Dll)) { throw "missing $Dll" }

$outDir = 'C:\Users\hong\AppData\Local\Temp\plesem'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$exe = Join-Path $outDir 'ple_win_sem.exe'

Push-Location $outDir
try {
    Remove-Item $exe, "$outDir\*.obj" -Force -ErrorAction SilentlyContinue
    & cl.exe /nologo /W3 /O2 "/Fo:$outDir\" "/Fe:$exe" $src `
        /link /OUT:$exe kernel32.lib user32.lib msvcrt.lib
    if (-not (Test-Path $exe)) { throw 'compile produced no exe' }
    & $exe $Dll
    Write-Host ("probe exit code: {0}" -f $LASTEXITCODE) -ForegroundColor Cyan
} finally {
    Pop-Location
}
