# Build the Windows PLE/SSD native helper (IOCP reads).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_ple_ssd_io_win_build.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_ple_ssd_io_win_build.ps1 -OutDir 'D:\some\dir'

param(
    [string]$Repo = 'D:\code\vllm-windows',
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'
$WinRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # -> d:\code\vllm-windows
. (Join-Path $PSScriptRoot '_flashtest_env.ps1') -Repo $Repo

$src = Join-Path $Repo 'vllm\models\qwen4_exp\nvidia\ple_ssd_io_win.c'
if (-not (Test-Path $src)) { throw "missing $src" }

if (-not $OutDir) { $OutDir = Join-Path $WinRoot '_dev\out\ple_ssd_io' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$dll = Join-Path $OutDir 'ple_ssd_io_win.dll'
$lib = Join-Path $OutDir 'ple_ssd_io_win.lib'
$fo = '/Fo:' + $OutDir + '\'

Push-Location $OutDir
try {
    # A stale DLL must not be able to masquerade as a fresh build.
    Remove-Item $dll, $lib, "$OutDir\*.obj" -Force -ErrorAction SilentlyContinue
    & cl.exe /nologo /W3 /O2 /LD $fo "/Fe:$dll" "/Fd:$($OutDir)\ple_ssd_io_win.pdb" $src `
        /link /OUT:$dll "/IMPLIB:$lib" kernel32.lib
    if ($LASTEXITCODE -ne 0) { throw "cl exited $LASTEXITCODE" }
    if (-not (Test-Path $dll)) { throw "compile produced no dll" }
    $fi = Get-Item $dll
    Write-Host "built $dll ($($fi.Length) B, $($fi.LastWriteTime))" -ForegroundColor Green
    Write-Host ("sha256 " + (Get-FileHash $dll -Algorithm SHA256).Hash) -ForegroundColor DarkGray
    Write-Host "point additional_config ple_ssd_native_library at this file" -ForegroundColor Cyan
} finally {
    Pop-Location
}
