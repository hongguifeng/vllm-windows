# Build the unbuffered-IO shim DLL that Python loads through ctypes.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_nbshim_build.ps1

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # -> d:\code\vllm-windows
. (Join-Path $PSScriptRoot '_flashtest_env.ps1')

$src = Join-Path $Root '_dev\probe\_nbshim.c'
$outDir = Join-Path $Root '_dev\out\nbshim'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$dll = Join-Path $outDir 'nbshim.dll'
$lib = Join-Path $outDir 'nbshim.lib'
$fo = '/Fo:' + $outDir + '\'

Push-Location $outDir
try {
    & cl.exe /nologo /W3 /O2 /LD $fo "/Fe:$dll" "/Fd:$($outDir)\nbshim.pdb" $src `
        /link /OUT:$dll "/IMPLIB:$lib" kernel32.lib user32.lib
    if (-not (Test-Path $dll)) { throw "compile produced no dll" }
    $fi = Get-Item $dll
    Write-Host "built $dll ($($fi.Length) B, $($fi.LastWriteTime))" -ForegroundColor Green
    Write-Host ("sha256 " + (Get-FileHash $dll -Algorithm SHA256).Hash) -ForegroundColor DarkGray
} finally {
    Pop-Location
}
