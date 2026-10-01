# Build humming's cubin patcher as a MSVC DLL and register it as a precompiled
# artifact, so humming never shells out to g++ at serve time.
#
#   & D:\code\vllm-windows\_dev\bin\_humming_cubinpatch.ps1                       # flashtest venv
#   & D:\code\vllm-windows\_dev\bin\_humming_cubinpatch.ps1 -Venv D:\code\vllm-windows

[CmdletBinding()]
param(
    [string]$Venv = 'D:\code\vllm-windows\.flashnext-root'
)

$ErrorActionPreference = 'Stop'
. D:\code\vllm-windows\_dev\bin\_flashtest_env.ps1 -VenvPath "$Venv\.venv"

$py = "$Venv\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "no venv python at $py" }

$sitePackages = & $py -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"
$humming = Join-Path $sitePackages 'humming'
$src = Join-Path $humming 'csrc\patch_cubin.cpp'
if (-not (Test-Path $src)) { throw "no $src" }

$work = Join-Path $env:TEMP 'humming_cubinpatch'
New-Item -ItemType Directory -Force -Path $work | Out-Null
Remove-Item -Force "$work\libcubinpatch.dll" -ErrorAction SilentlyContinue

$cl = @(
    'cl', '/nologo', '/O2', '/std:c++17', '/DNDEBUG', '/MD', '/EHsc', '/utf-8',
    "/I`"$($env:CUDA_PATH)\include`"", "`"$src`"",
    "/Fo`"$work\patch_cubin.obj`"",
    '/link', '/DLL', "/OUT:`"$work\libcubinpatch.dll`""
)
$buildLog = Join-Path $work 'build.log'
(& cmd.exe /c ($cl -join ' ')) *>&1 | Out-File $buildLog -Encoding utf8
if (-not (Test-Path "$work\libcubinpatch.dll")) {
    Get-Content $buildLog | Write-Output
    throw 'cl.exe produced no libcubinpatch.dll'
}
Write-Host ("built libcubinpatch.dll  size=" + (Get-Item "$work\libcubinpatch.dll").Length)

$native = Join-Path $humming '_native\x86_64'
New-Item -ItemType Directory -Force -Path $native | Out-Null
Copy-Item "$work\libcubinpatch.dll" "$native\libcubinpatch.dll" -Force

$hash = & $py -c "import humming.utils.jit as j; print(j.hash_path_content(r'$src', releative=True))"
$manifestPath = Join-Path $native 'manifest.json'
$manifest = @{}
if (Test-Path $manifestPath) {
    foreach ($p in (Get-Content $manifestPath -Raw | ConvertFrom-Json).PSObject.Properties) {
        $manifest.($p.Name) = $p.Value
    }
}
$manifest.'libcubinpatch.dll' = "$hash".Trim()
($manifest | ConvertTo-Json -Depth 5) | Set-Content -Path $manifestPath -Encoding ascii
Write-Host "manifest -> libcubinpatch.dll = $($manifest.'libcubinpatch.dll')"

$probe = Join-Path $work 'probe.py'
Set-Content -Path $probe -Encoding ascii -Value @"
from humming.utils.cubin import may_build_cubin_patcher, _RC_MESSAGE
lib = may_build_cubin_patcher()
print('cubin patcher loaded from', lib._handle)
print('dry probe on a missing file should raise:', _RC_MESSAGE.get(2, 'n/a'))
"@
& $py $probe
