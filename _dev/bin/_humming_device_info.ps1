# Build humming's device-info CPython extension with MSVC.
#
# humming builds humming/_device_info at first use by shelling out to g++, which
# does not exist here, and Python on Windows only loads .pyd files anyway. So
# compile the same source with cl.exe and drop the result into
# humming/_native/<arch>/, where humming looks for precompiled artifacts
# (manifest.json has to carry the source hash or the artifact is ignored).
#
#   & D:\code\vllm-windows\_dev\bin\_humming_device_info.ps1                    # flashtest venv
#   & D:\code\vllm-windows\_dev\bin\_humming_device_info.ps1 -Venv D:\code\vllm-windows

[CmdletBinding()]
param(
    [string]$Venv = 'D:\code\vllm-windows'
)

$ErrorActionPreference = 'Stop'
. D:\code\vllm-windows\_dev\bin\_flashtest_env.ps1 -Repo $Venv

$py = "$Venv\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "no venv python at $py" }

$sitePackages = & $py -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"
$humming = Join-Path $sitePackages 'humming'
if (-not (Test-Path "$humming\csrc\device_info.cpp")) { throw "no humming package at $humming" }

$cuda      = $env:CUDA_PATH
$pyInclude = & $py -c "import sysconfig;print(sysconfig.get_paths()['include'])"
# A uv-installed standalone CPython keeps its import libraries next to the
# interpreter, not in the venv, so follow pyvenv.cfg back to the real home.
$pyLibs = & $py -c @"
import os, sys
for parent in (os.path.dirname(sys.base_prefix), sys.base_prefix):
    cand = os.path.join(parent, 'libs')
    if os.path.isfile(os.path.join(cand, 'python%d%d.lib' % sys.version_info[:2])):
        print(cand); break
"@
$pyLibs = "$pyLibs".Trim()
if (-not (Test-Path "$pyLibs\python312.lib")) { throw "no python312.lib under $pyLibs" }

$work = Join-Path $env:TEMP 'humming_device_info'
New-Item -ItemType Directory -Force -Path $work | Out-Null
Remove-Item -Force "$work\_device_info.pyd" -ErrorAction SilentlyContinue
# python312.lib forwards to python3.lib, which only lives next to it.
$env:LIB = "$pyLibs;$env:LIB"

$cl = @(
    'cl', '/nologo', '/O2', '/std:c++17', '/DNDEBUG', '/MD', '/EHsc', '/utf-8',
    "/I`"$pyInclude`"", "/I`"$cuda\include`"",
    "`"$humming\csrc\device_info.cpp`"",
    "/Fo`"$work\device_info.obj`"",
    '/link', '/DLL', "`"$pyLibs\python312.lib`"", "`"$cuda\lib\x64\cuda.lib`"",
    "/OUT:`"$work\_device_info.pyd`""
)
Write-Host "cl $([string]::Join(' ', $cl))"
$buildLog = Join-Path $work 'build.log'
(& cmd.exe /c ($cl -join ' ')) *>&1 | Out-File $buildLog -Encoding utf8
if (-not (Test-Path "$work\_device_info.pyd")) {
    Get-Content $buildLog | Write-Output
    throw 'cl.exe produced no _device_info.pyd'
}
Write-Host ("built _device_info.pyd  size=" + (Get-Item "$work\_device_info.pyd").Length)

$native = Join-Path $humming '_native\x86_64'
New-Item -ItemType Directory -Force -Path $native | Out-Null
Copy-Item "$work\_device_info.pyd" "$native\_device_info.pyd" -Force

$hash = & $py -c "import humming.utils.jit as j; print(j.hash_path_content(r'$humming\csrc\device_info.cpp', releative=True, text_only=False))"
$manifestPath = Join-Path $native 'manifest.json'
$manifest = @{}
if (Test-Path $manifestPath) {
    foreach ($p in (Get-Content $manifestPath -Raw | ConvertFrom-Json).PSObject.Properties) {
        $manifest.($p.Name) = $p.Value
    }
}
$manifest.'_device_info.pyd' = "$hash".Trim()
($manifest | ConvertTo-Json -Depth 5) | Set-Content -Path $manifestPath -Encoding ascii
Write-Host "manifest $manifestPath -> _device_info.pyd = $($manifest.'_device_info.pyd')"

& $py -c "from humming.utils.device import _get_device_info_extension as g; m = g(); print('loaded', m.__file__)"
