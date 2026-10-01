# Build humming's nvrtc_compile helper with MSVC and register it as a
# precompiled artifact, so humming never shells out to g++ at serve time.
#
# Same shape as _humming_device_info.ps1: cl.exe builds the source, the result
# lands in humming/_native/x86_64/, and manifest.json has to carry the source
# hash or humming ignores the artifact.
#
#   & D:\code\vllm-windows\_dev\bin\_humming_nvrtc.ps1                       # flashtest venv
#   & D:\code\vllm-windows\_dev\bin\_humming_nvrtc.ps1 -Venv D:\code\vllm-windows

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
$src = Join-Path $humming 'csrc\nvrtc_compile.cpp'
if (-not (Test-Path $src)) { throw "no $src" }

$cuda = $env:CUDA_PATH
$work = Join-Path $env:TEMP 'humming_nvrtc'
New-Item -ItemType Directory -Force -Path $work | Out-Null
Remove-Item -Force "$work\nvrtc_compile.exe" -ErrorAction SilentlyContinue

$cl = @(
    'cl', '/nologo', '/O2', '/std:c++17', '/DNDEBUG', '/MD', '/EHsc', '/utf-8',
    "/I`"$cuda\include`"", "`"$src`"",
    "/Fo`"$work\nvrtc_compile.obj`"",
    "/Fe`"$work\nvrtc_compile.exe`""
)
$buildLog = Join-Path $work 'build.log'
(& cmd.exe /c ($cl -join ' ')) *>&1 | Out-File $buildLog -Encoding utf8
if (-not (Test-Path "$work\nvrtc_compile.exe")) {
    Get-Content $buildLog | Write-Output
    throw 'cl.exe produced no nvrtc_compile.exe'
}
Write-Host ("built nvrtc_compile.exe  size=" + (Get-Item "$work\nvrtc_compile.exe").Length)

$native = Join-Path $humming '_native\x86_64'
New-Item -ItemType Directory -Force -Path $native | Out-Null
Copy-Item "$work\nvrtc_compile.exe" "$native\nvrtc_compile.exe" -Force

$hash = & $py -c "import humming.utils.jit as j; print(j.hash_path_content(r'$src', releative=True))"
$manifestPath = Join-Path $native 'manifest.json'
$manifest = @{}
if (Test-Path $manifestPath) {
    foreach ($p in (Get-Content $manifestPath -Raw | ConvertFrom-Json).PSObject.Properties) {
        $manifest.($p.Name) = $p.Value
    }
}
$manifest.'nvrtc_compile.exe' = "$hash".Trim()
($manifest | ConvertTo-Json -Depth 5) | Set-Content -Path $manifestPath -Encoding ascii
Write-Host "manifest -> nvrtc_compile.exe = $($manifest.'nvrtc_compile.exe')"

$probe = Join-Path $work 'probe.py'
Set-Content -Path $probe -Encoding ascii -Value @"
import humming.utils.nvrtc as n
print('nvrtc lib :', n.get_nvrtc_library_path())
print('binary    :', n.may_build_nvrtc_compile_binary())
"@
& $py $probe
