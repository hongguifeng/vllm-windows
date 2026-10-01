# Build humming's launcher through torch's cpp_extension (MSVC) and then
# register the result as humming's precompiled launcher, so a plain
# `vllm serve` never has to compile anything at start time.
#
#   & D:\code\vllm-windows\_dev\bin\_humming_launcher.ps1                       # flashtest venv
#   & D:\code\vllm-windows\_dev\bin\_humming_launcher.ps1 -Venv D:\code\vllm-windows
#   & D:\code\vllm-windows\_dev\bin\_humming_launcher.ps1 -Rebuild              # ignore the cache

[CmdletBinding()]
param(
    [string]$Venv = 'D:\code\vllm-windows',
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'
. D:\code\vllm-windows\_dev\bin\_flashtest_env.ps1 -Repo $Venv

$py = "$Venv\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "no venv python at $py" }

$sitePackages = & $py -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"
$humming = Join-Path $sitePackages 'humming'
$csrc = Join-Path $humming 'csrc\launcher'
if (-not (Test-Path "$csrc\launcher.cpp")) { throw "no $csrc" }

$cache = Join-Path $env:USERPROFILE '.humming\cache'
if ($Rebuild -and (Test-Path $cache)) {
    Write-Host 'dropping the whole humming JIT cache'
    Remove-Item -Recurse -Force $cache
}

$buildProbe = Join-Path $env:TEMP 'humming_launcher_build.py'
Set-Content -Path $buildProbe -Encoding ascii -Value @"
import glob, os
import humming.ops.utils as u

u.init_humming_launcher()
print('launcher built')
cache = u.jit_utils.get_humming_cache_dir()
hits = glob.glob(os.path.join(cache, 'launcher', '*', '*', 'humming_launcher.pyd'))
print('artifacts:', hits)
"@
& $py $buildProbe
if ($LASTEXITCODE -ne 0) { throw "launcher build failed (exit $LASTEXITCODE)" }

$found = & $py -c "
import glob, os
import humming.ops.utils as u
cache = u.jit_utils.get_humming_cache_dir()
hits = glob.glob(os.path.join(cache, 'launcher', '*', '*', 'humming_launcher.pyd'))
print(hits[-1] if hits else '')
"
$found = "$found".Trim()
if (-not $found -or -not (Test-Path $found)) { throw "no built humming_launcher.pyd under $cache" }
Write-Host ("built launcher: $found  size=" + (Get-Item $found).Length)

$native = Join-Path $humming '_native\x86_64'
New-Item -ItemType Directory -Force -Path $native | Out-Null
# humming looks for libhumming_launcher.so and hands it to torch.ops.load_library,
# which is extension-agnostic on Windows; keep its name so no code has to change.
Copy-Item $found "$native\libhumming_launcher.so" -Force

$hash = & $py -c "import humming.utils.jit as j; print(j.hash_path_content(r'$csrc', releative=True, text_only=False))"
$manifestPath = Join-Path $native 'manifest.json'
$manifest = @{}
if (Test-Path $manifestPath) {
    foreach ($p in (Get-Content $manifestPath -Raw | ConvertFrom-Json).PSObject.Properties) {
        $manifest.($p.Name) = $p.Value
    }
}
$manifest.'libhumming_launcher.so' = "$hash".Trim()
($manifest | ConvertTo-Json -Depth 5) | Set-Content -Path $manifestPath -Encoding ascii
Write-Host "manifest -> libhumming_launcher.so = $($manifest.'libhumming_launcher.so')"

$useProbe = Join-Path $env:TEMP 'humming_launcher_use.py'
Set-Content -Path $useProbe -Encoding ascii -Value @"
import humming.ops.utils as u
u.init_humming_launcher()
print('launcher init OK from the precompiled artifact')
path = u._get_precompiled_launcher_path()
print('precompiled path:', path)
"@
& $py $useProbe
