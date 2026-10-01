# Build the flash-next worktree (origin/main + PLE-SSD patch + win32 port).
#
#   & D:\code\vllm-windows\_dev\bin\_flashtest_build.ps1              # full build (~37 min)
#   & D:\code\vllm-windows\_dev\bin\_flashtest_build.ps1 -DryRun      # toolchain probe only
#   & D:\code\vllm-windows\_dev\bin\_flashtest_build.ps1 -Clean       # drop build/ and .so/.pyd first
#
# Incremental rebuilds are the norm afterwards: ninja reuses everything in
# build/, so a one-file CUDA change costs seconds rather than the full pass.

[CmdletBinding()]
param(
    [string]$Repo = 'D:\code\vllm-windows',
    [switch]$DryRun,
    [switch]$Clean
)

$ErrorActionPreference = 'Continue'
$log = 'D:\code\vllm-windows\_dev\out\_flashtest_build.log'
function Log($t) { $t | Out-File $log -Append -Encoding utf8 }

. D:\code\vllm-windows\_dev\bin\_flashtest_env.ps1 -Repo $Repo

'=== BUILD START ' + (Get-Date) | Out-File $log -Encoding utf8
Log "repo = $Repo"

if ($Clean) {
    Log '--- cleaning build/ and in-tree extensions ---'
    if (Test-Path "$Repo\build") { Remove-Item -Recurse -Force "$Repo\build" }
    Get-ChildItem "$Repo\vllm" -Recurse -Include '*.so','*.pyd' -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

$py = "$Repo\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Log "FATAL: no venv python at $py -- run _flashtest_provision.ps1 first"
    Write-Host "no venv python at $py -- run _flashtest_provision.ps1" -ForegroundColor Red
    exit 1
}
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    Log "FATAL: uv not on PATH (a uv-created venv has no pip inside it)"
    Write-Host "uv not on PATH" -ForegroundColor Red
    exit 1
}

Log '--- toolchain check ---'
(& where.exe cl.exe)   *>&1 | Out-File $log -Append -Encoding utf8
(& where.exe nvcc.exe) *>&1 | Out-File $log -Append -Encoding utf8
(& where.exe cmake.exe) *>&1 | Out-File $log -Append -Encoding utf8
(& where.exe ninja.exe) *>&1 | Out-File $log -Append -Encoding utf8
(& $py -c "import sys,torch;print('py',sys.version.split()[0],'| torch',torch.__version__,'| cuda',torch.version.cuda)") *>&1 | Out-File $log -Append -Encoding utf8

if ($DryRun) {
    Get-Content $log | Write-Output
    exit 0
}

Log '--- uninstall any vllm in this venv ---'
(& $uv pip uninstall vllm --python $py) *>&1 | Out-File $log -Append -Encoding utf8

Log '--- uv pip install . --no-build-isolation -v ---'
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
Push-Location $Repo
(& $uv pip install . --no-build-isolation -v --python $py) *>&1 | Out-File $log -Append -Encoding utf8
$rc = $LASTEXITCODE
Pop-Location
Log "=== PIP EXIT=$rc ==="

Log '--- extension modules produced ---'
Get-ChildItem "$Repo\vllm" -Recurse -Include '*.pyd','*.so' -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty FullName | Out-File $log -Append -Encoding utf8

'=== BUILD DONE ' + (Get-Date) + ' rc=' + $rc | Out-File $log -Append -Encoding utf8
Write-Host "build rc=$rc   log: $log" -ForegroundColor $(if ($rc -eq 0) { 'Green' } else { 'Red' })
exit $rc
