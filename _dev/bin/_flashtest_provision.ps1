# Provision an isolated venv for the flash-next worktree build.
#
#   & D:\code\vllm-windows\_dev\bin\_flashtest_provision.ps1            # full run
#   & D:\code\vllm-windows\_dev\bin\_flashtest_provision.ps1 -Recreate  # throw the venv away first
#
# The main repo's .venv stays untouched: it is what serves the 27B, and a
# half-broken flash-next build must not be able to take it down.
#
# Steps mirror _deps.ps1 (proven on this rig) but resolve against the worktree's
# own requirements, which now carry sys_platform branches for win32.

[CmdletBinding()]
param(
    [string]$Repo = 'D:\code\vllm-windows',
    [switch]$Recreate
)

$ErrorActionPreference = 'Continue'
$log = 'D:\code\vllm-windows\_dev\out\_flashtest_deps.log'
if (-not (Test-Path 'D:\code\vllm-windows\_dev\out')) {
    New-Item -ItemType Directory -Path 'D:\code\vllm-windows\_dev\out' -Force | Out-Null
}

"=== PROVISION START $(Get-Date) ===" | Out-File $log -Encoding utf8
function Log($t) { $t | Out-File $log -Append -Encoding utf8 }

$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
Log "uv = $uv"

$venv = "$Repo\.venv"
if ($Recreate -and (Test-Path $venv)) {
    Log "--- removing $venv ---"
    Remove-Item -Recurse -Force $venv
}
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    Log "--- uv venv $venv --python 3.12 ---"
    & $uv venv $venv --python 3.12 *>&1 | Out-File $log -Append -Encoding utf8
    Log "--- exit=$LASTEXITCODE ---"
}
$py = "$venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Log "FATAL: no python at $py"
    Write-Host "no venv python at $py" -ForegroundColor Red
    exit 1
}
if (-not $uv) {
    Log "FATAL: uv not on PATH"
    Write-Host "uv not on PATH" -ForegroundColor Red
    exit 1
}

$TORCH_INDEX = 'https://download.pytorch.org/whl/cu130'

function Step($title, [scriptblock]$block) {
    "" | Out-File $log -Append -Encoding utf8
    "=== $title ===" | Out-File $log -Append -Encoding utf8
    & $block *>&1 | Out-File $log -Append -Encoding utf8
    "--- exit=$LASTEXITCODE ---" | Out-File $log -Append -Encoding utf8
}

# 1) torch stack first, straight off the cu130 index. The build needs torch
#    present because we install vLLM with --no-build-isolation.
Step 'step1 torch stack' {
    & $uv pip install --python $py torch==2.11.0+cu130 torchaudio==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url $TORCH_INDEX
}

# 2) build requirements (cmake, ninja, setuptools-rust, the win32 torch pin).
#    unsafe-best-match: the cu130 index carries stale copies of PyPI tools (it
#    has cmake 3.25.0 only) and uv otherwise refuses to look past the first
#    index that mentions a package.
Step 'step2 build reqs' {
    & $uv pip install --python $py -r "$Repo\requirements\build\cuda.txt" --extra-index-url $TORCH_INDEX --index-strategy unsafe-best-match
}

# 3) cuda requirements -- pulls common.txt too
Step 'step3 cuda reqs' {
    & $uv pip install --python $py -r "$Repo\requirements\cuda.txt" --extra-index-url $TORCH_INDEX --index-strategy unsafe-best-match
}

# 4) windows-only extras (winloop, triton-windows, xformers, portalocker)
Step 'step4 windows reqs' {
    & $uv pip install --python $py -r "$Repo\requirements\windows.txt" --extra-index-url $TORCH_INDEX --index-strategy unsafe-best-match
}

"" | Out-File $log -Append -Encoding utf8
"=== PROVISION DONE $(Get-Date) ===" | Out-File $log -Append -Encoding utf8
& $uv pip list --python $py 2>&1 | Out-File $log -Append -Encoding utf8

Write-Host "log: $log" -ForegroundColor DarkGray
Write-Host "verify with: & D:\code\vllm-windows\_dev\bin\_flashtest_env.ps1; & $py -c 'import torch;print(torch.__version__)'"
