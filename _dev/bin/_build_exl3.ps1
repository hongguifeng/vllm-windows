# Build the EXL3 CUDA extension (vllm/_exl3_C.pyd) for the Windows fork.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File _dev\bin\_build_exl3.ps1
#   powershell ... -File _dev\bin\_build_exl3.ps1 -OutDir 'D:\tmp\exl3' -Jobs 4
#
# Needs MSVC 2022, CUDA 13.x and the repo .venv (torch 2.11+cu130). The
# extension is SM 8.0 only; it builds without a visible GPU.

param(
    [string]$Repo = 'D:\code\vllm-windows',
    [string]$OutDir = '',
    [string]$BuildDir = '',
    [int]$Jobs = 6,
    [switch]$Verbose
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_env.ps1')

# torch's compiler probe decodes cl.exe output with the OEM code page and dies
# on the localized MSVC banner; build_standalone.py checks the ABI itself.
$env:TORCH_DONT_CHECK_COMPILER_ABI = '1'
$env:MAX_JOBS = "$Jobs"

if (-not $OutDir) { $OutDir = Join-Path $Repo 'vllm' }
if (-not $BuildDir) { $BuildDir = Join-Path $Repo '_dev\out\exl3_build' }

$script = Join-Path $Repo 'csrc\exl3\build_standalone.py'
if (-not (Test-Path $script)) { throw "missing $script" }

$buildArgs = @($script, '--out', $OutDir, '--build-dir', $BuildDir)
if ($Verbose) { $buildArgs += '--verbose' }

& (Join-Path $Repo '.venv\Scripts\python.exe') @buildArgs
if ($LASTEXITCODE -ne 0) { throw "build_standalone.py exited $LASTEXITCODE" }

$pyd = Join-Path $OutDir '_exl3_C.pyd'
if (-not (Test-Path $pyd)) { throw "build produced no $pyd" }
$item = Get-Item $pyd
Write-Host "built $pyd ($($item.Length) B, $($item.LastWriteTime))" -ForegroundColor Green
Write-Host ("sha256 " + (Get-FileHash $pyd -Algorithm SHA256).Hash) -ForegroundColor DarkGray
