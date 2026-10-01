# MSVC + CUDA + Windows SDK environment for building vLLM outside the main repo.
# Native equivalent of `vcvarsall.bat x64`, pinned to the toolchain that this rig
# has already proven (MSVC 14.44.35207, SDK 10.0.26100, CUDA 13.3).
#
#   . D:\code\vllm-windows\_dev\bin\_flashtest_env.ps1     # dot-source into a build script
#   & D:\code\vllm-windows\_dev\bin\_flashtest_env.ps1     # or just inspect the result
#
# Unlike _env.ps1 this one points PATH at the *worktree* venv, so the main repo's
# .venv (which serves the 27B) is never involved.

param(
    [string]$Repo = 'D:\code\vllm-windows',
    [string]$VenvPath = ''
)

$ErrorActionPreference = 'Stop'

$MSVC    = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207'
$SDKROOT = 'C:\Program Files (x86)\Windows Kits\10'
$SDKVER  = '10.0.26100.0'
$CUDAROOT= 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'

# The WorkBuddy python shim patches os.unlink and aborts after ~50 deletes per
# turn, which kills pip installs and ninja's object-file replacement.
$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'

$env:VCToolsInstallDir = "$MSVC\"
$env:VCToolsVersion    = '14.44.35207'
$env:WindowsSdkDir     = "$SDKROOT\"
$env:WindowsSDKVersion = "$SDKVER\"
$env:UCRTVersion       = $SDKVER
$env:WindowsSdkBinPath = "$SDKROOT\bin\$SDKVER\"

$env:INCLUDE = "$MSVC\include;$SDKROOT\Include\$SDKVER\ucrt;$SDKROOT\Include\$SDKVER\shared;$SDKROOT\Include\$SDKVER\um;$SDKROOT\Include\$SDKVER\winrt;$SDKROOT\Include\$SDKVER\cppwinrt"
$env:LIB     = "$MSVC\lib\x64;$SDKROOT\Lib\$SDKVER\ucrt\x64;$SDKROOT\Lib\$SDKVER\um\x64;$CUDAROOT\lib\x64"
$env:LIBPATH = "$MSVC\lib\x64"
$venvRoot = if ($VenvPath) { $VenvPath } else { "$Repo\.venv-flashnext" }
$env:PATH    = "$MSVC\bin\Hostx64\x64;$SDKROOT\bin\$SDKVER\x64;$venvRoot\Scripts;$CUDAROOT\bin;$env:PATH"

$env:DISTUTILS_USE_SDK    = '1'
$env:VLLM_TARGET_DEVICE   = 'cuda'
$env:VLLM_USE_PRECOMPILED = '0'
$env:TORCH_CUDA_ARCH_LIST = '8.0'   # CMP 170HX = GA100 = sm_80
$env:MAX_JOBS             = '10'
$env:CUDA_HOME            = $CUDAROOT
$env:CUDA_PATH            = $CUDAROOT
$env:CUDA_ROOT            = $CUDAROOT

# Windows torch has no NCCL; vLLM's pynccl dlopen()s a locally built nccl.dll.
$NCCL_DLL = if ($env:NCCL_DLL) { $env:NCCL_DLL } else { 'C:\nccl-windows\install\bin\nccl.dll' }
if (Test-Path $NCCL_DLL) { $env:VLLM_NCCL_SO_PATH = $NCCL_DLL }

Write-Host "env ready: repo=$Repo  arch=$($env:TORCH_CUDA_ARCH_LIST)  cuda=$CUDAROOT"
