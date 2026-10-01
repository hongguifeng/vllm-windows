$ErrorActionPreference = 'Continue'

$REPO    = 'D:\code\vllm-windows'
$DEV     = "$REPO\_dev"      # 本地脚本/文档/输出的唯一落点（bin|bench|test|probe|patches|docs|out）
$MSVC    = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207'
$SDKROOT = 'C:\Program Files (x86)\Windows Kits\10'
$SDKVER  = '10.0.26100.0'
$CUDAROOT= 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'

$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'
$env:VCToolsInstallDir  = "$MSVC\"
$env:VCToolsVersion     = '14.44.35207'
$env:WindowsSdkDir      = "$SDKROOT\"
$env:WindowsSDKVersion  = "$SDKVER\"
$env:UCRTVersion        = $SDKVER
$env:WindowsSdkBinPath  = "$SDKROOT\bin\$SDKVER\"

$env:INCLUDE = "$MSVC\include;$SDKROOT\Include\$SDKVER\ucrt;$SDKROOT\Include\$SDKVER\shared;$SDKROOT\Include\$SDKVER\um;$SDKROOT\Include\$SDKVER\winrt;$SDKROOT\Include\$SDKVER\cppwinrt"
$env:LIB     = "$MSVC\lib\x64;$SDKROOT\Lib\$SDKVER\ucrt\x64;$SDKROOT\Lib\$SDKVER\um\x64"
$env:LIBPATH = "$MSVC\lib\x64"
$env:PATH    = "$MSVC\bin\Hostx64\x64;$SDKROOT\bin\$SDKVER\x64;$REPO\.venv\Scripts;$CUDAROOT\bin;$env:PATH"

$env:DISTUTILS_USE_SDK   = '1'
$env:VLLM_TARGET_DEVICE  = 'cuda'
$env:VLLM_USE_PRECOMPILED= '0'
$env:TORCH_CUDA_ARCH_LIST= '8.0'
$env:MAX_JOBS            = '10'
$env:CUDA_HOME           = $CUDAROOT
$env:CUDA_PATH           = $CUDAROOT
$env:CUDA_ROOT           = $CUDAROOT

# --- NCCL for Windows (required for --tensor-parallel-size / --pipeline-parallel-size) ---
# Windows torch has no NCCL, but vLLM's pynccl dlopen()s the dll itself via ctypes,
# so we only need to point it at a locally built nccl.dll. Build it with:
#     & D:\code\vllm-windows\_dev\bin\_build_nccl.ps1
# Auto-detected here so _run.ps1 and _serve_mtp.py need no changes.
$NCCL_DLL = if ($env:NCCL_DLL) { $env:NCCL_DLL } else { 'C:\nccl-windows\install\bin\nccl.dll' }
if (Test-Path $NCCL_DLL) { $env:VLLM_NCCL_SO_PATH = $NCCL_DLL }
# Leave VLLM_DISTRIBUTED_USE_SPLIT_GROUP unset (default 0): that path hardcodes
# "cpu:gloo,cuda:nccl" and demands torch's own NCCL, which Windows does not have.
