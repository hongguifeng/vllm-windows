$ErrorActionPreference = 'Continue'

$REPO    = 'D:\code\vllm-windows'
$DEV     = "$REPO\_dev"      # 本地脚本/文档/输出的唯一落点（bin|bench|test|probe|patches|docs|out）

# Toolset / SDK / CUDA are discovered, not pinned: a stale version silently leaves
# nvcc without a host compiler ("nvcc fatal : Cannot find compiler 'cl.exe' in PATH").
function _LatestDir([string]$root, [string]$leaf) {
    if (-not (Test-Path -LiteralPath $root)) { return $null }
    $c = Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
         Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName $leaf) }
    $c = $c | Sort-Object { try { [Version]$_.Name } catch { $_.Name } } -Descending
    if ($c) { return $c[0].FullName }
    return $null
}

$VSROOT  = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path -LiteralPath $vswhere) {
    $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($found) { $VSROOT = "$found".Trim() }
}
$MSVC    = _LatestDir "$VSROOT\VC\Tools\MSVC" 'bin\Hostx64\x64\cl.exe'
if (-not $MSVC) { $MSVC = "$VSROOT\VC\Tools\MSVC\14.44.35207" }
$SDKROOT = 'C:\Program Files (x86)\Windows Kits\10'
$sdkDir  = _LatestDir "$SDKROOT\Include" 'um'
if ($sdkDir) { $SDKVER = Split-Path -Leaf $sdkDir } else { $SDKVER = '10.0.26100.0' }
$cudaDir = _LatestDir 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' 'bin\nvcc.exe'
if ($cudaDir) { $CUDAROOT = $cudaDir } else { $CUDAROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }

$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'
$env:VCToolsInstallDir  = "$MSVC\"
$env:VCToolsVersion     = Split-Path -Leaf $MSVC
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
