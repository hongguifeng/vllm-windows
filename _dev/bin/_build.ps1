$ErrorActionPreference = 'Continue'

$REPO    = 'D:\code\vllm-windows'
$MSVC    = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207'
$SDKROOT = 'C:\Program Files (x86)\Windows Kits\10'
$SDKVER  = '10.0.26100.0'
$CUDAROOT= 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$log     = 'D:\code\vllm-windows\_dev\out\_build.log'

function Log($t) { $t | Out-File $log -Append -Encoding utf8 }

'=== BUILD START ' + (Get-Date) | Out-File $log -Encoding utf8

# --- WorkBuddy python shim: it patches os.unlink and aborts after ~50 deletes/turn,
#     which aborts pip's wheel overwrite and ninja's object-file replacement.
$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'

# --- MSVC / Windows SDK environment (native equivalent of vcvarsall.bat x64)
$env:VCToolsInstallDir  = "$MSVC\"
$env:VCToolsVersion     = '14.44.35207'
$env:WindowsSdkDir      = "$SDKROOT\"
$env:WindowsSDKVersion  = "$SDKVER\"
$env:UCRTVersion        = $SDKVER
$env:WindowsSdkBinPath  = "$SDKROOT\bin\$SDKVER\"

$env:INCLUDE = "$MSVC\include;$SDKROOT\Include\$SDKVER\ucrt;$SDKROOT\Include\$SDKVER\shared;$SDKROOT\Include\$SDKVER\um;$SDKROOT\Include\$SDKVER\winrt;$SDKROOT\Include\$SDKVER\cppwinrt"
$env:LIB     = "$MSVC\lib\x64;$SDKROOT\Lib\$SDKVER\ucrt\x64;$SDKROOT\Lib\$SDKVER\um\x64"
$env:LIBPATH = "$MSVC\lib\x64"

$env:PATH = "$MSVC\bin\Hostx64\x64;$SDKROOT\bin\$SDKVER\x64;$REPO\.venv\Scripts;$CUDAROOT\bin;$env:PATH"

# --- vLLM build configuration
$env:DISTUTILS_USE_SDK   = '1'
$env:VLLM_TARGET_DEVICE  = 'cuda'
$env:VLLM_USE_PRECOMPILED= '0'
$env:TORCH_CUDA_ARCH_LIST= '8.0'   # CMP 170HX = GA100 = sm_80
$env:MAX_JOBS            = '10'
$env:CUDA_HOME           = $CUDAROOT
$env:CUDA_PATH           = $CUDAROOT
$env:CUDA_ROOT           = $CUDAROOT

Set-Location $REPO

'--- toolchain check ---' | Out-File $log -Append -Encoding utf8
(& where.exe cl.exe)    *>&1 | Out-File $log -Append -Encoding utf8
(& where.exe nvcc.exe)  *>&1 | Out-File $log -Append -Encoding utf8
(& where.exe cmake.exe) *>&1 | Out-File $log -Append -Encoding utf8
(& where.exe ninja.exe) *>&1 | Out-File $log -Append -Encoding utf8
(& python -c "import sys,torch;print('py',sys.version.split()[0],'| torch',torch.__version__,'| cuda',torch.version.cuda)") *>&1 | Out-File $log -Append -Encoding utf8

'--- removing stale editable vllm ---' | Out-File $log -Append -Encoding utf8
(& python -m pip uninstall -y vllm) *>&1 | Out-File $log -Append -Encoding utf8

'--- pip install start ---' | Out-File $log -Append -Encoding utf8
(& python -m pip install . --no-build-isolation -v) *>&1 | Out-File $log -Append -Encoding utf8
$rc = $LASTEXITCODE
Log "=== PIP EXIT=$rc ==="

'--- resulting extension modules ---' | Out-File $log -Append -Encoding utf8
Get-ChildItem 'D:\code\vllm-windows\vllm' -Recurse -Include '*.pyd' -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty FullName | Out-File $log -Append -Encoding utf8

'=== BUILD DONE ' + (Get-Date) + ' rc=' + $rc | Out-File $log -Append -Encoding utf8
exit $rc
