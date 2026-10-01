# Build NCCL for Windows (required for tensor / pipeline parallelism).
#
# Windows torch ships WITHOUT NCCL, but vLLM's pynccl does not go through torch:
# it dlopen()s the library itself via ctypes, using VLLM_NCCL_SO_PATH.
# So a local nccl.dll is all that is missing for --tensor-parallel-size to work.
#
# Source: https://github.com/SystemPanic/nccl-windows (branch "nccl-windows", NOT main)
#
# Usage:
#   & D:\code\vllm-windows\_dev\bin\_build_nccl.ps1                 # build for sm_80 + sm_86
#   & D:\code\vllm-windows\_dev\bin\_build_nccl.ps1 -Arch 8.6       # current card only (faster)
#   & D:\code\vllm-windows\_dev\bin\_build_nccl.ps1 -Reclone        # wipe and re-clone
#
# After a successful build, _env.ps1 auto-detects the dll, so _run.ps1 / _serve_mtp.py
# pick it up with no further changes.

param(
    # CUDA arch list. Mixing archs is explicitly supported upstream
    # ("you can mix different archs if you have distinct GPU archs").
    # 80 = CMP 170HX (GA100), 86 = RTX 3090 (GA102).
    [string]$Arch = '80;86',

    [string]$Dir = 'C:\nccl-windows',

    # Wipe $Dir and re-clone.
    [switch]$Reclone,

    # Skip the clone step (use existing checkout).
    [switch]$NoClone
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\_env.ps1"

$dll = Join-Path $Dir 'install\bin\nccl.dll'

Write-Host "=== NCCL for Windows build ===" -ForegroundColor Cyan
Write-Host "arch       : $Arch"
Write-Host "source dir : $Dir"
Write-Host "install    : $(Join-Path $Dir 'install')"
Write-Host "target dll : $dll"
Write-Host ""

foreach ($tool in @('git', 'cmake', 'ninja')) {
    $p = Get-Command $tool -ErrorAction SilentlyContinue
    if (-not $p) { throw "$tool not found on PATH. cmake/ninja live in $REPO\.venv\Scripts" }
    Write-Host ("{0,-6} {1}" -f $tool, $p.Source)
}
# ninja must be able to spawn cl.exe; _env.ps1 already prepends the MSVC bin dir.
Write-Host ("nvcc   " + (Get-Command nvcc -ErrorAction SilentlyContinue).Source)
Write-Host ""

if ($Reclone -and (Test-Path $Dir)) {
    Write-Host "Removing $Dir (-Reclone) ..."
    Remove-Item -Recurse -Force $Dir
}

if (-not $NoClone) {
    if (Test-Path (Join-Path $Dir '.git')) {
        Write-Host "Repo already present, skipping clone. Use -Reclone to start over."
    } else {
        Write-Host "Cloning SystemPanic/nccl-windows (branch nccl-windows) ..."
        # core.longpaths=true is required; NCCL has deep source paths.
        git clone --single-branch --branch nccl-windows `
            https://github.com/SystemPanic/nccl-windows.git $Dir
        if ($LASTEXITCODE -ne 0) { throw "git clone failed ($LASTEXITCODE)" }
    }
}

Push-Location $Dir
try {
    Write-Host ""
    Write-Host "Configuring ..." -ForegroundColor Cyan
    & cmake -S . -B build -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        "-DCMAKE_CUDA_ARCHITECTURES=$Arch" `
        -DCMAKE_INSTALL_PREFIX=install
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

    Write-Host ""
    Write-Host "Building (MAX_JOBS=$env:MAX_JOBS) ..." -ForegroundColor Cyan
    & cmake --build build --parallel --target install
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

if (-not (Test-Path $dll)) { throw "Build finished but $dll is missing." }

$sizeMB = [math]::Round((Get-Item $dll).Length / 1MB, 1)
Write-Host ""
Write-Host "=== OK ===" -ForegroundColor Green
Write-Host "nccl.dll : $dll ($sizeMB MB)"

# Load-probe through the exact code path vLLM uses (ctypes, not torch).
$probe = @"
import ctypes, sys
try:
    lib = ctypes.CDLL(r'$dll')
    fn = getattr(lib, 'ncclGetVersion', None)
    if fn is None:
        print('LOADED but ncclGetVersion missing'); sys.exit(1)
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    v = ctypes.c_int()
    rc = fn(ctypes.byref(v))
    print('ctypes load OK, rc=%d nccl version=%d' % (rc, v.value))
except Exception as e:
    print('LOAD FAILED:', type(e).__name__, e); sys.exit(1)
"@
$probe | Out-File -Encoding ascii "$env:TEMP\_nccl_probe.py"
& "$PSScriptRoot\.venv\Scripts\python.exe" "$env:TEMP\_nccl_probe.py"
Remove-Item "$env:TEMP\_nccl_probe.py" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Next: _env.ps1 auto-detects this dll, so nothing else to set."
Write-Host "With both cards installed:" -ForegroundColor Cyan
Write-Host "  & D:\code\vllm-windows\_dev\bin\_run.ps1 -Serve -Model D:\models\Qwen3.8-27B-W4A16-AutoRound-fast \``"
Write-Host "      -MaxLen 32768 -MaxSeqs 4 -BatchedTokens 4096 -MTP -Tp 2"
Write-Host ""
Write-Host "Note: for a 2-card run, size the model for the SMALLER card (24 GB) and leave" -ForegroundColor Yellow
Write-Host "      gpu_memory_utilization conservative -- each rank must hold an equal shard." -ForegroundColor Yellow
