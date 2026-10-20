<#
One-off PP/TP probe launcher. Copies the flag set that
_dev/bin/_flashnext_struct_serve.ps1 composes for production, but adds
--pipeline-parallel-size / --tensor-parallel-size so the two can be compared
on identical settings.  Nothing here is production: it writes only new log
files and never touches the service scripts or their config.

  & _dev\probe\_pp_tp_smoke.ps1 -Mode pp -Size 2 -LogPrefix pp2_eager_nople
#>
[CmdletBinding()]
param(
    [ValidateSet('pp', 'tp')] [string]$Mode = 'pp',
    [int]$Size = 2,
    [int]$MaxLen = 16384,
    [int]$MaxSeqs = 4,
    [int]$BatchedTokens = 2048,
    [double]$MemUtil = 0.90,
    [int]$Port = 9401,
    [string]$LogPrefix = 'pp2',
    [string]$Model = 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP',
    [string]$ServedName = 'Qwen3.8-Flash-Next',
    [string]$VenvDir = 'D:\code\vllm-windows\.flashnext-root',
    [string]$LoadFormat = '',
    [switch]$Mtp,
    [switch]$NoPle,
    [switch]$NoGraphs,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# .venv/Lib/site-packages holds a stale upstream vLLM copy (0.29.1.dev0) that wins
# whenever the repo root is not on sys.path first; pin the fork tree explicitly so
# a run can never silently execute the wrong engine.
$env:PYTHONPATH = 'D:\code\vllm-windows'

function EscJson([string]$Json) {
    return ($Json -replace '"', '\"')
}
$repo = 'D:\code\vllm-windows'
$py = "$VenvDir\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) { throw "python not found: $py" }

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:TORCH_CUDA_ARCH_LIST = '8.0'
$env:VLLM_TARGET_DEVICE = 'cuda'
$env:CUDA_PATH = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$env:CUDA_HOME = $env:CUDA_PATH
$env:CUDA_ROOT = $env:CUDA_PATH
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:False'
$env:VLLM_PROBE_INC_HUMMING_LOWBIT = '1'
$env:VSLANG = '1033'  # MSVC must emit English: flashinfer reads compiler output as utf-8
$env:VLLM_NCCL_SO_PATH = 'C:\nccl-windows\install\bin\nccl.dll'
# flashinfer/humming JIT shell out to ninja, and nvcc needs cl.exe: without the
# toolchain on PATH the warmup dies with FileNotFoundError [WinError 2].
. 'D:\code\vllm-windows\_dev\bin\_env.ps1'
$env:PATH = "$VenvDir\.venv\Scripts;$env:PATH"
# No P2P on this box (NCCL says so itself); do not let it probe for one.
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_IGNORE_DISABLED_P2P = '1'

$argv = @(
    '-u', '-m', 'vllm.entrypoints.openai.api_server',
    '--model', $Model,
    '--served-model-name', $ServedName,
    '--trust-remote-code',
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--dtype', 'bfloat16',
    '--max-model-len', "$MaxLen",
    '--max-num-seqs', "$MaxSeqs",
    '--max-num-batched-tokens', "$BatchedTokens",
    '--gpu-memory-utilization', "$MemUtil",
    '--mamba-cache-mode', 'align',
    '--language-model-only',
    '--disable-custom-all-reduce'
)
if ($Mode -eq 'pp') {
    $argv += @('--pipeline-parallel-size', "$Size")
} else {
    $argv += @('--tensor-parallel-size', "$Size")
}
if (-not $NoGraphs) {
    $argv += @('--compilation-config', (EscJson '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'))
} else {
    $argv += '--enforce-eager'
}
if (-not $NoPle) {
    $pleDll = 'D:/code/vllm-windows/_dev/out/ple_ssd_io/ple_ssd_io_win.dll'
    if (-not (Test-Path -LiteralPath 'D:\code\vllm-windows\_dev\out\ple_ssd_io\ple_ssd_io_win.dll')) {
        throw 'ple_ssd_io_win.dll missing'
    }
    # Start-Process -ArgumentList neither quotes elements nor keeps embedded
    # double quotes, so the JSON needs the same backslash escape the production
    # launcher uses (its EscJson).  No spaces either, for the same reason.
    $pleCfg = '{"ple_ssd_offload":true,"ple_ssd_native_library":"' + $pleDll +
        '","ple_ssd_io_depth":512,"ple_ssd_cache_mb":8192,"ple_ssd_workers":16' +
        ',"ple_ssd_prefetch_tokens":16384}'
    $argv += @('--additional-config', (EscJson $pleCfg))
}
if ($LoadFormat) {
    $argv += @('--load-format', $LoadFormat)
}
if ($Mtp) {
    $argv += @('--speculative-config', (EscJson '{"method":"mtp","num_speculative_tokens":3}'))
}

$logDir = "$repo\_dev\out\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$out = "$logDir\${LogPrefix}_$stamp.stdout.log"
$err = "$logDir\${LogPrefix}_$stamp.stderr.log"
Write-Host "MODE=$Mode SIZE=$Size MaxLen=$MaxLen MemUtil=$MemUtil Mtp=$Mtp Ple=$(-not $NoPle) Graphs=$(-not $NoGraphs)"
Write-Host "CMD: $py $($argv -join ' ')"
Write-Host "LOG: $out"
if ($DryRun) { exit 0 }

$p = Start-Process -FilePath $py -ArgumentList $argv -PassThru -NoNewWindow `
    -RedirectStandardOutput $out -RedirectStandardError $err
Set-Content -LiteralPath "$logDir\$LogPrefix.pid" -Value $p.Id
Write-Host "PID: $($p.Id)"
