# One-off Windows launcher for the EXL3 checkpoint on 2x CMP 170HX (TP=2).
#
#   & _dev\probe\_exl3_tp2_smoke.ps1 -NoGraphs          # first bring-up, eager
#   & _dev\probe\_exl3_tp2_smoke.ps1                    # PIECEWISE graphs
#   & _dev\probe\_exl3_tp2_smoke.ps1 -Mtp               # + MTP draft (SPEC_N=3)
#
# Writes only new logs under _dev\out\logs and never touches the production
# service scripts. Stop it with _dev\probe\_exl3_tp2_stop.ps1.
[CmdletBinding()]
param(
    [string]$Model = 'D:\models\Qwen3.8-Flash-Next-exl3',
    [int]$Port = 9400,
    [int]$MaxLen = 32768,
    [int]$MaxSeqs = 4,
    [int]$BatchedTokens = 2048,
    [double]$MemUtil = 0.95,
    [switch]$NoGraphs,
    [switch]$Mtp,
    [int]$MtpTokens = 3,
    [string]$LogPrefix = 'exl3_tp2',
    [string]$VenvDir = 'D:\code\vllm-windows\.flashnext-root',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\code\vllm-windows'

function EscJson([string]$Json) { return ($Json -replace '"', '\"') }

$py = "$VenvDir\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) { throw "python not found: $py" }

# Toolchain for the flashinfer/humming JIT (cl.exe, ninja).
. "$repo\_dev\bin\_env.ps1"

$env:PYTHONPATH = $repo
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '1,2'
$env:VLLM_TARGET_DEVICE = 'cuda'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:False'
$env:VSLANG = '1033'
# No P2P between the two 170HX; NCCL said so itself in the PP/TP smoke.
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_IGNORE_DISABLED_P2P = '1'
$env:VLLM_NCCL_SO_PATH = 'C:\nccl-windows\install\bin\nccl.dll'
$env:VLLM_EXL3_PLE_PLACEMENT = 'ssd'
# The kernel set the recipe serves with (serve.sh defaults); the library
# defaults are the slow baseline kernels.
$env:VLLM_EXL3_PLE_OVERLAP = '1'
$env:VLLM_EXL3_PLE_NATIVE_READ = '1'
$env:VLLM_EXL3_MOE_V2 = '1'
$env:VLLM_EXL3_DENSE_SPLITK = '1'
$env:VLLM_EXL3_DENSE_MMA = '1'
$env:VLLM_EXL3_DENSE_PREFILL_TC = '1'
$env:VLLM_EXL3_MOE_PREFILL_TC = '1'
$env:PATH = "$VenvDir\.venv\Scripts;$env:PATH"

$argv = @(
    '-u', '-m', 'vllm.entrypoints.openai.api_server',
    '--model', $Model,
    '--served-model-name', 'Qwen3.8-Flash-Next-EXL3',
    '--trust-remote-code',
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--dtype', 'bfloat16',
    '--max-model-len', "$MaxLen",
    '--max-num-seqs', "$MaxSeqs",
    '--max-num-batched-tokens', "$BatchedTokens",
    '--gpu-memory-utilization', "$MemUtil",
    '--mamba-cache-mode', 'align',
    '--tensor-parallel-size', '2',
    '--enable-expert-parallel',
    '--disable-custom-all-reduce',
    '--load-format', 'exl3',
    '--mm-encoder-tp-mode', 'data'
)
if ($NoGraphs) {
    $argv += '--enforce-eager'
} else {
    $argv += @('--compilation-config', (EscJson '{"cudagraph_mode":"PIECEWISE"}'))
}
if ($Mtp) {
    $argv += @(
        '--speculative-config',
        (EscJson "{`"method`":`"mtp`",`"num_speculative_tokens`":$MtpTokens}")
    )
}

$logDir = "$repo\_dev\out\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$out = "$logDir\${LogPrefix}_$stamp.stdout.log"
$err = "$logDir\${LogPrefix}_$stamp.stderr.log"
Write-Host "EXL3 TP2 port=$Port MaxLen=$MaxLen MemUtil=$MemUtil Mtp=$Mtp Graphs=$(-not $NoGraphs)"
Write-Host "CMD: $py $($argv -join ' ')"
Write-Host "LOG: $out"
if ($DryRun) { exit 0 }

$p = Start-Process -FilePath $py -ArgumentList $argv -PassThru -NoNewWindow `
    -RedirectStandardOutput $out -RedirectStandardError $err
"$($p.Id) $out" | Set-Content -Path "$repo\_dev\out\logs\exl3_tp2.last"
Write-Host "pid $($p.Id)"
