# AutoRound 3bpw-MTP at TP=2 on the two CMP 170HX (the PLE table stays on SSD,
# each rank streaming only its own vocabulary rows).
#
#   & _dev\probe\_autoround_tp2_smoke.ps1 -NoMtp        # isolate the PLE change
#   & _dev\probe\_autoround_tp2_smoke.ps1               # production shape
#
# Mirrors the production argv except -Port, the served name, TP=2 and the
# missing-P2P flags. Stop it with _dev\probe\_autoround_tp2_stop.ps1.
[CmdletBinding()]
param(
    [string]$Model = 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP',
    [string]$ServedName = 'Qwen3.8-Flash-Next-AutoRound-TP2',
    [int]$Port = 9401,
    [int]$Size = 2,
    [string]$Gpus = '1,2',
    [int]$MaxLen = 262144,
    [int]$MaxSeqs = 4,
    [int]$BatchedTokens = 2048,
    [double]$MemUtil = 0.96,
    [int]$MtpTokens = 3,
    [int]$PleCacheMb = 8192,
    [switch]$NoMtp,
    [switch]$NoEager,
    [switch]$EnableExpertParallel,
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
$env:CUDA_VISIBLE_DEVICES = $Gpus
$env:VLLM_TARGET_DEVICE = 'cuda'
$env:VSLANG = '1033'
# No P2P between the two 170HX; NCCL reports it and we keep it that way.
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_IGNORE_DISABLED_P2P = '1'
$env:VLLM_NCCL_SO_PATH = 'C:\nccl-windows\install\bin\nccl.dll'
$env:PATH = "$VenvDir\.venv\Scripts;$env:PATH"

$additional = EscJson ("{`"ple_ssd_offload`":true," +
    "`"ple_ssd_native_library`":`"D:/code/vllm-windows/_dev/out/ple_ssd_io/ple_ssd_io_win.dll`"," +
    "`"ple_ssd_io_depth`":512,`"ple_ssd_cache_mb`":$PleCacheMb," +
    "`"ple_ssd_workers`":16,`"ple_ssd_prefetch_tokens`":16384}")
$mm = EscJson '{"max_pixels":1310720}'

$argv = @(
    '-u', '-m', 'vllm.entrypoints.openai.api_server',
    '--model', $Model,
    '--served-model-name', $ServedName,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--data-parallel-rpc-port', "$(29950 + ($Port % 1000))",
    '--dtype', 'bfloat16',
    '--max-model-len', "$MaxLen",
    '--max-num-seqs', "$MaxSeqs",
    '--max-num-batched-tokens', "$BatchedTokens",
    '--gpu-memory-utilization', "$MemUtil",
    '--mamba-cache-mode', 'align',
    '--tensor-parallel-size', "$Size",
    '--disable-custom-all-reduce',
    '--limit-mm-per-prompt', (EscJson '{"image":{"count":999}}'),
    '--mm-processor-kwargs', $mm,
    '--additional-config', $additional,
    '--enable-auto-tool-choice',
    '--tool-call-parser', 'qwen3_xml',
    '--reasoning-parser', 'qwen3',
    '--default-chat-template-kwargs', (EscJson '{"enable_thinking":false}'),
    '--enable-prompt-tokens-details'
)
if ($EnableExpertParallel) { $argv += '--enable-expert-parallel' }
if (-not $NoMtp) {
    $spec = EscJson ("{`"method`":`"mtp`",`"num_speculative_tokens`":$MtpTokens," +
        "`"adaptive_k`":{`"min`":1,`"max`":$MtpTokens,`"log_interval`":200}}")
    $argv += @('--speculative-config', $spec)
}

$logDir = "$repo\_dev\out\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path "$repo\_dev\out\autoround_tp2" | Out-Null
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$out = "$logDir\autoround_tp2_$stamp.stdout.log"
$err = "$logDir\autoround_tp2_$stamp.stderr.log"

Write-Host "AutoRound TP=$Size gpus=$Gpus port=$Port maxlen=$MaxLen mtp=$(-not $NoMtp) eager=$NoEager"
Write-Host "CMD: $py $($argv -join ' ')"
Write-Host "LOG: $out"
if ($DryRun) { exit 0 }

$p = Start-Process -FilePath $py -ArgumentList $argv -PassThru -NoNewWindow `
    -RedirectStandardOutput $out -RedirectStandardError $err
"$($p.Id) $out" | Set-Content -Path "$repo\_dev\out\autoround_tp2\last"
Write-Host "pid $($p.Id)"
