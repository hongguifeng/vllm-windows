# One-off launcher for the single-card AutoRound 3bpw experiments: DFlash 2
# drafter (P2), draft-count scheduling (P3), scheduler/attention knobs (P4) and
# the KV offload tier (P5). Production parity except for what a switch changes.
#
#   & _dev\probe\_flashnext_arm.ps1 -Spec dflash -SpecN 7
#   & _dev\probe\_flashnext_arm.ps1 -Spec mtp -Schedule '[[1,2,7],[3,4,5],[5,8,3]]'
#   & _dev\probe\_flashnext_arm.ps1 -Spec none -MaxBatched 8192 -BlockSize 1680
#   & _dev\probe\_flashnext_arm.ps1 -Spec dflash -KvOffloadGb 32
#
# Stop with _dev\probe\_flashnext_arm_stop.ps1.
[CmdletBinding()]
param(
    [string]$Model = 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP',
    [string]$Drafter = 'D:\models\Qwen3.8-Flash-Next-DFlash2',
    [string]$ServedName = 'Qwen3.8-Flash-Next-Arm',
    [int]$Port = 9402,
    [int]$Gpu = 2,
    [ValidateSet('dflash', 'mtp', 'none')]
    [string]$Spec = 'dflash',
    [int]$SpecN = 7,
    [string]$Schedule = '',
    [int]$MaxLen = 262144,
    [int]$MaxSeqs = 4,
    [int]$MaxBatched = 2048,
    [double]$MemUtil = 0.96,
    [int]$BlockSize = 0,
    [int]$PrefixMatchUnit = 0,
    [int]$LongPrefillThreshold = -1,
    [string]$IndexerKvDtype = '',
    [int]$KvOffloadGb = 0,
    # Caps the GPU KV pool so evictions (and therefore tier refills) happen at a
    # size a test can reach. 0 leaves the pool at gpu_memory_utilization's size.
    [int]$KvCacheGiB = 0,
    [int]$PleCacheMb = 8192,
    # Production runs the acceptance-adaptive policy; the recipe's default is a
    # fixed draft count, so arms need to pick one explicitly.
    [switch]$AdaptiveK,
    [string]$LogPrefix = 'arm',
    [string]$VenvDir = 'D:\code\vllm-windows\.flashnext-root',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\code\vllm-windows'

function EscJson([string]$Json) { return ($Json -replace '"', '\"') }

$py = "$VenvDir\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) { throw "python not found: $py" }

. "$repo\_dev\bin\_env.ps1"
$env:PYTHONPATH = $repo
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$env:VLLM_TARGET_DEVICE = 'cuda'
$env:VSLANG = '1033'
$env:PATH = "$VenvDir\.venv\Scripts;$env:PATH"

$additional = EscJson ("{`"ple_ssd_offload`":true," +
    "`"ple_ssd_native_library`":`"D:/code/vllm-windows/_dev/out/ple_ssd_io/ple_ssd_io_win.dll`"," +
    "`"ple_ssd_io_depth`":512,`"ple_ssd_cache_mb`":$PleCacheMb," +
    "`"ple_ssd_workers`":16,`"ple_ssd_prefetch_tokens`":16384}")

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
    '--max-num-batched-tokens', "$MaxBatched",
    '--gpu-memory-utilization', "$MemUtil",
    '--mamba-cache-mode', 'align',
    '--limit-mm-per-prompt', (EscJson '{"image":{"count":999}}'),
    '--mm-processor-kwargs', (EscJson '{"max_pixels":1310720}'),
    '--additional-config', $additional,
    '--enable-auto-tool-choice',
    '--tool-call-parser', 'qwen3_xml',
    '--reasoning-parser', 'qwen3',
    '--default-chat-template-kwargs', (EscJson '{"enable_thinking":false}'),
    '--enable-prompt-tokens-details'
)
if ($BlockSize -gt 0) { $argv += @('--block-size', "$BlockSize") }
if ($PrefixMatchUnit -gt 0) { $argv += @('--prefix-match-unit', "$PrefixMatchUnit") }
if ($LongPrefillThreshold -ge 0) {
    $argv += @('--long-prefill-token-threshold', "$LongPrefillThreshold")
}
if ($IndexerKvDtype) {
    $argv += @('--attention-config', (EscJson "{`"indexer_kv_dtype`":`"$IndexerKvDtype`"}"))
}
if ($KvOffloadGb -gt 0) {
    $kv = EscJson ("{`"kv_connector`":`"OffloadingConnector`"," +
        "`"kv_role`":`"kv_both`"," +
        "`"kv_connector_extra_config`":{`"cpu_bytes_to_use`":$($KvOffloadGb * 1GB)}}")
    $argv += @('--kv-transfer-config', $kv)
}
if ($KvCacheGiB -gt 0) {
    $argv += @('--kv-cache-memory-bytes', "$([long]($KvCacheGiB * 1GB))")
}
if ($Spec -ne 'none') {
    $specText = "{`"method`":`"$Spec`""
    if ($Spec -eq 'dflash') {
        $drafterPath = $Drafter -replace '\\', '/'
        $specText += ",`"model`":`"$drafterPath`""
    }
    $specText += ",`"num_speculative_tokens`":$SpecN"
    if ($AdaptiveK) {
        $specText += ",`"adaptive_k`":{`"min`":1,`"max`":$SpecN,`"log_interval`":200}"
    }
    if ($Schedule) {
        $specText += ",`"num_speculative_tokens_per_batch_size`":$Schedule"
    }
    if ($KvOffloadGb -gt 0) { $specText += ",`"disable_eagle_block_drop`":true" }
    $specText += "}"
    $argv += @('--speculative-config', (EscJson $specText))
}

$logDir = "$repo\_dev\out\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path "$repo\_dev\out\arms" | Out-Null
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$out = "$logDir\${LogPrefix}_$stamp.stdout.log"
$err = "$logDir\${LogPrefix}_$stamp.stderr.log"

Write-Host "arm spec=$Spec n=$SpecN sched=$Schedule gpu=$Gpu port=$Port batched=$MaxBatched block=$BlockSize pmu=$PrefixMatchUnit kvoff=$KvOffloadGb"
Write-Host "CMD: $py $($argv -join ' ')"
Write-Host "LOG: $out"
if ($DryRun) { exit 0 }

$p = Start-Process -FilePath $py -ArgumentList $argv -PassThru -NoNewWindow `
    -RedirectStandardOutput $out -RedirectStandardError $err
"$($p.Id) $out" | Set-Content -Path "$repo\_dev\out\arms\last"
Write-Host "pid $($p.Id)"
