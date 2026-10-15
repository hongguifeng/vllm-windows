<#
.SYNOPSIS
  Windows-side supervisor for one mini-swe-agent arm. Survives WSL reboots by
  relaunching the WSL-side run from Windows, then drives evaluation.
.DESCRIPTION
  Why a Windows supervisor: every WSL background job dies when WSL restarts, and
  `wsl.exe` calls return as soon as the shell exits, so a plain
  `setsid nohup ... &` launched from a tool call is not enough to survive a reboot.
  This script polls state from Windows and relaunches as needed.
.PARAMETER Target
  Number of instances expected in the subset.
.PARAMETER MaxEmptyRestarts
  How many times an all-empty preds.json may be deleted and restarted before the
  supervisor gives up. This is the guard against the failure mode where every
  episode produces an empty patch: without a cap the supervisor restarts forever.
#>
param(
    [Parameter(Mandatory = $true)][string] $RunTag,
    [Parameter(Mandatory = $true)][string] $Subset,
    [Parameter(Mandatory = $true)][string] $ApiBase,
    [string] $BenchDir = '/home/hong/bench',
    [string] $OutName = '',
    [string] $ModelName = 'openai/Qwen3.8-Flash-Next',
    [int] $Target = 0,
    [int] $Workers = 2,
    [int] $MaxTokens = 1536,
    [int] $StepLimit = 60,
    [ValidateSet('off', 'medium', 'xhigh')][string] $Effort = 'medium',
    [string] $EvalDataset = '',
    [int] $EvalWorkers = 4,
    [int] $EvalTimeout = 2400,
    [string] $ExtraRunArgs = '',
    [int] $MaxEmptyRestarts = 2,
    [int] $PollSeconds = 45,
    [int] $MaxPolls = 400,
    [string] $LogFile = ''
)

$ErrorActionPreference = 'Continue'
if (-not $OutName) { $OutName = $RunTag }
$outDir = "$BenchDir/$OutName"
if (-not $EvalDataset) { $EvalDataset = $Subset }
if (-not $LogFile) { $LogFile = "D:\code\vllm-windows\_dev\out\logs\supervise_$RunTag.log" }
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
function Log($m) { "$((Get-Date -Format 'HH:mm:ss')) $m" | Out-File -FilePath $LogFile -Append }

$think = ''
if ($Effort -ne 'off') { $think = "--enable-thinking --reasoning-effort $Effort" }
$runCmd = "cd $BenchDir; setsid nohup $BenchDir/.venv/bin/python $BenchDir/run_mini.py " +
    "--output $outDir --api-base $ApiBase --subset $Subset --model-name '$ModelName' " +
    "--workers $Workers --max-tokens $MaxTokens --step-limit $StepLimit $think $ExtraRunArgs " +
    ">> $outDir.sup.log 2>&1 < /dev/null & sleep 6"
$aliveCmd = "pgrep -fc '[r]un_mini.py --output $outDir'"
$stateCmd = "$BenchDir/.venv/bin/python $BenchDir/arm_state.py --out $outDir"
$evalCmd = "cd $BenchDir; setsid nohup $BenchDir/.venv/bin/python -m swebench.harness.run_evaluation " +
    "--dataset_name $EvalDataset --predictions_path $outDir/preds.json --run_id $RunTag " +
    "--max_workers $EvalWorkers --timeout $EvalTimeout --report_dir $BenchDir " +
    ">> eval_$RunTag.log 2>&1 < /dev/null & sleep 6"
$evalAlive = "pgrep -fc '[s]webench.harness.*$RunTag'"
$reportCmd = "ls $BenchDir/*$RunTag.json >/dev/null 2>&1 && echo YES || echo NO"

function Num($cmd) {
    $o = (& wsl.exe -e bash -lc $cmd 2>$null | Out-String).Trim()
    if ($o -match '^(\d+)$') { return [int]$Matches[1] } else { return -1 }
}
function State {
    $o = (& wsl.exe -e bash -lc $stateCmd 2>$null | Out-String).Trim()
    if ($o -match '^(\d+)\s+(\d+)$') { return @([int]$Matches[1], [int]$Matches[2]) }
    return @(-1, -1)
}

Log "supervisor start tag=$RunTag target=$Target out=$outDir api=$ApiBase effort=$Effort"
$emptyRestarts = 0
for ($i = 0; $i -lt $MaxPolls; $i++) {
    $st = State
    if ($Target -gt 0 -and $st[0] -ge $Target -and $st[1] -lt $st[0]) {
        Log "inference complete: preds=$($st[0]) empty=$($st[1])"; break
    }
    if ($Target -gt 0 -and $st[0] -ge $Target -and $st[1] -eq $st[0]) {
        $emptyRestarts++
        if ($emptyRestarts -gt $MaxEmptyRestarts) {
            Log "ABORT: all patches empty after $emptyRestarts restarts; endpoint or protocol is broken"
            exit 2
        }
        Log "all-empty preds=$($st[0]) -> deleting, restarting ($emptyRestarts/$MaxEmptyRestarts)"
        & wsl.exe -e bash -lc "rm -f $outDir/preds.json" 2>$null | Out-Null
    }
    $a = Num $aliveCmd
    if ($a -le 0) {
        $dp = (& wsl.exe -e bash -lc 'docker ps -q >/dev/null 2>&1 && echo OK || echo BAD' 2>$null | Out-String).Trim()
        if ($dp -ne 'OK') { Log 'docker not ready -> waiting'; Start-Sleep -Seconds 45; continue }
        Log "run_mini dead (state=$($st -join '/')) -> relaunching"
        & wsl.exe -e bash -lc $runCmd 2>$null | Out-Null
    }
    Start-Sleep -Seconds $PollSeconds
}
$st = State
if ($Target -gt 0 -and $st[0] -lt $Target) { Log "giving up on inference (preds=$($st[0]))"; exit 1 }

for ($i = 0; $i -lt 120; $i++) {
    $done = (& wsl.exe -e bash -lc $reportCmd 2>$null | Out-String).Trim()
    if ($done -eq 'YES') { Log "eval report present -> DONE"; break }
    if ((Num $evalAlive) -le 0) {
        Log 'eval dead -> relaunching'
        & wsl.exe -e bash -lc $evalCmd 2>$null | Out-Null
    }
    Start-Sleep -Seconds $PollSeconds
}
Log 'supervisor exit'
