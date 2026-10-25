# Alternating restart A/B for the sm_80 thin GEMM on ONE card.
#
# Two servers on two cards is not a fair A/B: the cards differ (slot, lane
# width) by more than the effect being measured. This restarts the same card
# with the switch off and on, in alternating order, and benches every arm with
# the same client.
#
#   & _dev\bin\_ab_thin_gemm_restart.ps1 -Gpu 2 -Port 9402
#
# Logs land in _dev\out\logs, results in _dev\out\ab_restart_<arm>.json.
[CmdletBinding()]
param(
    [int]$Gpu = 2,
    [int]$Port = 9402,
    [string[]]$Arms = @('off', 'on', 'off', 'on'),
    [int]$Reps = 4,
    [int]$HealthTimeoutSec = 900
)

$ErrorActionPreference = 'Continue'
$repo = 'D:\code\vllm-windows'
$serve = "$repo\_dev\bin\_flashnext_struct_serve.ps1"
$py = "$repo\.venv\Scripts\python.exe"

function Stop-Arm {
    $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -eq $Port })
    foreach ($l in $listeners) {
        $child = $l.OwningProcess
        $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$child").ParentProcessId
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$child" | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $child -Force -ErrorAction SilentlyContinue
        if ($parent) { Stop-Process -Id $parent -Force -ErrorAction SilentlyContinue }
    }
    Start-Sleep -Seconds 6
}

function Start-Arm([string]$Arm) {
    $a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $serve,
        '-Model', 'D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP',
        '-ServedName', 'Qwen3.8-Flash-Next',
        '-Venv', "$repo\.flashnext-root", '-Gpu', "$Gpu", '-Port', "$Port",
        '-MaxLen', '262144', '-MaxSeqs', '4', '-BatchedTokens', '2048',
        '-BlockSize', '1680', '-PrefixMatchUnit', '16', '-MemUtil', '0.96',
        '-FullWeights', '-WithVision', '-PleSsd', '-PleDepth', '512',
        '-PleCacheMb', '8192', '-PleWorkers', '16', '-PlePrefetchTokens', '16384',
        '-Mtp', '-MtpTokens', '3', '-Graphs')
    if ($Arm -eq 'off') { $a += '-NoThinGemm' }  # thin GEMM is on by default
    $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $a -WindowStyle Minimized -PassThru
    Write-Host ("[{0}] launcher {1}" -f $Arm, $p.Id)
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
            if ($r.StatusCode -eq 200) {
                Write-Host ("[{0}] ready after {1:N0}s" -f $Arm,
                    ((Get-Date) - (Get-Date).AddSeconds(-1)).TotalSeconds)
                return $true
            }
        } catch { }
    }
    Write-Host ("[{0}] NOT ready within {1}s" -f $Arm, $HealthTimeoutSec)
    return $false
}

foreach ($arm in $Arms) {
    Stop-Arm
    if (-not (Start-Arm $arm)) { continue }
    foreach ($conc in @(4, 1)) {
        $out = "$repo\_dev\out\ab_restart_${arm}_c$conc.json"
        Write-Host ("[{0}] bench conc={1} reps={2}" -f $arm, $conc, $Reps)
        & $py "$repo\_dev\probe\_ab_thin_gemm.py" --port $Port --arm "$arm-c$conc" `
            --reps $Reps --conc $conc --out $out 2>&1 | Select-Object -Last 6
    }
}
Stop-Arm
Write-Host 'done'