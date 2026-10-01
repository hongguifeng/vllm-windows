# Correlate wall-clock GPU/CPU state with a single streaming request.
# Usage: pwsh -File _dev/probe/_watch_req.ps1 [-Tokens 512]
param(
    [int]$Tokens = 512,
    [int]$EnginePid = 14824,
    [int]$ApiPid = 9800,
    [string]$Label = "watch"
)
$ErrorActionPreference = "Continue"
$repo = "D:\code\vllm-windows"
$py = "C:\Users\hong\.workbuddy\binaries\python\versions\3.13.12\python.exe"
$out = Join-Path $repo "_dev\out\_watch_req.txt"
$cliOut = Join-Path $repo "_dev\out\_watch_req_client.txt"

$t0 = Get-Date
$cli = Start-Process -FilePath $py -PassThru -NoNewWindow -RedirectStandardOutput $cliOut `
    -ArgumentList @("$repo\_dev\bench\_itl_client.py", "--port", "8000", "--tokens", "$Tokens", "--warm", "0", "--label", $Label) `
    -WorkingDirectory "$repo\_dev\out"

$prevE = (Get-Process -Id $EnginePid -ErrorAction SilentlyContinue).TotalProcessorTime
$prevA = (Get-Process -Id $ApiPid -ErrorAction SilentlyContinue).TotalProcessorTime
$lines = @()
$lines += ("t_s   gpu%   smMHz    pwrW  temp   eng%   api%  allcpu%  top_proc")
$i = 0
while (-not $cli.HasExited -and $i -lt 400) {
    Start-Sleep -Milliseconds 500
    $g = (nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits) -split ','
    $pe = Get-Process -Id $EnginePid -ErrorAction SilentlyContinue
    $pa = Get-Process -Id $ApiPid -ErrorAction SilentlyContinue
    $eng = if ($pe) { [Math]::Round(($pe.TotalProcessorTime - $prevE).TotalMilliseconds / 500 / 10, 1) } else { -1 }
    $api = if ($pa) { [Math]::Round(($pa.TotalProcessorTime - $prevA).TotalMilliseconds / 500 / 10, 1) } else { -1 }
    if ($pe) { $prevE = $pe.TotalProcessorTime }
    if ($pa) { $prevA = $pa.TotalProcessorTime }
    $top = (Get-Process | Sort-Object CPU -Descending | Select-Object -First 1 -ExpandProperty ProcessName)
    $dt = ((Get-Date) - $t0).TotalSeconds
    $lines += ("{0,5:N1} {1,6} {2,7} {3,7} {4,5} {5,6} {6,6} {7,8}  {8}" -f `
        $dt, $g[0].Trim(), $g[1].Trim(), $g[2].Trim(), $g[3].Trim(), $eng, $api, 0, $top)
    $i++
    $cli.Refresh()
}
$lines += "client exited after " + ([Math]::Round(((Get-Date) - $t0).TotalSeconds, 1)) + " s"
$lines | Out-File -Encoding utf8 $out
if (Test-Path $cliOut) { $lines += (Get-Content $cliOut -Raw) }
$lines | Out-File -Encoding utf8 $out
