# Sample the engine's GPU memory buckets while a request decodes.
#   Non Local / Shared oscillating  -> the driver is paging (evict + reload)
#   flat while PCIe streams         -> the traffic is an app-controlled copy
$repo = "D:\code\vllm-windows"
$py = "C:\Users\hong\.workbuddy\binaries\python\versions\3.13.12\python.exe"
$out = Join-Path $repo "_dev\out\_paging_watch.txt"
$cli = Join-Path $repo "_dev\out\_paging_client.txt"
# The engine pid changes every boot, so discover its GPU instance instead of
# hardcoding it. Pick the largest Dedicated Usage: once paging is fixed the
# Non Local bucket legitimately reads 0, so it cannot be used to find the pid.
$ded = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples |
    Where-Object { $_.CookedValue -gt 1GB } | Sort-Object CookedValue -Descending
if (-not $ded) { throw "no GPU instance with >1GiB dedicated usage -- is the engine up?" }
$inst = $ded[0].InstanceName
Write-Host "engine GPU instance: $inst  (dedicated $([math]::Round($ded[0].CookedValue/1MB,1)) MiB)"
$pLocal = "\GPU Process Memory($inst)\Local Usage"
$pNon = "\GPU Process Memory($inst)\Non Local Usage"
$pShared = "\GPU Process Memory($inst)\Shared Usage"

$proc = Start-Process -FilePath $py -PassThru -NoNewWindow -RedirectStandardOutput $cli `
    -ArgumentList @("$repo\_dev\bench\_itl_client.py", "--port", "8000", "--tokens", "640", "--warm", "0", "--label", "paging") `
    -WorkingDirectory "$repo\_dev\out"

$lines = @("t_s   nonLocal   shared    local    rxMB/s  txMB/s")
$sw = [Diagnostics.Stopwatch]::StartNew()
while (-not $proc.HasExited -and $sw.Elapsed.TotalSeconds -lt 120) {
    $non = (Get-Counter -Counter $pNon).CounterSamples[0].CookedValue / 1MB
    $shr = (Get-Counter -Counter $pShared).CounterSamples[0].CookedValue / 1MB
    $loc = (Get-Counter -Counter $pLocal).CounterSamples[0].CookedValue / 1MB
    $t = (nvidia-smi dmon -s t -c 2 -o T | Select-Object -Last 1) -split '\s+' | Where-Object { $_ }
    $lines += ("{0,4:N1} {1,9:N1} {2,8:N1} {3,8:N1}  {4,6}  {5,6}" -f $sw.Elapsed.TotalSeconds, $non, $shr, $loc, $t[2], $t[3])
    $proc.Refresh()
}
$lines += (Get-Content $cli -Raw)
$lines | Out-File -Encoding utf8 $out
