# Per-process GPU memory: Local (device) vs Non Local (system RAM) usage.
# Non Local > 0 for the vLLM pid == the driver is backing device allocations
# with system RAM, i.e. it pages them back over PCIe during compute.
$out = "D:\code\vllm-windows\_dev\out\_procgpumem.txt"
$lines = @()
$paths = (Get-Counter -ListSet "GPU Process Memory").Paths | Where-Object { $_ -match "Usage|Committed" }
$lines += "paths: " + ($paths -join ' | ')
foreach ($p in $paths) {
    try {
        $s = (Get-Counter -Counter $p -ErrorAction Stop).CounterSamples
    } catch { continue }
    foreach ($x in $s) {
        if ($x.InstanceName -match "pid_14824|pid_9800" -and $x.CookedValue -gt 0) {
            $lines += ("  {0,-52} {1,-42} {2:N1} MiB" -f (($p -split '\\')[-1]), $x.InstanceName, ($x.CookedValue / 1MB))
        }
    }
}
$lines += "--- all instances summary ---"
foreach ($p in $paths) {
    try { $s = (Get-Counter -Counter $p -ErrorAction Stop).CounterSamples } catch { continue }
    $nz = $s | Where-Object { $_.CookedValue -gt 1048576 }
    $lines += ("  {0}: {1} instances >1MiB" -f (($p -split '\\')[-1]), $nz.Count)
    foreach ($x in $nz | Select-Object -First 6) {
        $lines += ("      {0} = {1:N1} MiB" -f $x.InstanceName, ($x.CookedValue / 1MB))
    }
}
$lines | Out-File -Encoding utf8 $out
