# Is the Windows GPU driver paging this process' device memory into system RAM?
# "Non Local Usage" / "Shared Usage" > 0 for the vLLM pid means the driver is
# evicting device allocations over PCIe, which matches the observed per-step
# host<->device traffic during decode.
$out = "D:\code\vllm-windows\_dev\out\_gpumem_counters.txt"
$lines = @()
$os = Get-CimInstance Win32_OperatingSystem
$lines += ("RAM total={0:N1} GiB  free={1:N1} GiB  commitLimit={2:N1} GiB  commitUsed={3:N1} GiB" -f `
    ($os.TotalVisibleMemorySize / 1MB), ($os.FreePhysicalMemory / 1MB), `
    ($os.TotalVirtualMemorySize / 1MB), (($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB))

$sets = (Get-Counter -ListSet "*gpu*" -ErrorAction SilentlyContinue).CounterSetName
$lines += "gpu counter sets: " + ($sets -join ', ')

foreach ($s in $sets) {
    $paths = (Get-Counter -ListSet $s).Paths | Where-Object { $_ -match "pid_14824|_Total|total" }
    foreach ($p in $paths) {
        try {
            $v = (Get-Counter -Counter $p -ErrorAction Stop).CounterSamples[0]
            $val = if ($v.CookedValue -gt 1e6) { "{0:N1} MiB" -f ($v.CookedValue / 1MB) } else { "{0:N3}" -f $v.CookedValue }
            $lines += ("  {0} = {1}" -f $p, $val)
        } catch { }
    }
}
$lines += "--- adapter ---"
foreach ($p in @("\GPU Adapter Memory(*)\Dedicated Usage", "\GPU Adapter Memory(*)\Shared Usage")) {
    try {
        $v = (Get-Counter -Counter $p -ErrorAction Stop)
        foreach ($s2 in $v.CounterSamples) {
            $lines += ("  {0} {1} = {2:N1} MiB" -f $p, $s2.InstanceName, ($s2.CookedValue / 1MB))
        }
    } catch { $lines += "  (missing) $p" }
}
$lines | Out-File -Encoding utf8 $out
