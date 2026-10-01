param(
    [switch]$Profiler,
    [int]$ProfilerActiveIterations = 30,
    [int]$Probe = 7
)
$argv = @()
if ($Profiler) {
    $argv += @(
        '--profiler-config.profiler', 'torch',
        '--profiler-config.active_iterations', $ProfilerActiveIterations,
        '--probe', $Probe
    )
}
"count={0}" -f $argv.Count | Write-Host
foreach ($a in $argv) { "token>[$a]<len={0}" -f $a.Length | Write-Host }
"cmdline> " + ($argv -join ' ') | Write-Host
