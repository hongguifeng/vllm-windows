[CmdletBinding()]
param(
    [string]$Repo = 'D:\code\vllm-windows',
    [string]$BufferedDll = '',
    [string]$DirectDll = '',
    [switch]$SkipWslCheck
)
$ErrorActionPreference = 'Stop'
$out = Join-Path $Repo '_dev\out'
if (-not $BufferedDll) { $BufferedDll = Join-Path $out 'ple_ssd_io\ple_ssd_io_win.dll' }
if (-not $DirectDll) { $DirectDll = Join-Path $out 'ple_ssd_io_direct_probe\ple_ssd_io_win.dll' }
foreach ($dll in @($BufferedDll, $DirectDll)) {
    if (-not (Test-Path -LiteralPath $dll)) { throw "Build the experimental DLL first: $dll" }
}
$python = Join-Path $repo '.venv-flashnext\Scripts\python.exe'
$probe = Join-Path $repo '_dev\probe\_ple_ssd_io_win_test.py'
function Snapshot {
    $disk = Get-CimInstance Win32_PerfRawData_PerfDisk_LogicalDisk -Filter "Name='D:'"
    $memory = Get-CimInstance Win32_OperatingSystem
    [pscustomobject]@{
        wall = (Get-Date).ToString('o')
        disk_read_bytes_raw = [uint64]$disk.DiskReadBytesPersec
        disk_read_ops_raw = [uint64]$disk.DiskReadsPersec
        disk_timestamp = [uint64]$disk.Timestamp_PerfTime
        disk_frequency = [uint64]$disk.Frequency_PerfTime
        available_gib = [math]::Round($memory.FreePhysicalMemory / 1MB, 3)
    }
}
$cases = @(
    @{ label = 'buffered_fresh_40101'; dll = $BufferedDll; seed = 40101; repeat = 3 },
    @{ label = 'buffered_same_40101'; dll = $BufferedDll; seed = 40101; repeat = 3 },
    @{ label = 'buffered_fresh_40102'; dll = $BufferedDll; seed = 40102; repeat = 3 },
    @{ label = 'direct_same_40101'; dll = $DirectDll; seed = 40101; repeat = 2 }
)
$observations = @()
$before = Snapshot
Start-Sleep -Seconds 2
$after = Snapshot
$observations += [pscustomobject]@{ label = 'idle_baseline'; before = $before; after = $after }
foreach ($case in $cases) {
    $windowsMetrics = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:9393/metrics'
    if ($windowsMetrics.Content -match 'vllm:num_requests_running\{[^\n]+\}\s+([0-9.]+)' -and [double]$Matches[1] -ne 0) {
        throw 'Windows service is busy; refusing shared-disk probe.'
    }
    if (-not $SkipWslCheck) {
        $wslMetrics = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 'http://127.0.0.1:8000/metrics'
        if ($wslMetrics.Content -match 'vllm:num_requests_running\{[^\n]+\}\s+([0-9.]+)' -and [double]$Matches[1] -ne 0) {
            throw 'WSL service is busy; refusing shared-disk probe.'
        }
    }
    $dll = $case.dll
    $before = Snapshot
    & $python $probe --dll $dll --rows 32768 --depth 256 --seed $case.seed --repeat $case.repeat --check 128 2>&1 |
        Tee-Object -FilePath (Join-Path $out ($case.label + '.log'))
    if ($LASTEXITCODE -ne 0) { throw "Probe failed: $($case.label)" }
    $after = Snapshot
    $observations += [pscustomobject]@{ label = $case.label; before = $before; after = $after }
    $observations | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $out 'buffered_cache_disk_snapshots.json')
}
$observations | ForEach-Object {
    $seconds = ($_.after.disk_timestamp - $_.before.disk_timestamp) / $_.before.disk_frequency
    [pscustomobject]@{
        label = $_.label
        seconds = [math]::Round($seconds, 3)
        volume_read_mib = [math]::Round(($_.after.disk_read_bytes_raw - $_.before.disk_read_bytes_raw) / 1MB, 3)
        volume_read_ops = $_.after.disk_read_ops_raw - $_.before.disk_read_ops_raw
        avail_gib = $_.after.available_gib
    }
} | Format-Table -AutoSize
