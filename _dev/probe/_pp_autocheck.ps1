<#
Half-hourly health check for the W4A16 checkpoint download.
Checks the NVMe first (disk 3), then WSL-side progress, with a hard timeout so a
dead disk cannot hang the check.
#>
$ErrorActionPreference = 'Continue'
$stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
"=== $stamp ==="
$d3 = Get-Disk | Where-Object { $_.Number -eq 3 -and $_.FriendlyName -match 'KIOXIA' }
if (-not $d3) {
    "DISK: KIOXIA (disk 3) NOT PRESENT - download cannot continue"
} else {
    "DISK: $($d3.FriendlyName) $($d3.OperationalStatus)"
    $out = & wsl bash /mnt/d/code/vllm-windows/_dev/probe/_pp_autocheck.sh 2>&1
    $out | ForEach-Object { $_ }
}
"gpu: " + ((nvidia-smi --query-gpu=index,memory.used --format=csv,noheader) -join '; ')
