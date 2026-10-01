# Stop any vLLM server started by the _dev scripts, then report GPU memory.
# Keep this a file: passing -Filter with quotes through bash inline -Command
# has repeatedly been mangled (WQL needs single quotes inside double quotes).
$ErrorActionPreference = 'Continue'

$procs = Get-CimInstance -ClassName Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'api_server|multiprocessing-fork|resource_tracker|EngineCore' }

if (-not $procs) {
    Write-Host 'no vllm python processes found'
} else {
    foreach ($p in $procs) {
        $head = $p.CommandLine
        if ($head.Length -gt 80) { $head = $head.Substring(0, 80) }
        Write-Host ("stopping {0}  {1}" -f $p.ProcessId, $head)
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Start-Sleep -Seconds 12
Write-Host '--- gpu memory after stop:'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | ForEach-Object { Write-Host $_ }
Write-Host '--- compute apps:'
$appsWithMem = nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
if ($appsWithMem) { $appsWithMem | ForEach-Object { Write-Host $_ } } else { Write-Host '(none)' }
