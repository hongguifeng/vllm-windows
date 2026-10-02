# Stop vLLM servers started by the _dev scripts, then report GPU memory.
#
#   & _stop_vllm.ps1                  # stop every vLLM python process (old behaviour)
#   & _stop_vllm.ps1 -Gpu 1           # stop only the service sitting on GPU1
#   & _stop_vllm.ps1 -Gpu 0           # ...or the one on GPU0
#   & _stop_vllm.ps1 -Port 9393       # select by --port instead of by card
#   & _stop_vllm.ps1 -Gpu 0 -DryRun   # show what it would stop, stop nothing
#   & _stop_vllm.ps1 -List            # roster only, stop nothing
#
# Why selection is not just "kill the process on the card": only EngineCore shows up
# in nvidia-smi's compute-app list. The API server holds the port and the ZMQ pair,
# and resource_tracker hangs off the same tree, so killing one half leaves the other
# half alive -- an orphaned EngineCore keeps the KV pool on the card, and the next
# start then comes up paging with no error logged (see start_server.ps1's -Force note).
# Card selection therefore seeds on the compute-app pid and pulls in the whole tree.
#
# A service held by WSL is invisible here: the guest's python.exe is not in
# Win32_Process and its allocation is attributed to vmmem, not to a pid nvidia-smi
# can report. Stop those from inside WSL (`bin/stop.sh`), not from here.
#
# Keep this a file: passing -Filter with quotes through bash inline -Command has
# repeatedly been mangled (WQL needs single quotes inside double quotes).
[CmdletBinding()]
param(
    # -1 (default) = every vLLM process, whatever card it is on.
    [int]$Gpu = -1,
    [string]$Port = '',
    [switch]$DryRun,
    [switch]$List
)

$ErrorActionPreference = 'Continue'

$procs = @(Get-CimInstance -ClassName Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'api_server|multiprocessing-fork|resource_tracker|EngineCore' })

if (-not $procs) {
    Write-Host 'no vllm python processes found'
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | ForEach-Object { Write-Host $_ }
    exit 0
}

# --- pid -> GPU index, from the per-card compute-app list ---------------------
$pidGpu = @{}
foreach ($line in @(nvidia-smi --query-gpu=index --format=csv,noheader)) {
    $g = ($line -split ',')[0].Trim()
    if ($g -eq '') { continue }
    foreach ($app in @(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader)) {
        $appPid = [int](($app -split ',')[0].Trim())
        if ($appPid -gt 0 -and -not $pidGpu.ContainsKey($appPid)) { $pidGpu[$appPid] = [int]$g }
    }
}

# --- process tree over the vLLM set only -------------------------------------
$byId = @{}
foreach ($p in $procs) { $byId[[int]$p.ProcessId] = $p }
$children = @{}
foreach ($p in $procs) {
    $pp = [int]$p.ParentProcessId
    if ($byId.ContainsKey($pp)) {
        if (-not $children.ContainsKey($pp)) { $children[$pp] = @() }
        $children[$pp] += [int]$p.ProcessId
    }
}

$selected = @{}
function Add-Down([int]$from) {
    if (-not $selected.ContainsKey($from)) { $selected[$from] = $true }
    foreach ($c in $children[$from]) { Add-Down ([int]$c) }
}
function Add-Up([int]$from) {
    if (-not $selected.ContainsKey($from)) { $selected[$from] = $true }
    $p = $byId[$from]
    if ($p) {
        $pp = [int]$p.ParentProcessId
        if ($byId.ContainsKey($pp)) { Add-Up ([int]$pp) }
    }
}

$seeds = @()
if ($Port) {
    foreach ($p in $procs) {
        if ($p.CommandLine -match ('--port[\s=]+' + [regex]::Escape($Port))) { $seeds += [int]$p.ProcessId }
    }
    if (-not $seeds) { Write-Host "WARNING: nothing listening matches --port $Port" -ForegroundColor Yellow }
} elseif ($Gpu -ge 0) {
    foreach ($p in $procs) {
        if ($pidGpu.ContainsKey([int]$p.ProcessId) -and $pidGpu[[int]$p.ProcessId] -eq $Gpu) {
            $seeds += [int]$p.ProcessId
        }
    }
    if (-not $seeds) {
        Write-Host "WARNING: no Windows vLLM process is holding GPU$Gpu." -ForegroundColor Yellow
        $held = nvidia-smi --id=$Gpu --query-gpu=memory.used --format=csv,noheader,nounits
        $apps = @(nvidia-smi --id=$Gpu --query-compute-apps=pid --format=csv,noheader)
        if (([double]$held -gt 2048) -and (-not $apps)) {
            Write-Host "         GPU$Gpu still reports $held MiB used but no Windows-side compute app:" -ForegroundColor Yellow
            Write-Host '         that is a WSL allocation (charged to vmmem). Stop it inside WSL:' -ForegroundColor Yellow
            Write-Host '           wsl.exe -e bash -lc "~/code/qwen3.8-flash-next-cmp170hx/bin/stop.sh --check"' -ForegroundColor Yellow
        } elseif (([double]$held -gt 2048)) {
            Write-Host "         GPU$Gpu used $held MiB by pids: $($apps -join ', ') -- not python.exe here." -ForegroundColor Yellow
        }
    }
} else {
    $seeds = @($procs | ForEach-Object { [int]$_.ProcessId })
}

foreach ($s in $seeds) { Add-Up ([int]$s); Add-Down ([int]$s) }

function Format-Row($p) {
    $port = [regex]::Match($p.CommandLine, '--port[\s=]+(\d+)').Groups[1].Value
    $gpu = if ($pidGpu.ContainsKey([int]$p.ProcessId)) { "gpu$($pidGpu[[int]$p.ProcessId])" } else { 'gpu?' }
    $tail = if ($port) { "  port $port" } else { '' }
    $head = $p.CommandLine
    if (-not $head) { $head = '(command line not readable)' }
    if ($head.Length -gt 80) { $head = $head.Substring(0, 80) }
    return "  {0}  {1}  {2}{3}" -f $p.ProcessId, $gpu, $head, $tail
}

Write-Host '--- vllm processes found:'
foreach ($p in $procs) { Write-Host (Format-Row $p) }

$kill = @($selected.Keys | Sort-Object)
$keep = @($procs | Where-Object { -not $selected.ContainsKey([int]$_.ProcessId) })

if ($List) { exit 0 }
if (-not $seeds) { Write-Host 'nothing selected; nothing stopped.' -ForegroundColor Yellow; exit 0 }

Write-Host ("--- stopping {0} process(es) on {1}:" -f $kill.Count,
    $(if ($Port) { "port $Port" } elseif ($Gpu -ge 0) { "GPU$Gpu" } else { 'every card' })) -ForegroundColor Yellow
foreach ($id in $kill) { Write-Host (Format-Row $byId[$id]) }
if ($keep) {
    Write-Host '--- left alone:' -ForegroundColor DarkGray
    foreach ($p in $keep) { Write-Host ('  ' + (Format-Row $p).TrimStart()) }
}

if ($DryRun) {
    Write-Host '--- dry run: nothing was stopped.' -ForegroundColor Yellow
    exit 0
}

foreach ($id in $kill) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }

Start-Sleep -Seconds 12
Write-Host '--- gpu memory after stop:'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | ForEach-Object { Write-Host $_ }
Write-Host '--- compute apps:'
$appsWithMem = nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
if ($appsWithMem) { $appsWithMem | ForEach-Object { Write-Host $_ } } else { Write-Host '(none)' }

# If the requested card still holds memory that no Windows pid owns, say so rather
# than letting the next boot look like it failed for no reason.
if ($Gpu -ge 0) {
    $held = nvidia-smi --id=$Gpu --query-gpu=memory.used --format=csv,noheader,nounits
    $apps = @(nvidia-smi --id=$Gpu --query-compute-apps=pid --format=csv,noheader)
    if (([double]$held -gt 2048) -and (-not $apps)) {
        Write-Host "GPU$Gpu still holds $held MiB owned by nothing Windows can see (WSL/vmmem)." -ForegroundColor Yellow
    }
}

$left = @(Get-CimInstance -ClassName Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'api_server|multiprocessing-fork|resource_tracker|EngineCore' })
if ($left) {
    Write-Host '--- still running:' -ForegroundColor DarkGray
    foreach ($p in $left) { Write-Host ('  ' + (Format-Row $p).TrimStart()) }
}
