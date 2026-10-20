# Copy the fork's patched engine files into the installed snapshot that the
# production service actually imports.
#
#   & _dev\bin\_sync_snapshot_overlay.ps1                 # apply the default set
#   & _dev\bin\_sync_snapshot_overlay.ps1 -DryRun         # show what would change
#   & _dev\bin\_sync_snapshot_overlay.ps1 -Groups core,dflash
#   & _dev\bin\_sync_snapshot_overlay.ps1 -Restore        # undo the last apply
#
# The service runs from .flashnext-root\.venv\Lib\site-packages\vllm, a copy
# installed on 2026-10-01, so editing the repo tree alone changes nothing there
# (the first KV-tier restart failed on exactly that). This overlay is the
# equivalent of the recipe's bind-mounted patches: an explicit list of files,
# each backed up before it is replaced.
[CmdletBinding()]
param(
    [ValidateSet('core', 'dflash', 'tp2', 'all')]
    [string[]]$Groups = @('core'),
    [string]$SnapshotVenv = 'D:\code\vllm-windows\.flashnext-root',
    [switch]$DryRun,
    [switch]$Restore,
    # Keep the previous backups instead of reusing them, i.e. always treat the
    # current snapshot contents as the baseline.
    [switch]$ReBaseline
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\code\vllm-windows'
$target = "$SnapshotVenv\.venv\Lib\site-packages"
$backupRoot = "$repo\_dev\out\overlay_backup"

# Files are grouped by the change that motivates them, so an arm that only needs
# one change does not drag the others into production.
$files = @{
    core = @(
        # Windows shared-memory region for the host KV tier (P5).
        'vllm\v1\kv_offload\cpu\shared_offload_region.py'
    )
    dflash = @(
        # Eagle3 capture for a DFlash 2 drafter and the guard around it.
        'vllm\models\qwen4_exp\nvidia\model.py'
        'vllm\model_executor\models\config.py'
        'vllm\v1\worker\gpu\model_runner.py'
    )
    tp2 = @(
        # TP-aware PLE row ownership; no-ops at TP=1.
        'vllm\models\qwen4_exp\nvidia\ple_ssd.py'
        'vllm\models\qwen4_exp\nvidia\ngram_embedding.py'
    )
}

if ($Groups -contains 'all') { $Groups = @('core', 'dflash', 'tp2') }
$selected = @()
foreach ($group in $Groups) { $selected += $files[$group] }
$selected = @($selected | Sort-Object -Unique)

if ($Restore) {
    $last = Get-ChildItem $backupRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $last) { throw "no overlay backup under $backupRoot" }
    Write-Host "restoring from $($last.FullName)"
    foreach ($relative in $selected) {
        $saved = Join-Path $last.FullName $relative
        if (-not (Test-Path -LiteralPath $saved)) {
            Write-Host "  no backup for $relative (left alone)"
            continue
        }
        $destination = Join-Path $target $relative
        if ($DryRun) {
            Write-Host "  would restore $relative"
            continue
        }
        Copy-Item -LiteralPath $saved -Destination $destination -Force
        Write-Host "  restored $relative"
    }
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupDir = Join-Path $backupRoot $stamp
$changed = 0
$unchanged = 0
$missing = 0

foreach ($relative in $selected) {
    $source = Join-Path $repo $relative
    $destination = Join-Path $target $relative
    if (-not (Test-Path -LiteralPath $source)) {
        Write-Host "  MISSING in tree: $relative" -ForegroundColor Red
        $missing++
        continue
    }
    if (-not (Test-Path -LiteralPath $destination)) {
        Write-Host "  MISSING in snapshot: $relative" -ForegroundColor Red
        $missing++
        continue
    }
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    $destHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    if ($sourceHash -eq $destHash) {
        Write-Host "  same    $relative"
        $unchanged++
        continue
    }
    Write-Host "  update  $relative" -ForegroundColor Yellow
    $changed++
    if ($DryRun) { continue }
    if ($ReBaseline -or -not (Test-Path -LiteralPath (Join-Path $backupDir $relative))) {
        $saved = Join-Path $backupDir $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $saved) | Out-Null
        Copy-Item -LiteralPath $destination -Destination $saved -Force
    }
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

Write-Host ("groups [{0}]: {1} updated, {2} already in sync, {3} missing" -f
    ($Groups -join ','), $changed, $unchanged, $missing)
if ($DryRun) { Write-Host 'dry run: nothing was written' }
elseif ($changed -gt 0) {
    Write-Host "backup: $backupDir"
    Write-Host 'restart the service for the change to take effect: .\stop_vllm.ps1 -Port 9393; .\start_qwen38_flash_next.ps1'
}
if ($missing -gt 0) { exit 1 }
