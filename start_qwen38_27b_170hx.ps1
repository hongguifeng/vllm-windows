# Dedicated launcher for the 2x CMP 170HX rig (GA100, sm_80, 64 GB each).
#
#   .\start_server_170hx.ps1              # GPU1, the config every 170HX measurement used
#   .\start_server_170hx.ps1 -TP2         # EXPERIMENTAL tensor parallel over both cards
#   .\start_server_170hx.ps1 -VisionOffload   # restore the 3090-era tower-offload behaviour
#   .\start_server_170hx.ps1 -DryRun      # print the resolved argv, start nothing
#
# It is a thin wrapper around start_server.ps1 -- all the pre-flight, logging and
# guard logic lives there. This file only encodes the 170HX-specific defaults,
# each one tied to a measurement (see _dev/out/_ab_results/170hx_vs_3090.md):
#
#   * -MaxGraphCapture 64  THE 170HX fix (2026-09-26 "D" run). start_server's
#     derived default (8, from -MaxSeqs 8) left every cohort of 2+ requests
#     eager: the capture ceiling is counted in TOKENS and spec verify costs 8
#     tokens per request, so c1 had a graph and c2/c4/c8 did not (+45 ms/step,
#     itl50 66-73 ms). With 64: c1 192.3 / c2 312.5 / c4 552.5 / c8 869.6 t/s.
#   * No vision-tower offload by default. --cpu-offload-gb 1 +
#     --cpu-offload-params visual + VLLM_OFFLOAD_RELEASE_AFTER_FORWARD existed
#     to free ~0.9 GiB on a 24 GB card; on 64 GB the tower just lives on the
#     card and every image skips the offload-copy round trip.
#     The 170HX timings above were taken WITH the offload, so this switch is
#     the one unmeasured default here -- text-only is unaffected either way.
#   * CUDA_VISIBLE_DEVICES=1 by default. Every number in the A/B report is a
#     single-GPU0 number; TP is opt-in via -TP2 and has never run on this rig
#     (it needs the SystemPanic/nccl-windows build via VLLM_NCCL_SO_PATH).
#
# Deliberately NOT changed from start_server.ps1, and why:
#   * -MaxLen is auto now (0 = min(262,144 model ceiling, 95% of pool)); the
#     4096-granular result leaves the rest of the pool to prefix caching.
#     Override explicitly for A/B.
#   * -MaxSeqs 8. Your load is <=C3; raising it only shrinks the per-request
#     share of the KV pool.
#   * DFlash2 stays the default. On the FIXED config (cg64) it wins at every
#     concurrency: c1 2.98x over no-spec, c2 2.5x, c4 2.2x, c8 1.85x. (The
#     earlier "spec is net-negative at c>=2" finding was the eager-mode bug.)

[CmdletBinding()]
param(
    [string]$Model  = 'D:\models\Qwen3.8-27B-W4A16-AutoRound-fast',
    [string]$Drafter = 'D:\models\Qwen3.8-27B-DFlash2-W4A16',
    [string]$ServedName = 'qwen3.8-27b',
    [ValidateSet('dflash', 'mtp', 'none')]
    [string]$Spec = 'dflash',
    [int]$Gpu = 1,
    # 0 (default) = AUTO: min(model max_position_embeddings from config.json
    # -- 262,144 on both the target and the drafter -- and 95% of what the
    # auto-filled KV pool can hold per the fit formula). Falls back to the
    # proven 71,680 when the pool is too small or the config is unreadable.
    # NOTE: any -MaxLen change invalidates the torch.compile cache, so the
    # first boot after this recomputes (~114 s extra).
    [int]$MaxLen = 0,
    # 0 (default) = AUTO-FILL: measure GPU0's free VRAM at launch and give KV
    # everything past the non-KV budget plus a 3 GiB safety margin. On the
    # 3090 the pin had to sit at 5.8e9 because a bigger one tipped WDDM into
    # paging the weights; the 64 GB card has no such cliff, and a filled pool
    # is genuinely useful (3 full-length concurrent requests need ~16.7 GiB of
    # KV where the old pin held 5.4). A full-length seq costs ~78 KB/token on
    # this model (bf16, 16 full-attn + 48 GDN layers).
    # Pass a byte count explicitly for A/B work.
    [long]$KvBytes = 0,
    [int]$Port = 8000,
    [int]$MaxSeqs = 8,
    # Opt in to tensor parallel across both 170HX cards. Experimental: no NCCL
    # has ever been exercised on this Windows/masqueraded-GA100 rig.
    [switch]$TP2,
    # Opt OUT of the 170HX default (tower resident on the card) and back to the
    # 3090-era --cpu-offload-gb behaviour. Only useful for A/B work.
    [switch]$VisionOffload,
    [switch]$TextOnly,
    [switch]$NoAsync,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = 'Continue'
$REPO = 'D:\code\vllm-windows'

# Which card(s) the server may see. Single card = every measurement ever taken
# on this rig; dual visibility is only for the TP2 experiment.
if ($TP2) {
    $env:CUDA_VISIBLE_DEVICES = '0,1'
} else {
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
}

# ------------------------------------------------------- KV pin auto-fill --
# Card-side non-KV budget, assembled from the 3090-era measured anchors
# (start_server.ps1 pre-flight comments) adjusted for this config:
#   weights+drafter   16.3 GiB  (15,060 MiB main + 1,221 MiB drafter)
#   CUDA graphs        1.33 GiB (capture ceiling 64; 1.08 at the derived 8)
#   allocator misc     0.2  GiB (workspace rounding that survived measurement)
#   vision tower       1.3  GiB (0.9 resident + ~0.4 encoder workspace,
#                               both first-image triggers; 0 when -TextOnly
#                               or -VisionOffload, 0.45+0.4 sharded under TP2)
$weightsGiB = if ($TP2) { 8.4 } else { 16.3 }
$visionGiB  = if ($TextOnly) { 0.0 }
              elseif ($VisionOffload) { 0.0 }
              elseif ($TP2) { 0.85 }
              else { 1.3 }
$nonKvGiB   = $weightsGiB + 1.33 + 0.2 + $visionGiB
# Safety margin. The 3090 paging cliff fired with ~0.2 GiB of headroom; 3 GiB
# is deliberately generous because this configuration has never been measured
# and WDDM behaviour near the top of a 64 GB card is unknown territory.
$marginGiB = 3.0
# Ghost-VRAM correction, REVISED 2026-09-26 16:45 after a chunked-bandwidth
# probe: 63x1 GiB chunks all ran at HBM speed (~1500 GB/s), only the 64th fell
# to 2.2 GB/s (PCIe speed = silently demoted to system RAM). So the card IS
# ~63 GiB of real HBM (mem_get_info total 62.59 GiB agrees); the old "4.5 GiB
# decimal-GB ghost" theory is DEAD. What remains off the top is ~1.4 GiB of
# driver reserve. The 15:25 refusal of a 40.99 GiB buffer is now attributed to
# HOST COMMIT pressure (bench + WSL + vLLM were near the then-139.9 GB limit),
# not VRAM capacity. Keep 3.0 here (not the bare 1.4): MCDM overcommits
# silently, so an oversized pin does not crash -- it demotes and slows down,
# and the only symptom would be itl/rx regression. Budget conservatively.
$deadGiB = 3.0

if ($KvBytes -le 0) {
    try {
        $csv = & nvidia-smi --id="$Gpu" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
        $f = ($csv | Select-Object -First 1) -split ','
        $freeGiB = ([double]$f[0] - [double]$f[1]) / 1024   # MiB -> GiB
        $pinGiB  = [math]::Floor($freeGiB - $deadGiB - $nonKvGiB - $marginGiB)
        # WDDM mirrors every VRAM allocation ~1:1 into system COMMIT. A 37 GiB
        # pin means ~37 GB of commit on top of whatever else is live (another
        # GPU workload, WSL, ...). The 15:34 BSoD happened when probe + bench
        # + WSL together ate ~130 of the 139.9 GB commit limit -- 0xEF,
        # CRITICAL_PROCESS_DIED. Shrink the pin rather than repeat that.
        $os = Get-CimInstance Win32_OperatingSystem
        $commitFreeGiB = $os.FreeVirtualMemory / 1MB
        $pinGiB = [math]::Min($pinGiB, [math]::Floor($commitFreeGiB - 8))
        if ($pinGiB -lt 5.4) {
            Write-Host "auto-fill: system commit headroom too low ($([math]::Round($commitFreeGiB,1)) GiB free of a $([math]::Round($os.TotalVirtualMemorySize/1MB,1)) GiB limit); falling back to the proven 5.8e9 pin" -ForegroundColor Yellow
            $KvBytes = 5800000000
        } else {
            $KvBytes = [long]($pinGiB * 1GB)
        }
    } catch {
        Write-Host "auto-fill: nvidia-smi unavailable, falling back to the proven 5.8e9 pin" -ForegroundColor Yellow
        $KvBytes = 5800000000
    }
}

$splat = @{
    Model           = $Model
    Drafter         = $Drafter
    ServedName      = $ServedName
    Spec            = $Spec
    MaxLen          = $MaxLen
    KvBytes         = $KvBytes
    Port            = $Port
    MaxSeqs         = $MaxSeqs
    MaxGraphCapture = 64
    TP              = $(if ($TP2) { 2 } else { 1 })
}
if ($TextOnly)      { $splat.TextOnly = $true }
if (-not $TextOnly -and -not $VisionOffload) { $splat.NoVisionOffload = $true }
if ($NoAsync)       { $splat.NoAsync = $true }
if ($Force)         { $splat.Force = $true }
if ($DryRun)        { $splat.DryRun = $true }

# ----------------------------------------------------- context auto-max --
# The fit formula (start_server.ps1, calibrated on the 3090 anchors) is
#   tokens = (KvBytes - 1.091e9) / 81,859 + 0.2122 * MaxLen
# vLLM refuses to boot when max_model_len exceeds the pool, so solve for the
# largest self-consistent MaxLen: MaxLen*(1-0.2122) <= (Kv-1.091e9)/81,859.
# The 5% haircut covers the formula's extrapolation beyond its 71,680-token
# calibration range -- if the estimate is off, vLLM dies with an explicit
# "max seq len is larger than KV cache" instead of paging, so the failure is
# loud and -MaxLen exists as the manual override.
if ($MaxLen -le 0) {
    $modelMaxPos = 0
    try {
        # Multimodal configs nest it (Qwen3_5: text_config.text_config-free,
        # the value lives under text_config); check both spots.
        $cfg = Get-Content "$Model\config.json" -Raw | ConvertFrom-Json
        $modelMaxPos = [int]$cfg.text_config.max_position_embeddings
        if ($modelMaxPos -le 0) { $modelMaxPos = [int]$cfg.max_position_embeddings }
    } catch { }
    $poolCeil = [long][math]::Floor(0.95 * ([double]($KvBytes - 1091000000) / (81859 * (1 - 0.2122))))
    $MaxLen = [int][math]::Floor([math]::Min([double]$modelMaxPos, [double]$poolCeil) / 4096) * 4096
    if ($modelMaxPos -le 0 -or $MaxLen -lt 71680) {
        Write-Host "context auto: pool too small ($poolCeil) or config unreadable -- falling back to the proven 71,680" -ForegroundColor Yellow
        $MaxLen = 71680
    }
}
# The splat below is built before this block ran in an earlier revision and
# shipped --max-model-len 0 to vLLM; keep the table in sync with the resolved
# value no matter where the computation ends up.
$splat.MaxLen = $MaxLen
$KvGiB = [math]::Round($KvBytes / 1GB, 1)
Write-Host ''
Write-Host '==== 170HX launcher ====' -ForegroundColor Cyan
Write-Host ("gpu            : " + $env:CUDA_VISIBLE_DEVICES + $(if (-not $TP2) { '  (single card; -TP2 for the untested dual-card experiment)' }))
Write-Host "graphs         : 64  (2026-09-26 D-run fix; derived default 8 leaves c>=2 eager)"
Write-Host ("context        : $MaxLen  " + $(if ($MaxLen -eq 262144) { '(model max_position_embeddings)' } elseif ($MaxLen -gt 71680) { "(pool-limited; model ceiling is 262,144)" } else { '(proven 3090 fallback)' }))
Write-Host ("kv pin         : $KvGiB GiB  " + $(if ($KvBytes -eq 5800000000 -and $KvGiB -lt 6) { '(fallback -- card was not empty)' } else { "(auto-fill: free VRAM minus $deadGiB GiB ghost zone minus ~$([math]::Round($nonKvGiB,1)) GiB non-KV minus $marginGiB GiB margin)" }))
$estKvTok = [long][math]::Floor(($KvBytes - 1091000000) / 81859 + 0.2122 * $MaxLen)
Write-Host ("                 ~{0:N0} KV tokens in the pool (fit formula; a full-length seq is ~78 KB/token)" -f $estKvTok)
Write-Host ("vision         : " + $(if ($TextOnly) { 'off' } elseif ($VisionOffload) { 'on, tower offloaded to RAM (3090-era)' } else { 'on, tower resident on the card (170HX default; unmeasured vs offload)' }))
Write-Host ''

& "$REPO\start_server.ps1" @splat
exit $LASTEXITCODE
