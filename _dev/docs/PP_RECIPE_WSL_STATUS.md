# AlexBrandes PP recipe on this box (WSL2) — state and measurements 2026-10-24

> The attempt to run the same 4-bit checkpoint on the native Windows fork is
> documented in [W4A16_WINDOWS_VS_WSL_FINDINGS.md](W4A16_WINDOWS_VS_WSL_FINDINGS.md):
> it loads there but cannot initialise its MoE under PP; the six blockers and the
> effort estimates are recorded in that file.

## Where things live
- Recipe checkout: `~/pp-recipe` (WSL ext4, on D: via `D:\wsl\ext4.vhdx`)
- Checkpoint: `~/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound` (168.8 GiB, 31 files,
  every file sha256-verified against the ModelScope manifest; 19 LFS blobs match the pinned
  HF revision `4c67bf68`)
- Draft head: `~/pp-recipe/models/draft-head-w4a16.safetensors` (330,265,872 B, sha256 matches
  `versions.env`)
- Image: `ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d…` (30.7 GB on disk)
- Logs: `~/pp-recipe/{image.log,ms_dl.log,ms_fix.log,start.log}`, container `hong-pc`

## WSL adaptations applied (all needed; upstream assumes bare metal)
1. `start.sh`: `--gpus '"device=1,2"'` → `--gpus all -e CUDA_VISIBLE_DEVICES=1,2`.
   Docker's device selector is ignored under WSL; without this the engine takes the 3090.
2. `vm.overcommit_memory=1` (set once per WSL boot via a `--privileged` container).
   Otherwise the loader's mmap of the 95.37 GiB PLE shard fails with ENOMEM under load.
3. `-e VLLM_WSL2_ENABLE_PIN_MEMORY=1`. vLLM gates pinned memory on WSL behind this
   (CUDA graph capture refuses unpinned H2D copies without it).
4. `HOST_TUNE=0` in `.env` — `host-tune.sh` needs sudo and `nvidia-smi -pl` is refused in WSL.

## Model config as served
PP=2 (26/22 layers), W4A16 AutoRound, MTP **k=3** (best of the sweep), block 1680, 262,144 ctx,
`FULL_AND_PIECEWISE` CUDA graphs, KV **1,245,920 tokens** (4.75× at 262k; k=5 gave 1,188,104).

## Measured (this box: WSL 48 GB RAM, PCIe Gen2 x8, 170HX x2)
| level | tok/s |
|---|---|
| c1 aggregate / per-request decode | 68.3 / 72.6 |
| c2 | 110.4 |
| c4 | 174.4 |
| prefill (13.8k prompt) | 2,766 |

Reference host (Ubuntu, 93 GB RAM, Gen2 x4): c1 157–226, c2 212–362, c4 385–560, prefill ~6,900.

## Why it is slower here: PLE table reads hit NVMe
- Decode of 300 tokens (4.0 s) drove **9,413 reads / 39.3 MB / 2,336 IOPS / 9.7 MB/s**,
  average 4.1 KiB per read (from `/sys/block/sdd/stat`).
- The n-gram table is 95.37 GiB; this VM has 48 GB, so the page cache can hold under half
  of it (reference host has 93 GB). Every miss costs ~0.19 ms (measured 4 KiB random read).
- The fork says so itself: `PLE host-ids: worker publish slower than 5.0s (NVMe gather contention?)`.
- MTP is working (mean acceptance 2.5–2.7 of 5 drafts), so the shortfall is not speculation.

## MTP depth sweep (median of 5 passes each, same prompts, salt per run)
| level | k=5 | k=3 | delta |
|---|---|---|---|
| c1 aggregate | 72.0 | **77.9** | +8% |
| c2 aggregate | 114.0 | **128.3** | +13% |
| c4 aggregate | 200.3 | 202.9 | +1% |
| prefill | 3,484 | **3,616** | +4% |

Both configs show a cold first pass (c4 ~125 on pass 1, ~200 afterwards), and c1 varies
±20% run to run, so only medians over several passes are meaningful. Fewer draft positions
means fewer PLE table lookups per step, which matches the NVMe-bound diagnosis above.

## Next levers, cheapest first
1. Lower `num_speculative_tokens` (5 → 3): measured +8% c1, +13% c2, +4% prefill (k=3 now the default).
2. More WSL memory: hit rate is ~22% today (see below), so 48 → 72 GB buys roughly +20–25% c1
   (≈95–105 tok/s), not the +100% an earlier note implied. The table is 95.37 GiB and the host has
   88 GB, so a fully resident table is impossible here.
3. Accept: prefill 3.5–3.8k tok/s; decode is NVMe-bound, not link-bound.

## Hit-rate arithmetic (why more RAM only helps a little)
- PLE gather is 16 n-gram rows per token position (`heads=16`, `head_dim=160`).
  With k=3 → 4 positions per step → 64 lookups/step; ~37 steps/s → ~2,370 lookups/s.
- Measured disk reads during decode: 1,854/s at 4 KiB (`/sys/block/sdd/stat`).
  ⇒ **hit rate ≈ 22%** with ~37 GB of page cache against a 95.37 GiB table.
- Each miss costs ~0.19 ms (measured 4 KiB random read), i.e. ~0.35 s of NVMe wait per second
  of decode — the dominant per-step cost, and the reason c4 is bimodal (concurrent streams
  multiply misses on one drive).
- `VLLM_PLE_MMAP_PREWARM=1` exists (streams the table once at load) but cannot keep it resident.
