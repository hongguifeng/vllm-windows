# Fork survey: `Morrowmake/vllm-cmp170hx` — what is worth importing

Surveyed 2026-10-05. The extracted patches live in `_dev/fork_cmp170hx/patches/`.
Neither they nor the commit dump is committed (both are generated); rebuild the
dump with:

```bash
git fetch morrow ampere
git log --no-merges --date=short --format='%ad|%s' main..morrow/ampere \
    > _dev/fork_cmp170hx/fork_commits.txt
```

## 1. What the fork is

| | |
| --- | --- |
| Repo | `Morrowmake/vllm-cmp170hx`, default branch **`ampere`** |
| Upstream base | `e55d076f89` (2026-09-25), `v0.30.1rc0-181` (README) |
| Our base | `main` = `3ce95f3619` (2026-09-18); merge-base with the fork = `71fc70d3ae` |
| Commits ahead of our `main` | **570** (555 non-merge): **382 carry an upstream PR number** (upstream delta we simply don't have), **173 are fork-original** |
| Target | NVIDIA CMP 170HX (GA100, sm_80), 180 W/card, PCIe-only (no NVLink) |
| First supported model | GLM-5.3-Flash: 320B MoE, W4A16, 262,144 ctx, DFlash2 spec decode |
| Layout | TP4 and PP4 across 4 cards; recipe repo pins an exact fork commit |
| Conventions | every perf feature behind an env var, **default off**, with a startup banner and a kill-switch table; model-specific work is gated by model family and its kernels check the exact shapes they were built for and hand anything else back to upstream |

Published numbers (recipe release 1.5.0, TP4): streaming decode 267 tok/s at C=1,
759 tok/s at C=8, cold prefill 2,657 tok/s, KV pool 1.18M tokens at 262K ctx.

**The fork is Linux-only** (containers, `/dev/shm`, `torch.utils.cpp_extension.load`
with a POSIX toolchain). Nothing in it is Windows-aware.

## 2. Applicability filter for this project

Our rig: 2x CMP 170HX (sm_80, 64 GB), **Windows**, single-GPU serving of
`Qwen3.8-Flash-Next` (`model_type=qwen4_exp`) and `Qwen3.8-27B-W4A16-AutoRound-fast`.

`qwen4_exp` text config: 48 layers, `full_attention_interval=4` (hybrid: 12 full-attn

+ 36 GDN linear-attention layers, conv kernel 4, `mamba_ssm_dtype=float32`), MoE with
**512 experts / top-10 / shared expert**, `hc_count=4` hyper-connections (mHC),
n-gram **PLE** (`ngram_size=3`, `heads_per_ngram=8`) which we offload to SSD,
`mtp_num_hidden_layers=1`, QSA indexer (`indexer_budget=2048`, `compress_ratio=4`),
AutoRound 2/8-bit mixed quant (`packing_format=auto_round:auto_gptq`, group 64).
Spec decode: MTP (3 tokens) or DFlash2. `--max-num-seqs 4`, `--max-num-batched-tokens 2048`.

Measured on this card (`_dev/docs/BENCH_170HX_BASELINE.md`,
`_dev/out/_ab_results/170hx_vs_3090.md`): decode ITL50 ~20 ms at C=1,
prefill 2,513 tok/s @1k, 2,660 @4k, 2,527 @16k, 2,341 @32k.

### Not applicable (do not import)

+ **Sparse-MLA / DSA indexer kernels**, `TRITON_MLA_SPARSE` backend, split-K decode,
  PTX e4m3 dequant, Gluon prefill — our attention is full attention + GDN; our QSA
  top-k goes through `torch.ops._C.{cooperative,persistent}_topk`, not through
  `indexer_topk.py`, so the tie-fix/sort work does not transfer mechanically.
+ **GLM-5/5.3 fused decode and prefill kernels** (mHC, KDA, MoE router v2, prologue
  fuse, thin GEMM v74) — shape-locked to GLM's head counts and widths.
+ **Compiled Ampere Marlin** (`vllm._ampere_marlin_C`) — guarded by "SM 8.0, bf16
  activations, 288 local/global experts, top-8 routing, hidden width 4096, uint4b8
  group-128, SiLU clamp 10.0". Our 512 experts / top-10 / hidden 2560 / AutoRound
  group-64 is ineligible, and it would need an MSVC+nvcc build here.
+ **Pipeline-parallel knobs** (`VLLM_PP_SPREAD_DECODES`, `VLLM_PP_PACKED_HOP`,
  `VLLM_PP_DRAFT_TAIL_STAGE`) and the PP block-table bugfix — we never run PP.
+ **Host-staged all-reduce / PCIe-P2P custom all-reduce** — TP-only, Linux-only
  (`/dev/shm` + a JIT C++ extension). See tier 3.
+ **Engram huge-page packing** (#56926) — Linux `MADV_HUGEPAGE`; the target is
  `deepseek_v41`, not `qwen4_exp`.

## 3. Tier 1 — cherry-pick now (small, `vllm/` side applies cleanly with `git apply -3`)

| Commit | Subject | Why it matters here | Apply |
| --- | --- | --- | --- |
| `d5051abaf1` | [Bugfix][Mamba] Restore prompt-tail prefix-cache hits with MTP (#58368) | One line in `MambaManager` (`use_eagle` → `drop_eagle_checkpoint_block`). Our model is GDN (`MambaSpec`) + MTP + prefix caching — the exact configuration the bug hits. | clean |
| `48d8880d09` | [Bugfix][Qwen4Exp] Keep pinned PLE prefetch ids out of the CUDA graph pool (#58489) | Our PLE does a side-stream pinned-host prefetch under breakable CUDA graphs; graph-pool segments could be reused under the in-flight ids. | clean |
| `6491f481a7` | [Bugfix] Stop allocator fragmentation from shrinking the KV cache during memory profiling (#58430) | MoE workspaces grow during the dummy profile run and free smaller buffers; without a split limit the profile peak is inflated and the KV pool comes out smaller. We run `--gpu-memory-utilization 0.96` with auto-filled KV. | clean (test file conflicts) |
| `71891bdb94` | [Perf] Parallelize registered CUDA Triton kernel warmup at startup (#58582) | Startup wall time. | clean, `envs.py` included |
| `e6c07ea576` | [Bugfix][KV Cache] Fix incremental multimodal block hashing (#51694) | Vision is on by default and prefix caching is live. | clean |
| `f6aa2919bc` | [Bugfix][Core] Keep every multimodal feature in the partial-block KV event (#58288) | Same area. | clean |
| `70fc359d25` | [Bugfix][KV Offload] Retain offload event metadata through batch translation (#57453) | We run KV offload on Windows (`_dev/docs/KV_OFFLOAD_WINDOWS.md`). | clean |
| `62a08ef13b` | [Bugfix] Clamp the sampler's block argmax to the vocabulary | Our `vocab_size=248,320` is not a multiple of the Triton block width; a degenerate tile can emit an id ≥ vocab. Fork-original, refs vllm#50843. | clean |

## 4. Tier 2 — features worth the work

| Commit | Feature | Payoff here | Cost |
| --- | --- | --- | --- |
| `d26e7d2548` | **Fair chunked prefill**: `--prefill-chunk-with-decodes N`, `--max-num-partial-prefills N` (both 0 = upstream behaviour, byte-identical schedule) | Our step budget is 2048 tokens and prefill runs ~2.5K tok/s, so while someone prefills, every step is ~0.8 s and co-resident decoders see ITL jump from ~20 ms to ~800 ms. Capping the prefill at 384 while anything decodes brings the step to ~150 ms. The second knob stops a short prompt from queueing behind a 30K-token prefill. | 3 conflicts in `scheduler.py`, 3 in `arg_utils.py` — all context drift (our file inlines `self.scheduler_config.long_prefill_token_threshold`, and lacks `_select_adaptive_k`, which their patch assumes). Their patch also touches `_mamba_block_aligned_split`, which we need for GDN. |
| `63dac5ab8f` | **Hybrid KV: charge mamba speculative scratch once per running request** (`KVCacheSpec.speculative_scratch_bytes`) | Accounting only, but it changes the reported KV capacity and the single-request fit check: +14.7% reported tokens on GLM at MTP n=3. Our model is hybrid GDN with MTP/DFlash and `max_num_seqs=4`; our launcher's auto `MaxLen`/`KvBytes` fit formula reads exactly this number. | clean |
| `d508456f85`, `abe9524dba` | `VLLM_KV_MAMBA_INFLIGHT_STATES`, `VLLM_KV_SWA_INFLIGHT_SCRATCH` — reserve what in-flight prefill chunks really hold | Only binds under async scheduling / PP; opt-in, one log line. Cheap insurance for the capacity report. | clean-ish |
| `ac877d6ad7` | `[Worker] Attribute start-up memory to modules and profile stages` (`VLLM_GLM5_MEM_ATTRIBUTION`) | Tells us where the 62 GiB goes at boot — directly useful for the 0.96 ceiling and the auto KV budget work. Mostly one new file. | `envs.py` conflict only |
| `031d63da62`, `9210eca1e6` | **Pinned Triton autotune on sm_80** (`VLLM_GLM5_FLA_PIN_AUTOTUNE`): `PinnedAutotuner` + a per-key config table | Our GDN prefill uses the same vendored `flash_linear_attention` ops, and every one of them is `@triton.autotune`'d with keys that exclude sequence length — so a boot keeps whichever config won for the first shape it saw, and most candidates change the reduction order ⇒ **boot-to-boot output differences**, plus autotune benchmarking at startup. The mechanism transfers as-is; the pin table must be re-measured for our GDN shapes on this card. | `envs.py` conflict only; a measurement sweep is the real cost |
| `d945300e8d` | **Acceptance-adaptive per-step draft count** (`adaptive_k` in `--speculative-config`) | Decides on the scheduler's CPU thread (composes with backends that reject device-side trimming) and reacts to content rather than batch size. We run MTP k=3 / DFlash2. Their GLM result: a fixed k latched at 2 and cost 16% of aggregate throughput. | 10 files, conflicts in `model_runner.py` and `speculator.py`; captures a uniform decode graph per k, so our capture-size list must cover the whole k range (cf. the `-MaxGraphCapture 64` finding) |
| `46cc42f176` | `[MoE] Optionally route padding rows to no expert` (`VLLM_GLM5_MOE_MASK_PADDING`) | CUDA-graph padding rows carry stale hidden states and take expert slots; Marlin's K split depends on the block layout, so real rows get last-bit different MoE output depending on the previous batch. We are a 512-expert MoE with graph padding. | `envs.py` conflict only |
| `3ddd38cf3e`, `638709c3d6` | Deterministic within-expert order for `moe_align_block_size`, then a chunked counting sort to keep it cheap | Determinism for the same reason; the second commit is the perf repair for the first. | conflicts in `envs.py`, `moe_align_block_size.py`, `kernel_warmup.py` |
| `561bfdcccd` | Reuse finalized DFlash context at cached boundaries | DFlash2 + prefix caching, which is our default spec-decode path on the 27B. | clean in scheduler/coordinator; `envs.py` + `kv_cache_manager.py` conflict |
| `04730e8270` | [DFlash] Capture the context K/V precompute in the draft CUDA graph (#57632) | Upstream, DFlash drafter. | clean |
| `5a57fd033d` | Frozen DFlash confidence logging + vectorized draft masking | Diagnostics for our DFlash2 acceptance behaviour. | conflicts in `model_runner.py`, `dflash2/speculator.py` |

## 5. Tier 3 — only if we go multi-card

| Commit | What | Verdict for us |
| --- | --- | --- |
| `0ec2a7af3d` | Host-staged all-reduce for PCIe-only nodes without P2P (`VLLM_GLM5_HOST_ALLREDUCE`, ~985 lines + tests) | Needs a Windows port: named file mapping instead of `/dev/shm`, and an MSVC build for the JIT extension. Only pays off if TP2/TP4 actually runs — it never has on this rig. |
| `61f9ac9328` | Let the custom all-reduce accept PCIe P2P (`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE`, `VLLM_CUSTOM_ALLREDUCE_ALGO`) + a 208-line P2P gate in `platforms/cuda.py` | Would need to know whether the 170HX pair grants peer access on Windows; today TP needs the SystemPanic `nccl-windows` build via `VLLM_NCCL_SO_PATH`. |
| `f671f1932d`, `016f4ff528`, `29140666b6`, `997b6ea14a` | Flags-in-data two-shot custom all-reduce; timed-out wait traps instead of returning; host-shm setup handshake across ranks | Same TP-only bucket; the "agree on setup across ranks" fix is the one to take if we ever enable the host path. |
| `5fa416150a`, `666f36dff7`, `49753f11bc`, `3de0edb3f6`, `378c37b009` | PP scheduling: spread decodes over micro-batches, pack each hop into one transfer, run the DFlash2 draft tail on an earlier stage | PP-only. |

## 6. Suggested execution order

1. Tier 1 as one batch of cherry-picks (they are independent and small):
   ```bash
   git remote add morrow https://github.com/Morrowmake/vllm-cmp170hx
   git fetch --depth 1000 morrow ampere
   for c in d5051abaf1 48d8880d09 6491f481a7 71891bdb94 e6c07ea576 f6aa2919bc 70fc359d25 62a08ef13b; do
       git cherry-pick -x -n $c || git cherry-pick --abort   # -n: stage, resolve, then commit yourself
   done
   ```
   Every commit that adds an env var conflicts in `vllm/envs.py` — that is the only
   recurring conflict and it is mechanical.
2. `63dac5ab8f` (hybrid KV scratch) — verify against the launcher's auto-fit formula
   before trusting the new `kv_cache_size_tokens`.
3. Fair prefill (`d26e7d2548`) + `ac877d6ad7` (memory attribution), then A/B ITL with a
   concurrent long-prefill load; the payoff is measurable with the existing bench suite.
4. Pinned fla autotune (`031d63da62`) — port the mechanism, then sweep GDN shapes on
   GPU1 to build our own pin table.
5. `adaptive_k` (`d945300e8d`) and the MoE determinism trio, only after 1–4 land.

Note on ordering: their fair-prefill patch is authored on top of `d945300e8d`
(its context includes `_select_adaptive_k`). Either pick `d945300e8d` first or resolve
the one hunk by hand — the conflict is context drift, not a logic clash.

## 7. Expected gains on this rig (single GPU, Qwen3.8-Flash-Next)

Facts taken from a real boot (`_dev/out/diag/gpu1_auto_kv_reload/before_qwen38-flash-next-win_20261002_133131.stdout.log`)
and from the launchers:

+ `enable_prefix_caching=True`, `enable_chunked_prefill=True`, `--max-num-batched-tokens 2048`,
  `--max-num-seqs 4`, `--mamba-cache-mode align`, `--async-scheduling`,
  `cudagraph_mode=FULL_AND_PIECEWISE`, capture sizes `[1,2,3,4,6,8,12,16,24]`,
  `speculative_config={method: mtp, num_spec_tokens: 2..3}`, PLE SSD (95.37 GiB on disk,
  512 MiB row cache, 16 workers, prefetch 16,384), vision on,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.
+ Attention/hash block size is **1600 tokens** (`Setting attention block size to 1600 tokens`,
  `kv lcm block sizes 1600`) — that is the prefix-cache hit granularity.
+ Boot: weights 184 s (target) + 11 s (MTP drafter), model load 46.53 GiB / 207 s,
  graph capture 4 s / 0.39 GiB, `init engine ... took 10.06 s` on a warm compile cache;
  on a cold compile cache the same boot was ~221 s with ~90 s of compilation
  (`_dev/out/_ab_results/170hx_vs_3090.md`).
+ `WARNING ... Speculative decoding (method=mtp) is enabled but no KV cache group could be
  identified as the draft model's.` — our MTP is exactly the configuration #58368 fixes:
  the coordinator sets `drop_eagle_checkpoint_block=True` (MTP is eagle-family) while the
  MambaManager's `use_eagle` stays `False`, so the prompt-tail junction is registered one
  hash unit too high and the tail hit misses.

| Item | Gain type | Expected size here | Verdict |
| --- | --- | --- | --- |
| `d5051abaf1` mamba prompt-tail hits with MTP | TTFT | up to one hash unit = **1600 tokens** of prefill skipped per continuation/repeat request (~0.6 s at 2.5K tok/s; a fully cached repeat goes from ~0.6 s to ~0.05 s) | take; verify via the `Prefix cache hit rate` line |
| `48d8880d09` PLE prefetch ids out of the graph pool | correctness | removes a real aliasing window: our ngram ids are built inside eager-break graph segments and the SSD/pinned lookup runs on a side stream; cost is one persistent int64 buffer (~MB) | take |
| `6491f481a7` allocator fragmentation while profiling | KV size | only on the auto-KV path (`KvGiB 0`); profile peak comes down ⇒ more free memory ⇒ bigger pool. Also our `expandable_segments:False` is no longer reset twice by the old setter. No effect when `--kv-cache-memory-bytes` is pinned (profiling is skipped) | take, then compare `Available KV cache memory` |
| `71891bdb94` parallel Triton warmup | startup | cold-cache boots only (the ~90 s compile); autotuning kernels still compile serially by the patch's own guard | take, cheap |
| `e6c07ea576`, `f6aa2919bc` mm block hashing / KV event | correctness + TTFT | vision is on and prefix caching is live; text-only throughput unchanged | take |
| `70fc359d25` KV offload event metadata | correctness | dormant unless we turn on the Windows KV offload (KV is fully resident on 64 GB today) | take as insurance |
| `62a08ef13b` sampler vocab clamp | robustness | `vocab_size=248,320` is not a multiple of the Triton block width; no throughput change | take |
| `d26e7d2548` **fair chunked prefill** | ITL | the big one. Budget 2048 tokens at ~2.5K tok/s ⇒ **~0.8 s per step** while anyone prefills ⇒ co-resident decoders go from ~20 ms ITL to ~800 ms. With `--prefill-chunk-with-decodes 384` the step is ~0.15 s ⇒ ITL ~150-170 ms, prefill throughput roughly unchanged. `--max-num-partial-prefills 2` stops a short prompt queueing behind a 30K prefill. No change at C=1 (full budget is kept when nothing decodes) | take; verify prefill does not stall (cap 384 < block size 1600 exercises the sub-block path in `_mamba_block_aligned_split`) |
| `63dac5ab8f` hybrid KV speculative scratch | reported capacity | **no-op at 262K**: the new branch only fires when the pool holds more than `max_num_seqs` full-length requests, and the engine reports `Maximum concurrency for 262,144 tokens: 1.89x` (< 4). It pays off only if we lower `max_model_len` (e.g. 71,680 ⇒ ~7x > 4, +10-15% reported capacity) | skip unless we serve shorter contexts |
| `d508456f85`, `abe9524dba` in-flight state / SWA scratch | reported capacity | only binds under async scheduling / PP; opt-in, one log line | cheap insurance |
| `ac877d6ad7` start-up memory attribution | diagnostics | breaks down the 46.53 GiB weights + PLE + graphs + workspaces at boot; no runtime change | take for the 0.96 ceiling work |
| `031d63da62` pinned fla autotune | determinism + startup | our GDN prefill ops are autotuned with `key=["H","K","V","BT"]` — **no sequence length** ⇒ a boot keeps whichever config won for the first shape, and candidates change `BV`/`num_warps`/`num_stages`, i.e. the reduction order ⇒ two boots can return different prefill output/logprobs. Pinning also removes ~17 autotuners x 10-24 configs of benchmarking from the first prefill. The pin table must be re-measured for our shapes (H=16/48, K=V=128, BT=64) | take the mechanism, then sweep |
| `d945300e8d` adaptive_k | decode throughput | MTP-only setup; saves verify width when acceptance is low (our tok/step ~1.9 with k=2-3). Needs one uniform decode graph per k, so capture sizes must cover the k range — more capture memory and a longer capture. Payoff uncertain until measured | measure before committing |
| `46cc42f176` + `3ddd38cf3e` + `638709c3d6` MoE padding mask, deterministic moe_align | determinism | real decode batches get padded to 6/8/12/16/24; stale padding rows take expert slots and shift the grouped-GEMM block layout and accumulation order ⇒ run-to-run logprob jitter. Gain is repeatability, not speed; the counting-sort commit pays back the align overhead | take if we want bit-stable logprobs |
| `561bfdcccd`, `5a57fd033d`, `04730e8270` DFlash | TTFT/throughput | **not for flash-next** (MTP drafter). They pay off on the `Qwen3.8-27B` + DFlash2 service | route to that service |

TP=1 means there is no all-reduce at all on this rig, so every tier-3 item (host-shm,
PCIe P2P, flags-in-data) is a zero by construction.

## 8. The other 382 commits

382 of the fork's commits are plain upstream (PR-numbered) commits from the week
between our `main` (2026-09-18) and their base (2026-09-25). Cherry-picking individual
fixes out of them is what sections 3–4 do; the alternative — rebasing `flash-next-win`
onto `origin/main` or their upstream base — would re-expose all of them at once, but
that is a rebase of our ~60 Windows/Flash-Next commits and needs its own plan.
