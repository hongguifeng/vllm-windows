# Brief for reviewer (gpt-6): Qwen3.8-Flash-Next on Windows native vLLM

You are a read-only reviewer. You may read the files listed in Appendix A. Do NOT
restart, stop, reconfigure or benchmark anything: the host's GPU0 (62.3 GiB) is
owned by a separate WSL2 engine belonging to the user and must not be touched.
GPU1 is currently idle and no server of ours is running.

Answer in English, concretely and quantitatively, ranked by expected value per
unit cost. Challenge our conclusions where you think they are wrong; tell us
which measurement you would run to settle each disagreement, and what it costs.

## 0. What we want from you

1. An audit of the porting fixes in §3 and the Windows PLE read path in §4: where
   do you see correctness or robustness risk that our tests did not cover?
2. A critique of our performance model (§7) and of the conclusions we drew (§8).
   Which conclusions are under-supported by the evidence we present?
3. Ranked recommendation for the next optimization step, given the ceilings and
   costs in §8/§9, and the list of things we have already disproven (§8.4).
4. Concrete decisive experiments, cheapest first, with a decision matrix
   ("if result A then hypothesis 1, if result B then hypothesis 2").

## 1. Mission and current status

Port a model that works under WSL2-native vLLM (`/home/hong/code/qwen3.8-flash-next-cmp170hx`,
vllm 0.29.1rc1.dev402+ga5a30471f.ple1, torch 2.13.0+cu130, triton 3.7.1) to
Windows-native vLLM in this repository. Route A (rebase onto `origin/main`) was
completed: 16 win32 conflicts resolved, rebuilt, structural smoke passed, and the
full 143 GiB checkpoint now serves end to end with the SSD-backed PLE path.

Current single-stream decode: **144.0 tok/s at MTP depth 2**, 116 tok/s at MTP
depth 1. Startup 170-250 s. GPU1 footprint 63.5 GiB.

## 2. Machine and environment facts

- Windows 11 native; two 64 GiB-class GPUs, SM80 (`TORCH_CUDA_ARCH_LIST=8.0`).
  GPU0 is the user's WSL2 engine; all our work uses GPU1.
- 88 GiB physical RAM. Windows free memory was 44-49 GiB during these runs.
  Commit limit 239.9 GiB. The project previously produced a BSoD when commit
  approached ~130 GB, so host-memory discipline is a real constraint.
- Measured pinned H2D bandwidth: 2.509 ms / 8 MiB = 3.3 GB/s on Windows,
  2.330 ms = 3.6 GB/s on WSL2. Both pinned by PCIe gen2 x8. Pinning buys only
  1.17-1.18x.
- Model: `Qwen4ExpForConditionalGeneration`. 48 layers = 36 GDN linear-attention
  + 12 QSA sparse-attention; MTP draft is 1 full-attention layer. head_dim 256,
  `hc_count=4`, `indexer_budget=2048`.
- PLE: a single layer (id 2), `ngram_size=3`, `heads_per_ngram=8` -> 16 heads x
  20M rows = 320M rows, 320 B per row = **95.37 GiB**, occupying shard
  `model-00001-of-00011.safetensors` alone. Total weights 143 GiB; GPU-resident
  47.32 GiB. 95.37 GiB does not fit in VRAM and does not fit in physical RAM, so
  the SSD backend is mandatory; upstream's pinned-host + UVA `EngramConfig(cpu_offload=True)`
  route is unusable here.
- Quantization: auto-round 3bpw consumed as `inc`; MoE layers are `RoutedExperts`
  with mixed bit-width per-expert overrides, served by the "humming" backend
  (pre-compiled cubins patched for this platform, see §3.6).
- Our vLLM build: `0.1.dev21640+gb629bcc71.cu133` in a worktree venv; torch
  2.11.0+cu130, triton-windows 3.6.0. WSL2 side has torch 2.13.0+cu130, triton 3.7.1.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` in our runs; the engine core and worker run
  in one process, which is why a single py-spy capture sees step dispatch and PLE.
- The step path actually taken is `EngineCore.step_with_batch_queue`, not `step`.
  vLLM's built-in `--enable-logging-iteration-details` only times
  `future.result()` on that path and reports ~0 ms; it is not a usable step metric.

## 3. Porting fixes shipped (please audit)

3.1 **Tree and build.** Rebased the fork onto `origin/main`; resolved 16 win32
conflicts (path handling, mmap, symm-mem, NCCL, MSVC toolchain files). Built with
MSVC. Two build-level bugs fixed:
  - MSVC/CuteDSL `C3545` template-error path and an `M_LOG2E` missing-define issue
    in a pre-compiled artifact.
  - vLLM `import uvloop` on a Windows path where it is unavailable.
3.2 **Model view skip bug.** A `.ple.` filename filter incorrectly skipped shards;
fixed, otherwise PLE weights never load.
3.3 **"humming" backend port.** 7 site-packages patches (file:line anchors in
ROUTE_A §6-§8) that make the INC quantization path accept `RoutedExperts` MoE
layers and route CUDA 2/3/5/6/7-bit weights to humming instead of dying in
`MoeWNA16Method`, plus patched cubins and a PE DLL. These are edits to installed
site-packages, env-gated, with `.orig` backups and a documented rollback. **They
are not yet archived as a `.patch` series** (open decision).
3.4 **QSA depends on vllm-flash-attn**, which had to be built on Windows; that
forced the MSVC fixes above.
3.5 **Incremental-compile hunk** is unconditional (not env-gated) in our installed
tree; noted as a maintenance hazard.
3.6 Pre-compiled humming artifacts were taken from the WSL side and patched, not
rebuilt from source on Windows.

## 4. Windows PLE SSD read path (the substantive engineering)

`vllm/models/qwen4_exp/nvidia/ple_ssd.py` (633 lines, upstream logic) plus a new
native reader `vllm/models/qwen4_exp/nvidia/ple_ssd_io_win.c` (327 lines) built
into `ple_ssd_io_win.dll`.

Design decisions we were forced into, and their root causes:
- **Synchronous unbuffered reads are refused process-wide on Windows**: opening
  with `FILE_FLAG_NO_BUFFERING` and doing a plain synchronous read returns
  error 87 (`ERROR_INVALID_PARAMETER`). Verified in a standalone C program.
- **Associating an IOCP with an unbuffered handle also fails with 87**, so the
  Windows path does NOT use IOCP. It uses one native reader thread with a depth
  queue plus a Python `ThreadPoolExecutor(workers)`, with **one event per
  request and `WaitForMultipleObjects`**.
- **Depth >= 128 failed with -87** until we realized the limit comes from
  `MAXIMUM_WAIT_OBJECTS = 64`, not from the filesystem. Depth is now capped at 64
  on Windows; the config knob `ple_ssd_io_depth` is set to 256 but the native side
  clamps to 64. (If that clamp is wrong somewhere, tell us: this is the kind of
  thing we want audited.)
- `O_BINARY` + `CreateFileW` with full share flags.
- Host staging buffers are `pin_memory=True`. On Windows
  `is_pin_memory_available()` returns True unconditionally; on WSL2 it is gated by
  `VLLM_WSL2_ENABLE_PIN_MEMORY` (default off, and their records say turning it off
  costs 30-35% there). On our machine pinning only buys 1.17x, so that 30-35%
  cannot reproduce here.
- The row cache is a plain `OrderedDict[int, bytes]` with
  `cache_limit = cache_mb*MiB // (row_bytes+128)`. Measured hit rate 77-80%.
- PLE requires eager or breakable CUDA graphs; vLLM auto-enables
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1` when it sees our additional-config.

Live behavior with full weights (measured with `VLLM_PLE_SSD_STATS=1`):
- `Future.result()` wait **9.8-10.4 ms per step**, of which
  `_ids_ready.synchronize()` (waiting for ngram ids to land on host) is
  **8.9-9.5 ms**. Actual row retrieval (cache + disk) is only 0.38-0.95 ms.
- Rows per step 350-520; 77-80% cache hits.
- Aligning PLE config (512 MiB row cache, 16 workers, prefetch 16384) cut disk
  reads from 15.5 MB/s to 3.3 MB/s (window reads 0.15 -> 0.03 GiB, ~4.7x less)
  with **no throughput change**. Disk is not the bottleneck.

## 5. Measurement discipline we were burned by (please sanity-check it)

- Their own records state that decode comparisons require >= 2500 warm-up decode
  tokens and low host memory (`vmmemWSL < 32 GB`); without those, the same
  configuration measured 22.2 ms vs 17.87 ms and 120.1 tok/s, i.e. a fake ±25%
  difference. We now use a steady-state single-stream tool
  (`_c1_traffic.py`: requests issued one at a time, always exactly one in flight,
  first N dropped, rate computed over the steady window), with >= 2700 tokens of
  warm-up and a free-RAM check before every run.
- Short 256-token runs have ±4-6% run-to-run spread; we do not accept conclusions
  from them.
- **Greedy output is not reproducible request-to-request on this engine** even on
  the same server (prefix-cache hits change prefill splitting and reduction order,
  so near-ties flip). Therefore text equality is NOT a usable correctness gate for
  A/B, and we stopped using it. Our correctness gate is now: adversarial unit tests
  plus a strict verification mode that compares against the reference implementation's
  own decisions (see §8.2).
- Benchmark provenance: the 110.99/667.13 numbers are from a *reference machine*
  (SATA SSD, 15 GiB RAM + 31 GiB swap) at **MTP=1**, not this machine. The
  131.19/305.42 numbers are this machine under WSL2-native at **MTP=2**. Any ratio
  computed across machine + MTP depth is void; we retired several such ratios.

## 6. Performance numbers (single stream unless noted)

| Config | c1 decode tok/s | mean acceptance length | step p50 (submit) |
|---|---|---|---|
| WSL2-native, this machine, MTP=2 (pass3) | 131.19 | (not recorded) | 17.2-17.6 ms |
| WSL2-native, this machine, MTP=1 (OPS 9.25) | 106.3-116.0 | 1.70 | 15.2-15.7 ms |
| Windows native, MTP=1, baseline (2 servers) | 116.8 / 115.7 | 1.91-2.00 (91-100%) | ~16.5 ms |
| Windows native, MTP=1, prepared launches | 114.9 | (not recorded) | 15.5 ms |
| Windows native, MTP=1, prepared + strict | 116.3 | (not recorded) | 15.6 ms |
| **Windows native, MTP=2, baseline** | **144.0** | **2.73 / 86.4%** | **17.4 ms** |
| **Windows native, MTP=2, prepared launches** | **143.5** | **2.73 / 86.7%** | **17.45 ms** |
| Windows native, MTP=1, PLE-free (struct view, graphs) | 132.4 / 133.0 (reproducible ±0.5%) | n/a | 14.6 ms |

Derived: at MTP=2 our per-step cost matches their MTP=2 per-step cost exactly
(17.4 vs 17.2-17.6 ms); our token-rate advantage is acceptance-rate driven
(2.73 vs 2.20 tokens per step). Our 86% acceptance is partly an artifact of a
highly repetitive single-stream prompt set, so it likely will not transfer to
real traffic. The PLE-free ceiling of 133 tok/s was only measured at MTP=1
(133 vs 113, +17%).

## 7. Current performance model

- Per step: submit = **15.3-17.5 ms wall**, of which only ~45% is on-CPU
  (py-spy non-idle samples; 1 sample = 10 ms of CPU at 100 Hz). The rest is
  blocked in `Future.result()` inside `_finalize_prefetch`.
- GPU work per step is only ~6 ms and the CPU is ~9 ms ahead of the GPU;
  utilization ~40%.
- `start_prefetch` does `self._stream.wait_stream(torch.cuda.current_stream())`,
  so the side stream queues behind everything already queued on the main stream.
  That is why the ids D2H wait measures ~9 ms.
- block/update/gap bookkeeping are 0.10/0.00/0.05 ms p50: scheduling is ~20 us and
  is not a bottleneck.
- PLE on GPU costs ~7.5 ms of GPU work per step; the SSD path converts that into a
  ~10 ms wait, net difference 2.5 ms/step (+17%). That is why removing the wait
  entirely is worth +17%, not +2x.
- The coordinator thread spends ~90% of its on-CPU samples inside
  `torch.cuda.synchronize()` spinning for ids. Whether that steals GIL time from
  the step thread is untested.

## 8. Optimizations attempted, with verdicts

8.1 **MTP depth 1 -> 2: +23.8%** (144.0 vs 116.3). Decomposed: tokens/step +40%
(1.95 -> 2.73) divided by step time +13% (16.8 -> 19.0 ms). Their same-machine
fair A/B was +13.5% (tok/step +29%, step time +12.5%). Same mechanism, different
magnitude, magnitude driven by acceptance rate. This is the one real gain we have.
An earlier conclusion that "MTP x2 is slower on Windows" (102.3 tok/s, p50 27.2 ms)
is **retracted**: it was measured before PLE settings were aligned, with different
instrumentation and load.

8.2 **Prepared Triton launches** (skip Triton's Python dispatch front end; cache
the `CompiledKernel`; cheap key over dtype, 16B pointer alignment, integer value
classes, exact constexpr values and compile options; refuse anything not fully
understood). Microbenchmark said 11.76 -> 5.29 us per call.
- Correctness: 17/17 adversarial unit checks (alignment flip, loss of divisibility
  by 16, stride 1 -> 2, constexpr change, options change, hook installed mid-run,
  kernel that reads a module global, changed tensor data). A strict verification
  mode re-ran Triton's own dispatch for every cache hit: **68,000 engine hits,
  0 mismatches**, covering steady decode, a 1592-token chunked prefill and a
  concurrent smoke test. (An initial 2 "mismatches" were our verifier's bug: the
  real `kernel_cache` key string embeds the full options repr while the binder
  returns `{}` when the caller passes no options.)
- Performance: **no change at either MTP depth** (114.9 vs 115.7; 143.5 vs 144.0).
- Worse: converting py-spy samples to CPU ms per step, CPU inside Triton dispatch
  paths went **up** with the fast path: MTP=1 0.85 -> 1.58 ms/step, MTP=2 1.14 ->
  1.56 ms/step. The microbenchmark saving did not transfer. Hypothesis: real vLLM
  launches are dominated by the launcher itself (global-scratch allocation, ctypes
  marshalling, `cuLaunchKernel`), which a prepared launch cannot bypass; the micro
  kernel had 8 pointers and 2 constexprs and no scratch.
- The previously claimed "1 ms off the submit window" is **downgraded**: baseline
  p50 itself swings 16.0-17.9 ms across servers, so the delta is inside the spread.
- Status: code kept, env-gated off by default. We see no reason to enable it.

8.3 **Fused mamba "align" block table** (one Triton kernel replacing six aten ops).
Bit-identical on 36 production widths, dtype preserved, 112.14 -> 32.22 us per call.
Throughput unchanged; kept as infrastructure, env off by default.

8.4 **Disproven, do not re-suggest**: `OMP_NUM_THREADS=1`; PLE 512 MiB / 16 workers
/ prefetch 16384 (disk reads down 4.7x, throughput unchanged);
`do_not_specialize` (a negative optimization; on Linux it is worse than the
default); `--async-scheduling` (no effect); fused align for throughput; prepared
launches for throughput; per-op micro-optimizations justified by the "Windows aten
tax" (see 8.5).

8.5 **Platform A/B on the identical microbenchmark** (empty GPU1; Windows
torch 2.11/triton 3.6 vs WSL2 torch 2.13/triton 3.7): Triton JIT call 11.76 us
(Windows) vs 14.66 us (Linux: Linux is slower); prepared launch 4.10 vs 4.32;
block-table 6 aten ops 91.8 vs 75.3 us (Windows 22% dearer); `slice.copy_` +32%;
`fill_` +20%; `cuda.synchronize()` +13%; small H2D helper 32 vs 16 us (Windows 2x).
Conclusion: the "Windows tax" is ~200-250 us per step, ~1% of a 27 ms step. It
does not explain the historical 46-78% gaps (those gaps were also computed across
machines and MTP depths and are void).

## 9. Open items and decisions

- Should the 7 humming site-packages patches be archived as a `.patch` series?
- Should the serve script's default flip to `-Graphs` (production should use graphs)?
- Is MTP=2 the right production default here (+23.8%) even though our acceptance
  measurement is prompt-set biased?
- The PLE-free ceiling (133 vs 113) was measured only at MTP=1. Before investing
  days into the ids-dependency work we would have to re-measure the ceiling at MTP=2.
- Two candidate levers remain, both expensive:
  (a) GPU-resident hot-row cache (~512 MiB ~ 1.66M rows, hit rate presumably near
      the host cache's 78%) to remove the ids D2H dependency, plus eviction and a
      miss fallback path;
  (b) compute ngram ids on the CPU (sampled tokens are known) which would need a
      CPU mirror of `ngram_context` kept consistent with the MTP draft, verified
      bit-for-bit.
- The unexplored CPU item is `gdn_attn.build`'s aten bookkeeping: ~9% of submit CPU
  ~ 0.65 ms/step. Given 8.2's lesson, cheaper CPU does not obviously mean faster.

## 10. Questions

Q1. In §3 and §4, what is the highest-risk defect you would look for first, and
    how would you test it without touching GPU0?
Q2. Is our "dispatch is not the bottleneck" claim (8.2) actually supported, given
    that the strict run (which pays an extra full dispatch per hit) measured
    116.3 tok/s, i.e. no slowdown? What would a stronger test look like, and what
    does it cost?
Q3. Given that per-step cost already matches the WSL2-native reference at MTP=2,
    what do you think limits us further: the ids dependency, the batch-queue
    overlap structure, GPU-side work we cannot see, or something else?
Q4. Where does the coordinator thread's ~90% `torch.cuda.synchronize()` spinning
    belong in this model? Would you test GIL contention, and how?
Q5. If you had one week, what would you build first: (a), (b), a cheaper PLE
    prefetch schedule, graph-side changes, or something not on our list?
Q6. What measurement protocol flaw in §5 would you add or remove?

## Appendix A. Files for audit (absolute paths)

- `D:\code\vllm-windows\vllm\triton_utils\prepared.py` (fast path + strict mode)
- `D:\code\vllm-windows\vllm\models\qwen4_exp\nvidia\ple_ssd_io_win.c` (Windows reader)
- `D:\code\vllm-windows\vllm\models\qwen4_exp\nvidia\ple_ssd.py` (row cache, prefetch, stats)
- `D:\code\vllm-windows\vllm\models\qwen4_exp\nvidia\model_state.py` (prefetch start, ngram context)
- `D:\code\vllm-windows\vllm\v1\attention\backends\utils.py` (`_mamba_align_gather_kernel`)
- `D:\code\vllm-windows\vllm\v1\engine\core.py` (`step_with_batch_queue`, `VLLM_STEP_STATS`)
- `D:\code\vllm-windows\vllm\platforms\cuda.py` (`is_pin_memory_available`)
- `D:\code\vllm-windows\_dev\docs\FLASH_NEXT_ROUTE_A_BUILD.md` (sections 10-17, the full build and measurement history)
- `D:\code\vllm-windows\_dev\docs\FLASH_NEXT_WINDOWS_FEASIBILITY.md` (model and machine facts)
- `D:\code\vllm-windows\_dev\probe\` (all measurement tools, including `_c1_traffic.py`, `_pyspy_submit_top.py`, `_prepared_launch_test.py`, `_step_rate.py`)
