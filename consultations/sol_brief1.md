# Round 8: the host-side step cost now has a call site and a composition. Is it worth attacking, and in what order?

New reviewer (GPT-6.1 Sol). Rounds 1-7 are settled; do not re-litigate them, build
on them. Read-only: do not restart, stop or reconfigure any running service or the
host; no GPU-heavy benchmarks. Answer in English, concretely and quantitatively,
ranked by expected value per unit cost. Where you are unsure, say what measurement
would settle it rather than guessing.

## 0. Instrumentation wall (settled, do not re-propose)

This GPU is a CMP mining-line SKU. CUPTI activity tracing, GPU metrics, Nsight
Systems tracing and torch.profiler CUDA activity are all blocked at the driver
level; elevation does not restore injection. Kernel-level GPU attribution is only
possible on the Linux/WSL driver side, which we are not using. Therefore the only
instruments that work here are: py-spy host CPU sampling, CUDA events around the
replay (the engine's own step windows), and in-process wall-clock timers. Two
consequences you should reason with:
- py-spy gives CPU time, not a position on the timeline;
- a host thread's "time inside the step" is partly a measurement of cadence, since
  the step thread is inside the step for essentially the whole period.

## 1. Established in rounds 1-7 (use as given)

1. The PLE/SSD host path is off the critical path. Row-assembly p50 varied 38x across
   decode windows while step length varied +/-1.5%; r(rows, step) = -0.676; the ids
   wait (p50 11.2-14.1 ms) has r(ids, step) = +0.001.
2. Acceptance length on our fixed code-weighted table at MTP x2 is 2.35, implying a
   67.5% per-position accept rate; the reference engine's live /metrics shows 68.3%
   accepted/draft. Acceptance is at parity. Their 131.19 tok/s was measured on ~115k
   prompt tokens per request with 95.3% prefix-cache hits, so it is not comparable to
   our fixed short-prompt table. Only per-step cost is comparable.
3. **Our own section 17 already measured per-step engine cost at parity with the
   reference engine: our submit p50 17.4 ms vs their 17.2-17.6 ms.** So there is no
   engine-level step-time gap to close. Any gain we are discussing is self-improvement
   inside our own engine, not catch-up.
4. PLE GPU work is roughly 10% of the step (net-difference screen: removing the PLE
   structure at MTP x2 gave +10.4%).
5. Named GPU levers: GDN/linear attention ~8.0 ms = 54% of forward device work; the
   drafter 2.36 ms = 13.7%; Triton dispatch ~1.14-1.44 ms/step for ~14 launches.
6. Ruled out with measurements: OMP/MKL pinning to 1 (production median 119.4 vs
   120.3, noise); Triton prepared-launch (submit p50 17.4 -> 17.45 at MTP x2);
   thread contention; breakable CUDA graphs and async scheduling (both auto-enabled
   in our build, verified from our own logs before spending an experiment).
7. Funding rule: a promoted change must move the fixed-table production median by
   at least 5% from baseline 119.2 tok/s. The meter is a fixed B arm. Host memory
   moves rates about 25% by itself, so every block records it. Warm-up gate: two
   consecutive blocks within 2% and no weight-load frame, otherwise discard the last.

## 2. New since round 7: the host cost is now localised

### 2.1 Cadence, device span and the idle gap (three 60 s captures, two configs)

Device span p50 per step held at **17.16 / 17.50 / 17.29 ms** while the cadence
drifted **19.47 / 20.23 / 20.85 ms**. All the drift is host side. The idle gap
(cadence minus device span) is **2.35 / 2.09 / 2.53 ms per step**, stable across
captures and across whether threads were pinned. **This gap is the ceiling on what
any host-side saving can buy.**

py-spy, three captures: the step thread is 12.2% of all samples every time; the
`Future.result` wait inside the PLE builder is 11.23 / 11.20 / 11.50 ms per step
(the stable term from round 7, already known to be covered by GPU work); remaining
"real CPU" 7.91 / 8.19 / 9.14 ms per step, which also drifts with cadence. Reliable
readings are ratios: attention backends 14.5 / 14.8 / 16.1% of step samples; Triton
dispatch 6.5 / 7.1 / 7.0%.

### 2.2 A wall-clock timer on build_attn_metadata (env-gated, three windows, two restarts)

Per step there are ~2 builds. Total build wall time **3.13 / 3.16 / 3.21 ms per step**
against idle gaps of 2.97 / 2.39 / 3.06 ms in the same windows. Builder time is 95.8%
of it. Splitting by batch size and call site:

| call site | calls | ms/step | p50 ms/call | when |
|---|---|---|---|---|
| `mamba_hybrid.py:313` (model side) | ~2,900 | **2.84** | **2.92** | before the forward is enqueued |
| `speculator.py:349` (drafter) | ~2,900 | 0.40 | 0.42 | after the enqueue |

Composition per step:

| builder | calls | ms/step | share | ms/call |
|---|---|---|---|---|
| `GDNAttentionMetadataBuilder` | 8,700 | **1.77** | 56% | 0.61 |
| `QSAMetadataBuilder` | 14,500 | 0.84 | 26% | 0.17 |
| `PleShortConvAttentionMetadataBuilder` | 2,900 | 0.56 | 17.5% | 0.57 |

The wall time (2.9 ms) is far larger than py-spy's CPU for the same region (~1.4 ms),
so roughly 1.2 ms is the thread **not running**. The prime suspect is three places in
`gdn_attn.py` where a CPU boolean mask indexes a GPU tensor, which becomes
`masked_select` and must therefore synchronize the device to size its output. In our
steady state (every sequence is spec-decoding, `num_prefills == 0`,
`num_decodes == 0`, so we take the cheap branch that already exists) exactly two of
those run each step:

```python
spec_state_indices_tensor = block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]
num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
```

### 2.3 Three "they differ from us" hypotheses were tested and all died

| Hypothesis | Evidence | Verdict |
|---|---|---|
| They have cross-KV-group metadata reuse (upstream #58762, merged 09-28) | their base commit `a5a30471f` authored 09-19, earlier than the merge; `grep -c update_block_table` on their MRV2 runner = 0 | no, neither side has it |
| They run the legacy runner (MRV1 has `update_block_table` reuse) | their server log contains only `[model_runner.py:...]`, never `gpu_model_runner.py` | no, same MRV2 |
| Their mamba block-table computation differs | their `utils.py` body is line-identical to our eager path, and their `envs.py` has no `VLLM_MAMBA_ALIGN_FUSED` at all | no, same 5-op code |

So the suspected inefficiencies are **shared**, and removing them would be a way to
get ahead rather than to catch up. Upstream #58851 (draft, unmerged) implements
cross-group metadata sharing; with our configuration there are only ~1.5 GDN groups
per build, so sharing would cut about a third of GDN, ~0.57 ms/step, ~+2.8%: below
our bar. That is why we are considering a different mechanism.

### 2.4 The one measurement gap that remains

We have shown the model-side build is ~2.9 ms of wall time and that it runs before
the forward is enqueued. We have **not** shown that this time is what the GPU idles
through. The evidence is only magnitude: 2.9 ms fits inside a 2.4-3.1 ms gap. It
could instead overlap with device work that was enqueued by the previous step.

## 3. The two candidate next steps

**A. Free (environment variable only, no code).** `VLLM_MAMBA_ALIGN_FUSED=1` switches
`mamba_get_block_table_tensor` from five torch operations (integer floor divide,
in-place clamp, an arange allocation on device, an int64 conversion copy, a gather)
to one Triton kernel. py-spy puts those operations at 0.32-0.42 ms of CPU per step.
**The fused branch exists only in our tree; the reference engine does not have it**, so
if it works it is a differentiator. Cost: one restart (~4 minutes) plus two measured
blocks (~15 minutes). Risk: our Triton dispatch penalty is ~30-60 us per launch and
the tensor-descriptor path was already measured as expensive elsewhere.

**B. About 15 lines of code.** In `gdn_attn.py`, when all sequences are
spec-decoding (`num_spec_decodes == num_reqs`, so the mask is all true) replace the
two masked-selects with plain slices, which we believe return identical values, and
thereby remove two device synchronizations from every decode step. Cost: deploy,
restart, three measured blocks, ~35 minutes. Correctness argument needed before
running: see Q3.

If both work, combined estimate 1.5-2.5 ms per step, i.e. +7% to +12%. The ceiling
for any host-side work is the idle gap: +11% to +15%.

## 4. Questions

### Q1 Ordering and cheaper alternatives
Is "measure A (free env), then code B" the right sequence, or is there a cheaper
decisive experiment we have not thought of? Rank the options you would consider.

### Q2 In the gap or overlapped?
What is the cheapest way, using only wall clocks, CUDA-event step windows and
py-spy, to establish whether the model-side build is what the GPU idles through,
rather than overlapping device work enqueued by the previous step? We cannot use
GPU-side tracers. Give a concrete experiment and the result pattern that would
settle it either way.

### Q3 Semantics traps in the fast path
Enumerate what could make `tensor[cpu_mask_all_true]` -> `tensor[:num_spec_decodes]`
not identical in this code path. Consider: CUDA graph capture-time builds
(`build_for_cudagraph_capture`) and whether replay reuses static metadata buffers;
padded or zero-length rows; ordering guarantees of `masked_select` versus a slice;
whether the kernel also consumes `spec_sequence_masks` as a GPU tensor and would
disagree if shapes changed; and whether `num_spec_decodes` derived from
`.sum().item()` can differ from `num_reqs` in steady state. What must we assert?

### Q4 Is one Triton launch actually cheaper here?
Quantify your expectation for A given: 5 torch ops versus 1 Triton kernel; our
measured Triton dispatch of ~1.14-1.44 ms per step over ~14 launches; the known
tensor-descriptor overhead in the tensordesc path. What would you measure to decide,
and what result would kill A?

### Q5 Re-absorption
Both engines pay the same `masked_select` sync. If we remove it, what stops the
recovered time from being re-absorbed by other host work (PLE row assembly, the
scheduler, Triton dispatch)? What observable would distinguish "the gap closed" from
"the gap moved elsewhere"?

### Q6 Portfolio ranking
Rank by expected value per unit cost, given everything above: (a) this metadata-build
attack; (b) GDN attention kernel work, 54% of forward device time; (c) the drafter,
2.36 ms; (d) acceptance length 2.35 -> 2.75, which round 7 said is worth +17% and
needs an engine-level drafting change; (e) anything not on our list. Justify in
numbers.

### Q7 Decision matrix
Give us a matrix for A and B: for each combination of observed throughput and observed
cadence/idle-gap changes, what does it prove or disprove about the model-side build
being the pacer? We want it before we run, so we do not misread the results.

## 5. Appendix: how the numbers were produced

`py-spy record --idle --format raw --duration 60 --pid <EngineCore>` during a fixed
two-pass load; analyzers tally step-thread samples by subsystem, call site and the
parent of every `wait` leaf. Step windows come from the engine's own log line
`Step timing window (full, tokens=3, n=199, wall ...)`, converted into period, device
span and idle gap. The build timer logs every 200 calls with p50/p95, a batch-size
split, a per-builder-class split and the caller file and line, and is compiled into
the deployed copy behind `VLLM_ATTN_BUILD_TIMING=1`, off by default. Production metric
is a fixed 60-request code-weighted table against a pinned KV cache of 14 GiB with
MTP x2, `max_model_len=4096`, `max_num_seqs=4`, full CUDA graphs, and all reads named
m1/m2/m3 with warm-up excluded.
