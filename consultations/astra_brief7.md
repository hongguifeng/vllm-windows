# Round 7: where does the ~20 ms GPU step go, and what is the cheapest honest way to attribute it

Read-only reviewer. Rounds 1-6 are settled; do not re-litigate them. Answer in
English, concretely and quantitatively, ranked by expected value per unit cost.

## What is now established (no need to re-derive)

1. **The PLE host path is not on the critical path.** From one run's decode
   windows (`tokens=5`, n=199 per 200-step window, 258 windows, MTP x4, fixed
   code-weighted table): row-assembly p50 spans **220 us to 8439 us (38x)** while
   step length p50 spans only **19945 us to 20560 us (+/-1.5%)**, and
   **r(rows, step) = -0.676**. ids wait p50 11.2-14.1 ms, **r(ids, step) = +0.001**.
   Windows where the host spent 8.4 ms assembling rows ran 20.0 ms steps; windows
   where it spent 0.22 ms ran 20.3 ms steps. The x4096-MiB cache arm (rows 6.0-6.5
   ms/step) also did not move step length. End-to-end corroboration: MTP x4 removes
   about one third of SSD lookups and moved the mixed-traffic median by +0.3%.
2. **The step is paced by the GPU**, one step behind: cross-step interval
   (previous consumer mark -> this step's ids landed) p50 = 20.0 ms at x4, and at
   x2 the ids producer queues behind ~16 ms of the previous step's GPU work.
   GPU utilisation is ~40% in these regimes.
3. **Enqueue cadence, not inherent latency, prices the ids wait**: FULL graph
   replay enqueues the whole step at once, so the host runs one step ahead of the
   GPU; eager enqueues kernel-by-kernel and the same `wait_stream` wait costs
   250 us instead of 10393 us (but eager throughput is 17.3 vs 147 tok/s, 8.5x).
4. **PLE GPU work is on the critical path**: the PLE-free structure view at MTP x2
   gave +10.4% with matched acceptance (it also changed weights, memory and graph
   state, so it is a net-difference screen, not a ceiling).
5. **The mixed-traffic gap is acceptance, not lifecycle.** Engine idle share across
   three campaign logs: 0-0.7%; no interval was prefill-only; generation throughput
   while running p50 116-118 tok/s. Same x2 depth: mean acceptance length 2.31-2.40
   on the fixed table vs 2.68-2.83 on repetitive synthetic traffic. Implied step:
   19.7 ms vs 18.7 ms. Per-position acceptance on the table at x4:
   0.86-0.89 / 0.64-0.73 / 0.45-0.65 / 0.36-0.58; drafted throughput 163.6 tok/s vs
   accepted 56.4 tok/s. So the extra draft passes at x4 buy about what they cost.
6. Everything today was measured under **async scheduling with the batch-queue step
   path** (the `block` stat exists only there), even though the serve script never
   passed `--async-scheduling`.
7. Funding rules we already accepted: below ~2% saved wall we do not fund host state
   mirroring; a matched intervention inside 3-5% kills an expensive PLE rebuild. The
   availability witness (non-intervening, presence-and-time only) returned
   `drafts_on_host=False` before dispatch with async scheduling on: this step's real
   draft values do not exist on the host before the step runs.

## The one unmeasured term

Roughly 90% of the ~20 ms GPU step has no kernel-level attribution. We know ~10% is
the PLE layer's GPU work (from the structure view). The whole day was spent on host
timing; the item that paces the step is the GPU. Acceptance 2.35 -> 2.75 on this table
would be +17%, which is larger than any lever we have measured, but it needs an
engine-level drafting change.

## Questions

### Q1 Cheapest attribution of one decode step's GPU time

Rank these by expected value per unit cost, and propose anything better:

- (a) torch/CUDA profiler via `--profiler-config` with `/start_profile` and
  `/stop_profile` over a short decode window (~200-400 steps), then aggregate kernel
  durations from the chrome trace.
- (b) CUPTI kernel activity collected out of band, if available in this build.
- (c) cudaEvent pairs wrapped around the FULL graph replay to time the replay itself,
  plus the existing in-graph marks (we already have 8 preallocated events per step on
  the PLE path; overhead measured at ~0).
- (d) ablation by configuration: run with parts of the step disabled (for example
  `--language-model-only`, or dropping the drafter) and diff step length.

Specifically: **what happens to a FULL CUDA graph replay under a torch/CUDA profiler in
this stack?** Does profiling serialize or distort graph replay, does it attribute time
per kernel correctly inside a replayed graph, and is there a mode that keeps replay
timing honest (for example profiling without CUPTI kernel tracing, or an nsys-style
external trace)? If a profiler arm is unreliable here, say so and name the fallback.

### Q2 What to measure first, given the pacing structure

The host is a full step ahead and its PLE work is hidden. Under that constraint, what
is the cheapest experiment that tells us whether the 20 ms is
(i) serialised GPU work that could overlap if enqueued differently, or
(ii) irreducible kernel time?
Is there a configuration that enqueues a FULL-graph step in pieces without falling back
to eager (PIECEWISE, or breaking the graph at a chosen layer), and would that be worth
testing for step length rather than for the ids wait? Note we already measured that a
graph-mode A/B existed earlier; we did not test step length against enqueue cadence.

### Q3 Decision matrix for the attribution arm

Give a matrix: what result funds which work. In particular, what fraction of step time
in a single kernel (or kernel group) would justify attacking it, given that at x4 a step
is 22.7 ms on repetitive traffic and ~20 ms on the fixed table, and our funding bar is
2% / 5% / 10% saved wall.

### Q4 The drafter

The largest known lever is acceptance length (2.35 -> 2.75 = +17% on this table). Within
this codebase, what is the cheapest drafter change with a plausible chance of moving
position-2 and position-3 acceptance (they are 0.70 and 0.55 on the table)? Is there any
existing per-request or per-decode-step hook for speculative depth, so that routing depth
by content is even implementable? A +2.6% routed gain (computed from class token shares)
is currently not worth engine plumbing; what would make it worth it?

### Q5 Guardrails we must respect

- GPU1 only; never GPU0 (the user's own WSL2 engine holds 62,277 MiB there).
- Warm-up discipline: `lazy` load pushes a 143 GiB page-in to first use; a warm-up block
  must be discarded, and two consecutive blocks must agree within 2% with no
  `load_weights` frames in between before any block is counted.
- Host memory must be low pressure when measuring; we now read
  `GlobalMemoryStatusEx` including `avail_pagefile` before every block.
- Measurement blocks run in the foreground (background children die with the tool
  command).
- Single-stream accounting only; do not propose 16-sequence concurrency experiments.
- Cross-restart generation is not identical at temperature 0 (MoE/atomic reductions),
  so cross-arm comparisons are aggregate only, never per request.

Give the shortest sequence of launches and blocks that answers Q1 and Q2 together, with
a per-launch cost estimate against our observed ~5 minute startup plus ~190 s per
measurement block.
