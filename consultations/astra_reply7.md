The cheapest honest path is:

1. Use CUDA events to establish the GPU span of the existing FULL replay.
2. Use an external Nsight Systems trace with CUDA graph node tracing for kernel attribution.
3. Use PyTorch Profiler only as a secondary diagnostic if Nsight is unavailable.
4. Do not spend time trying to make a FULL graph replay “piecewise” through the current configuration; vLLM does not expose that as a mode.

The existing timing code is already close to the needed measurement. [`StepTimingCollector`](D:/code/vllm-windows/vllm/v1/worker/gpu/async_utils.py:41) records GPU events around the model forward and drafter without synchronizing every step.

**Q1. Attribution ranking**

| Rank | Method | Value | Cost and limitation |
|---|---|---|---|
| 1 | Existing CUDA events, extended with an outer pair around FULL replay and a few in-graph phase pairs | Establishes the true GPU span with negligible distortion | Does not name individual kernels unless phase markers are added |
| 2 | Nsight Systems with `--cuda-graph-trace=node` | Best first kernel-level attribution for this stack | Requires one profiled launch and external Nsight tooling |
| 3 | Existing CUPTI activity collection, if already working | Can provide similar kernel activity with little application change | Integration and CUDA-version behavior are uncertain; validate against event timing |
| 4 | PyTorch Profiler with CUDA activity | Easy to enable and useful for kernel-name histograms and CPU/GPU chronology | Medium to high overhead; CUDA graph attribution can be version-sensitive |
| 5 | Configuration ablations | Useful for net-difference screening | Changes graph state, fusion, memory pressure, acceptance, and sometimes batch shape; weak for attributing the missing 90% |

The better version of option (c) is to keep the outer event pair and add a small number of event pairs inside the captured graph:

- target-model start/end;
- attention or KV-cache region;
- dense projection/MLP region;
- MoE dispatch and expert computation;
- sampler;
- drafter;
- any collective or explicit synchronization region.

The event records must be captured into the graph and reused on every replay. Read the elapsed times only after the block’s single final synchronization. This preserves the current single-stream accounting and avoids a per-step host synchronization.

For a FULL replay, the current runner explicitly calls `run_fullgraph()` for `CUDAGraphMode.FULL`; it does not invoke the model kernel-by-kernel from the host. [`gpu/model_runner.py`](D:/code/vllm-windows/vllm/v1/worker/gpu/model_runner.py:1734) shows that distinction.

With the current PyTorch profiler arm:

- vLLM starts `torch.profiler` with both CPU and CUDA activities hardcoded in [`gpu_worker.py`](D:/code/vllm-windows/vllm/v1/worker/gpu_worker.py:1233).
- A graph captured during startup remains a captured graph when profiling starts later through `/start_profile`.
- Profiling does not turn `cudaGraphLaunch` into a sequence of ordinary host launches. The replay remains one graph launch, and the graph’s kernels execute according to the captured graph.
- Kineto/CUPTI can generally report the constituent GPU kernels launched by a replay. They appear on the graph’s execution stream, rather than as ordinary host-dispatched operations.
- Kernel names and aggregate device durations are useful for ranking candidates, but graph-node attribution and graph timing have had CUDA/CUPTI-version-specific correctness issues. Treat the trace as an attribution screen until its kernel sum agrees with CUDA-event replay timing.
- CUPTI callbacks and profiler bookkeeping can perturb host enqueue time and sometimes GPU timing. The graph is not logically serialized into eager execution, but the profiled block is not a clean throughput benchmark.
- `with_stack=True`, shape recording, memory recording, and FLOP recording add additional overhead. Disable all of them for this experiment.
- `capture_torch_profiler=True` is the wrong arm for timing: it profiles graph capture at startup. Leave it disabled.

There is no current vLLM configuration switch for “PyTorch profiler with CPU activity but no CUDA activity”; the worker constructs the profiler with `["CPU", "CUDA"]`. A CPU-only profiler would avoid CUPTI kernel tracing but would not answer the kernel-attribution question anyway.

The preferred external arm is the CUDA profiler wrapper plus Nsight Systems:

```text
CUDA_VISIBLE_DEVICES=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  --output=full_gpu1 \
  vllm serve ... \
  --profiler-config.profiler=cuda
```

`CUDA_VISIBLE_DEVICES=1` makes physical GPU1 the only visible GPU. Verify the physical mapping before starting; do not rely on the process-local logical index alone.

This is the fallback if PyTorch Profiler gives any of the following:

- graph replay appears as only one unhelpful event;
- kernel totals disagree materially with CUDA-event replay time;
- replay timing changes by more than roughly 2–3%;
- repeated replays show implausible graph-node durations.

The vLLM profiling guide recommends this Nsight mode and specifically calls out `--cuda-graph-trace=node`: [`docs/contributing/profiling.md`](D:/code/vllm-windows/docs/contributing/profiling.md:170).

**Q2. What to measure first**

The cheapest decisive experiment is not FULL versus eager. It is:

1. Measure the GPU event span from immediately before `run_fullgraph()` to immediately after it.
2. Trace a short sequence of the same FULL replays with Nsight graph-node tracing.
3. Compare the event span with the kernel timeline and the gaps between kernels.

Interpretation:

| Observation | Interpretation |
|---|---|
| Kernel critical-path time nearly equals the event span, with no material gaps | The approximately 20 ms is predominantly irreducible work in the current graph |
| Kernel durations sum to materially less than the event span, with idle gaps or synchronization gaps | There is a scheduling, dependency, collective, or synchronization opportunity |
| One phase marker is much longer than its expected kernel sum | Inspect host/device synchronization, graph dependencies, or activity attribution before optimizing kernels |
| FULL replay span is stable, but eager only reduces the ids wait | The previous result is confirmed: eager changes enqueue cadence and host visibility, not GPU step cost |

Under the stated single-stream constraint, kernels issued to that stream cannot overlap merely because the host enqueues them earlier. FULL graph replay already submits the captured work together. The observed eager improvement in the ids wait is therefore a host/GPU pipeline effect; it is not evidence that the GPU’s 20 ms can be shortened by changing the host enqueue order.

There is no supported mode that takes one FULL graph and replays it in user-selected pieces. The available modes mean:

- `FULL`: one top-level graph replay;
- `PIECEWISE`: separate compiled or breakable graph segments, with eager operations between segments where required;
- `FULL_AND_PIECEWISE`: dispatches FULL and PIECEWISE according to batch type.

The current Inductor graph-partition option applies to piecewise execution. The FULL wrapper remains outside the partitioned function and captures the whole call. The breakable CUDA graph implementation also bypasses eager breaks in FULL mode; its breaks are a PIECEWISE mechanism. See [`compilation/breakable_cudagraph.py`](D:/code/vllm-windows/vllm/compilation/breakable_cudagraph.py:5) and [`CompilationConfig`](D:/code/vllm-windows/vllm/config/compilation.py:665).

A forced `PIECEWISE` arm is still a possible net-performance screen, but it is not a clean enqueue-cadence experiment. It changes:

- graph boundaries;
- compiler partitioning and fusion;
- attention execution;
- capture memory;
- host launch overhead;
- possibly the drafter’s execution mode.

For some multi-module MTP paths, piecewise support is explicitly incomplete. A result from this arm should be labeled “different execution plan,” not “FULL replay split into pieces.”

I would only run PIECEWISE after the FULL event-plus-Nsight arm, and only if the team is willing to consider changing the execution mode. It is not needed to decide whether the current FULL graph contains 20 ms of actual GPU work.

**Q3. Funding matrix**

Let:

- \(T\) be the measured GPU step span;
- \(f\) be the fraction of the critical-path span occupied by a kernel or kernel group;
- \(r\) be the fraction of that group’s time that a proposed optimization can remove.

The expected wall saving is approximately:

\[
\text{saving} \approx f \times r
\]

Use the critical-path group time, not a raw sum of overlapping activity.

For the two relevant step lengths:

| Funding bar | Fixed table, 20.0 ms | Repetitive traffic, 22.7 ms |
|---|---:|---:|
| 2% | 0.40 ms | 0.45 ms |
| 5% | 1.00 ms | 1.14 ms |
| 10% | 2.00 ms | 2.27 ms |

A practical decision matrix is:

| Measured critical-path group | Work to fund |
|---|---|
| Under 2%: under 0.40–0.45 ms | Do not fund a dedicated optimization |
| 2–5%: 0.40–1.14 ms | Fund only a cheap local change with a credible near-total removal or at least a 40–100% reduction |
| 5–10%: 1.0–2.27 ms | Fund targeted kernel investigation if a 50% reduction is plausible; this can clear the 5% bar |
| 10–20%: 2.0–4.5 ms | Fund serious optimization work; a 25–50% reduction clears the 5% bar |
| Over 20% | Fund unless the group is already known to be unavoidable or the optimization has unusually high engineering risk |

Examples:

- A 5% group must be almost completely removed to meet the 5% wall-saving bar.
- A 10% group needs roughly a 50% speedup to save 5% wall time.
- A 20% group needs only a 25% speedup to save 5%.
- To justify a 10% target from a 10% group requires near-total removal. That is a high bar.

Because the fixed-table step is approximately 20 ms, a measured 0.8 ms group is a 4% ceiling even if it could be eliminated perfectly. It cannot justify a 10% optimization project.

The 22.7 ms repetitive-traffic arm should be reported separately. A kernel may be large there because the accepted workload causes a different shape or because extra draft verification is being performed. It should not be treated as evidence that the same kernel is a 10% opportunity on the fixed table.

**Q4. The drafter**

There is no cheap runtime setting in this checkout that is likely to improve the quality of MTP position 2 or position 3 predictions.

The cheapest useful runtime experiment is to test different fixed depths, for example MTP x2, x3, and x4, and record:

- step GPU span;
- mean acceptance length;
- per-position acceptance;
- accepted tokens per second.

That tells you whether the low 0.70 and 0.55 survival at positions 2 and 3 makes those passes worthwhile. It does not improve their predictions; it only avoids paying for them when they are not profitable.

The existing dynamic speculative-depth support is batch-wide. The scheduler builds a lookup from `num_speculative_tokens_per_batch_size` and chooses K from the current number of scheduled requests:

```text
K = dynamic_sd_lookup[len(num_scheduled_tokens)]
```

That logic is in [`scheduler.py`](D:/code/vllm-windows/vllm/v1/core/sched/scheduler.py:1319). It is useful for a first no-plumbing experiment, but it cannot distinguish two requests in the same decode batch based on their content.

The current request object exposes acceptance metrics for observation, but those metrics do not feed a per-request K decision. There is no general per-request speculative-depth field in `SamplingParams` or `Request`. The available content-sensitive mechanisms are separate features:

- suffix decoding can choose a content-dependent speculative length;
- DSpark has a confidence-head-based adaptive verification path;
- the generic MTP path uses the configured depth and scheduler-level dynamic depth.

Neither provides a generic content-aware routing hook for native MTP.

The cheapest change with a plausible chance of moving positions 2 and 3 is therefore a better trained or better matched drafter:

- an available compatible EAGLE/EAGLE3 or draft-model checkpoint;
- a custom proposer through the existing `custom_class` mechanism;
- retraining or fine-tuning the MTP/speculator heads if model ownership permits it.

Changing `draft_sample_method`, rejection sampling, or graph mode will not make the underlying position-2 or position-3 logits more accurate. Changing K only truncates the draft.

Content-routed K is worth engine plumbing when a measured aggregate gain clears the funding bar:

- below 2%: reject;
- 2–5%: only accept if the implementation is nearly free and low risk;
- at least 5%: reasonable target for scheduler plumbing;
- around 10% or more: strong justification for per-request routing and graph-shape support.

The present 2.6% routed estimate is below the accepted funding threshold for new engine machinery. It becomes attractive if a real aggregate experiment shows at least 5% accepted-token throughput improvement on the fixed table, with stable acceptance and no more than a 2% step or lifecycle penalty. A 10% result would justify the added graph and scheduler complexity much more comfortably.

**Q5. Shortest measurement sequence**

I would use two launches, with an optional third only if the PIECEWISE net screen is desired.

| Launch | Configuration | Blocks | Estimated time |
|---|---|---:|---:|
| A | Existing FULL configuration, no profiler, CUDA-event timing | 1 discarded warm-up + 2 agreeing blocks | \(300 + 3×190 = 870\) s, about 14.5 min |
| B | Same FULL configuration, external Nsight Systems graph-node trace | 1 discarded warm-up + 2 agreeing blocks; trace only 20–40 steady decode steps | About 14.5 min, plus trace finalization |
| C, optional | Force `PIECEWISE`, same workload and event timing | 1 discarded warm-up + 2 agreeing blocks | About 14.5 min |

For every launch:

- expose physical GPU1 only;
- check `GlobalMemoryStatusEx`, including available pagefile, before every block;
- run blocks in the foreground;
- discard the first block because of the lazy 143 GiB page-in;
- count a block only after two consecutive blocks agree within 2%;
- confirm there are no `load_weights` frames between counted blocks;
- compare arms only as aggregate distributions across restarts.

Launch A is the timing authority. Launch B supplies the kernel breakdown. Do not use B’s profiled wall time as the baseline because the profiler can perturb it; instead, compare its CUDA-event replay spans against A.

If Nsight is unavailable, replace Launch B with PyTorch Profiler, using a short scheduled capture rather than 200–400 continuously active steps:

```json
{
  "profiler": "torch",
  "torch_profiler_dir": "/absolute/path/profile",
  "torch_profiler_with_stack": false,
  "torch_profiler_record_shapes": false,
  "torch_profiler_with_memory": false,
  "torch_profiler_with_flops": false,
  "ignore_frontend": true,
  "wait_iterations": 2,
  "warmup_iterations": 3,
  "active_iterations": 20
}
```

Use 20–40 active iterations first. At a 20 ms step, that gives 0.4–0.8 seconds of repeated GPU activity, usually enough for a kernel histogram while keeping the trace and flush cost manageable. Extend to 100–200 active iterations only if the workload has many shape classes or the first trace does not stabilize.

The decisive output from the first two launches is:

- FULL replay CUDA-event span;
- sum and critical path of traced kernels;
- time in explicit gaps or synchronization regions;
- PLE group time;
- drafter group time;
- whether any remaining group exceeds the 0.40 ms, 1.00 ms, or 2.00 ms funding thresholds.

That will attribute the missing step time with the lowest cost that remains quantitatively defensible.
