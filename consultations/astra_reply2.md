## Executive judgment

**Do not enable true overlapped I/O for performance yet. Keep prepared launches disabled. Prioritize validating the MTP=2 default, then measuring the PLE opportunity without mistaking blocked time for removable time.**

Your new evidence materially changes my round-1 assessment:

- **The EOF change is conservative hardening, not a demonstrated correction of wrong output.** My earlier assertion that an unbuffered request crossing EOF necessarily fails was too strong.
- **Synchronous handles with a non-null `OVERLAPPED` are a documented combination.** The absence of `FILE_FLAG_OVERLAPPED` establishes synchronous execution; it does not, by itself, establish invalid API usage.
- **The cancellation lifetime risk becomes substantially more important if you enable actual asynchronous reads.** The existing synchronous tests do not exercise that risk.
- **The explicit strict-mode mismatch fallback is fixed in the source I inspected.** One remaining qualification: `_strict_check()` still returns `True` on an exception or `reference is None`. That is “fallback on detected mismatch,” not fully fail-closed verification. Since prepared launches have no demonstrated benefit, leave them disabled rather than spending more performance effort there.

I read the updated reader, PLE implementation, prepared-launch implementation, graph-break implementation, and round-1 reply. I also checked Microsoft’s API documentation. **I made no changes, ran no probes or benchmarks, and did not access either GPU.** Your test counts and performance measurements below remain reported results, not independently rerun results.

---

## Q4.1 — Overlapped I/O: not yet

### The benefit is small enough to require a stronger justification

The entire measured row-retrieval interval is **0.38–0.95 ms/step**, including work that asynchronous disk I/O cannot eliminate.

For an approximately 19 ms engine step, eliminating that interval completely would improve throughput by approximately:

\[
\frac{19}{19-0.38}-1=2.0\%,\qquad
\frac{19}{19-0.95}-1=5.3\%.
\]

Those are optimistic bounds **if the entire interval is exposed on the critical path**. Actual overlapped-I/O savings should be smaller.

Also, worker count 8 → 16 hurting disk rate supports “more concurrency is not automatically better”; it does not independently establish the cause of that regression.

### There is more documentation support for the existing combination than I credited

Microsoft’s [`ReadFile` documentation](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile) explicitly describes synchronous handles with non-null `lpOverlapped`: the offset is taken from the structure and the call does not return until the operation completes.

The [`OVERLAPPED.hEvent` documentation](https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-overlapped) states:

> Functions such as ReadFile and WriteFile set this handle to the nonsignaled state before they begin an I/O operation. When the operation has completed, the handle is set to the signaled state.

That is a reasonable documentary basis for the event behavior you observed; it is not just an unexplained machine-specific accident. The `ReadFile` page’s most detailed event discussion is under asynchronous handles, so I would still favor a clearer synchronous implementation when next touching this area:

**For a deliberately synchronous backend, collect the byte count directly from `ReadFile` and process successful completion directly, rather than using the asynchronous harvest machinery.** This preserves Python-worker parallelism without introducing pending operations.

Two important corrections:

1. `GetLastError()` after successful `ReadFile` is not a meaningful success diagnostic. Record it only on failure.
2. The current `WAIT_MS=60000` **does not bound time spent inside synchronous `ReadFile`**. A stalled synchronous call has not reached your harvest timeout. Thus you do not currently have a reliable 60-second I/O deadline to preserve.

### EOF interpretation

`ReadFile` documents successful partial reads when a request starts before EOF and extends beyond it. Also, your reader accepts `got >= delta + row_bytes`, not necessarily `got == rounded_request`.

Therefore, correct returned rows alone do **not** prove that every rounded request returned its full requested byte count. Instrument `got` if that distinction matters.

Keep the new buffered-boundary policy, but retire the categorical comment that an unbuffered handle “cannot” deliver such a read. Classify the fix as **making the boundary policy explicit and conservative**, not evidence of previously corrupted rows.

### Minimum gate before a future asynchronous flip

The lifetime fix must come **before**, not after, enabling `FILE_FLAG_OVERLAPPED`.

| Gate | Required evidence |
|---|---|
| Ownership contract | One owner per reader; no concurrent `rows_read`/`rows_close`; caller output remains owned until every issued operation finishes |
| Completion state machine | Distinguish immediate success, pending, and immediate failure; do not wait for an operation that was never issued |
| Events | Prefer manual-reset events; explicitly retire/reset completed slots so inactive signaled events cannot dominate the wait loop |
| Normal operation | Both immediate and genuinely pending completions observed; exact rows and completion byte counts checked; slot reuse and mixed buffered/direct requests exercised |
| Cancellation | Stop submissions, cancel outstanding operations, then establish terminal completion for each issued operation |
| Failure return | Drain **before `rows_read` returns**, not merely before `rows_close` |
| Stress | Approximately 1,000 bounded cancel/complete/close cycles, with counts proving genuinely pending operations were exercised |

The **failure-return condition is essential**: buffered exact reads target the caller’s output directly. Returning an error while such a read is pending lets Python release or reuse that output before `rows_close` happens.

[`CancelIoEx`](https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-cancelioex) explicitly does not wait for completion. Cancellation may end in success, `ERROR_OPERATION_ABORTED`, or another terminal error. `ERROR_NOT_FOUND` from cancellation is not itself a completion-status check.

Do not implement “drain” as “wait another bounded interval, then free anyway.” If terminal completion cannot be established, the referenced memory must remain alive; cancellation does not guarantee a bounded teardown time.

### Is Application Verifier necessary?

**Not a prerequisite for retaining the synchronous implementation.** Assuming shutdown joins users of the reader, completed synchronous calls do not leave the pending-I/O lifetime problem described above.

For true async, I would require the reviewed lifetime invariant plus targeted pending/cancellation tests. Application Verifier is valuable supplementary evidence, not a substitute, and I would not install or configure it on the running host merely to unblock this decision.

The 500 clean cycles are useful regression evidence. They are not pending-cancellation coverage. Even under an ideal independent-trial model, zero failures in 500 trials only puts an approximate 95% upper bound of **0.6% per tested cycle** on failure probability.

**Decision: retain synchronous operation; make its semantics explicit later. Revisit async only if new traffic exposes materially more row-I/O time or operational requirements demand asynchronous cancellation.**

---

## Q4.2 — Dispatch perturbation: three conditions are sufficient, but placement matters

### Separate two different experiments

| Injection | What it measures |
|---|---|
| Delay immediately before actual eager kernel submission inside normal Triton dispatch | Sensitivity to additional dispatch latency at those launch sites |
| One delay after `execute_model` submission | End-of-submit / next-step scheduling slack |
| GPU delay on a dependency stream | Sensitivity to extending that GPU dependency |

These are related controls, not interchangeable measurements.

**One `time.sleep` at the end of submit is acceptable as a scheduling-slack experiment. It is not a clean dispatch experiment.** It also releases the GIL and may allow the PLE coordinator to progress, unlike extra Python argument-binding work.

### Where I would inject

For the dispatch-specific experiment:

- Use the **normal Triton path**, not prepared launches.
- Place the hook immediately before its actual eager compiled-kernel launch, after dispatch/binding work.
- Enable it only for marked steady decode steps.
- Exclude warmup, capture, compilation, and `warmup=True` calls.
- Do not wrap `CUDAGraph.replay`, modify captured kernels, or add graph breaks.

Your `BreakableCUDAGraphCapture.replay()` executes a sequence of graph replays and eager callables. The hook should affect only real Triton launches made by those eager callables.

**Specify the budget per engine step, but distribute it across eligible eager launches.** Use the measured eligible-launch count for each MTP depth, and log the actual aggregate injected duration. If there are no eligible launches, that step receives no dispatch perturbation—do not silently move the delay elsewhere.

For a GIL-holding dispatch approximation, use short calibrated native delays that retain the GIL. Avoid a Python busy loop whose interpreter scheduling introduces another variable. This approximates extra GIL-held dispatch work; it cannot reproduce every detail of Python scheduling. A GIL-releasing sleep is a separately labeled arm.

### Conditions, repeats, and slope

**Accept 0, +1, +3 ms/step at both MTP depths. Five delay values are unnecessary initially.**

Define:

\[
s=\frac{\Delta\text{steady wall time per engine step}}
        {\Delta\text{actual injected delay per step}}.
\]

Use engine-step wall time, not submit p50 alone; also record emitted tokens per step.

| Result | Interpretation |
|---|---|
| Upper 95% confidence bound on \(s\) below 0.1 | Operationally flat over this tested range |
| Lower bound above 0.3 | Material sensitivity |
| Lower bound above 0.7 | Most added delay is exposed |
| Confidence interval spans flat and material regions | Inconclusive |
| Flat at +1 ms, material increase at +3 ms | A slack threshold, not one global linear relationship |

At approximately 19 ms/step, a slope of 0.3 produces **0.9 ms extra time at +3 ms**, or roughly a 4.5% throughput loss—comfortably larger than ±1% noise. Establishing “flat below slope 0.1” is harder and needs an equivalence-style confidence bound, not merely a nonsignificant test.

Use:

- **Three paired blocks** per condition for screening.
- **Five paired blocks** for a decision if results are near a boundary.
- Approximately **500–1,000 steady steps per condition per block**, with matched prompt groups and balanced condition ordering.

Treat blocks/requests, not individual autocorrelated steps, as the replication units.

If a future test harness can select the delay by block without recapture, this requires **one planned initialization per MTP depth**, rather than six or ten startups. This is a proposal for a later experiment, not a request to reconfigure the live service.

### GPU-side control

**Optional, not required for the first pass.**

`torch.cuda._sleep` is a busy GPU kernel, not an abstract delay. It may interfere with concurrent kernels, consumes device resources, and its cycle count is not a portable wall-time specification.

A GPU1-only, event-calibrated delay on the relevant dependency stream can show that the measurement responds to an extended GPU dependency. It cannot cleanly separate “CPU slack” from “GPU slack” by itself.

Finally, a positive-delay experiment is asymmetric: **sensitivity to adding 3 ms does not prove that removing today’s dispatch overhead would recover the corresponding time.**

---

## Q4.3 — Event-scoped wait: the inspected code makes the immediate opportunity look small

This is my strongest source-based revision to the optimization plan.

The current sequence is:

```python
self._stream.wait_event(self._previous_use)
self._device_ids[:tokens].copy_(ngram_ids)  # current/main stream
self._stream.wait_stream(torch.cuda.current_stream())
```

The main-stream preservation copy is immediately followed by `wait_stream`.

**That wait covers work already submitted at that point; it does not automatically wait for later main-stream submissions.** In the inspected path, replacing it with an event recorded immediately after the preservation copy expresses essentially the same dependency.

### (a) What is semantically sufficient?

For the current staging design, the side-stream D2H copy must wait for:

1. Production of `ngram_ids`.
2. The main-stream copy into stable `_device_ids`.
3. Relevant prior-use constraints on reusable staging buffers.

Thus **an event after the IDs producer alone is insufficient if the side stream still reads `_device_ids` before its preservation copy finishes**. Record after preservation.

Copying directly from `ngram_ids` on the side stream is a different design. It must prevent graph-pool reuse until that copy finishes. The source explicitly explains why the preservation copy exists; do not remove that protection.

The existing `_previous_use` and `_copy_ready` constraints also need to remain, unless independently replaced with equivalent lifetime ordering.

### (b) Expected saving

**My prior for simply substituting the event is approximately zero, not 9 ms.**

A real opportunity exists if the producer is early inside a graph segment and unrelated kernels remain between that producer and the point where prefetch can start. But exploiting that requires more than replacing the existing wait:

- An appropriately placed event inside the captured work, plus safe data preservation; or
- Different segmentation/prefetch placement, with its own replay and lifetime costs.

Because MTP produces inputs needed for these IDs, much of the wait may be an unavoidable upstream dependency.

### (c) Measurement

On a later instrumented run, mark:

- Producer completion.
- Stable-ID preservation completion.
- Side-stream availability after previous use.
- D2H start/end.
- Host row-read start/end.
- H2D completion.
- Consumer arrival and actual start.

Use preallocated timing-event rings and collect completed records later. **Do not add per-step synchronization to read the timings.** Record corresponding host timestamps; use a profiler if cross-domain clock correlation is needed.

First compare instrumented versus uninstrumented execution to bound instrumentation overhead.

### (d) Decision matrix

| Observation | Conclusion / next step |
|---|---|
| Producer → preservation → D2H is already tight | Drop the event substitution as a performance project |
| Large unrelated-work interval after producer | Investigate earlier safe preservation/prefetch, not merely a different wait API |
| Previous-use ordering dominates side-stream availability | Consider staging-buffer overlap, with explicit lifetime accounting |
| Producer itself finishes late | Late IDs are mostly upstream compute dependency; event substitution cannot remove it |
| IDs become available earlier, but consumer timing is unchanged | Work was already hidden; no throughput benefit |
| Consumer advances by ≥0.5 ms/step, repeatably | Approximately ≥2.5% opportunity on a 19 ms step; worth a small implementation |

---

## Q4.4 — PLE ablation: none of the three alone prices the IDs dependency

| Ablation | Preserves model outputs? | Retains IDs dependency? | Valid use |
|---|---:|---:|---|
| Trimmed PLE-free view | No, generally | Usually not | Exploratory net comparison, not a strict ceiling |
| Full weights, exact prewarmed host rows | Yes | Yes | Measures avoidable SSD misses |
| Full weights, zero replacement rows | No | Depends on implementation | Structural timing experiment with controlled token/shape trace |

**Option (ii) is least confounded for measuring the SSD contribution. It does not isolate the IDs dependency.**

You do not need 95 GiB for that test. Record the unique rows needed by a bounded representative window and determine whether an exact replay cache fits host memory. Report its actual coverage; do not label a 90% hit run “SSD-free.”

For the IDs dependency, the cleaner experiment is a **record/replay intervention**:

1. Record a baseline MTP=2 window: row IDs, step shapes, emitted/draft token behavior, and lookup ordering.
2. Replay the same computational workload.
3. Compare exact rows obtained through the normal path with those same rows made available earlier.
4. Preserve allocations and graph segmentation as far as practical; enumerate residual differences.

That measures the value of earlier row availability on the recorded workload. It is an **oracle experiment**, not evidence that a deployable implementation can know future IDs.

If you use zero rows or trimmed weights, generated tokens and speculative acceptance can change. For timing attribution, freeze/replay the token and shape trace; otherwise the comparison combines implementation changes with a changed workload.

**Accept the cheap MTP=2 trimmed run as a screen, but stop calling its result a PLE-free ceiling. Neither a large nor a small gap alone proves the attainable gain from eliminating the IDs rendezvous.**

---

## Q4.5 — GPU cache: trace first; per-row hit rate is not the deciding metric

### Budget

For single-stream serving, a modest KV trade can be reasonable **if the remaining KV capacity still meets the maximum-context requirement with margin**.

| Row-storage budget | Share of 14 GiB KV | Rows at 320 bytes, before metadata |
|---:|---:|---:|
| 64 MiB | 0.45% | 209,715 |
| 128 MiB | 0.89% | 419,430 |
| 256 MiB | 1.79% | 838,860 |
| 512 MiB | 3.57% | 1,677,721 |

Add tags, indexing, replacement state, miss queues, staging, and allocator overhead. With **63.5 GiB already occupied**, I would not use the nominal remaining 0.5 GiB as a reliable deployment budget.

Start with **64–128 MiB**, funded explicitly, only after trace simulation. Do not change weights to make room.

### The important question: how many lookup batches avoid the host entirely?

With 350–520 rows per step, excellent row-level hit rates can still leave a miss almost every step.

Illustratively, assuming independent row hits:

| Row hit rate | All-hit probability, 350 rows | All-hit probability, 520 rows |
|---:|---:|---:|
| 99% | 3.0% | 0.54% |
| 99.9% | 70.5% | 59.4% |

A 90% all-hit probability requires approximately **99.970–99.980% per-row hits** under that simplified model.

Actual rows are correlated and may repeat, so simulate real lookup batches rather than using this independence calculation. The point is that my round-1 **“>90% row hits” gate was much too weak for removing the host rendezvous**.

### Eviction and misses

For an initial prototype, prefer:

- A trace-selected static hot set, or simple set-associative CLOCK/approximate-LRU.
- No eviction of rows still referenced by in-flight consumers.
- GPU-side hit gathering and compacted miss IDs.
- Asynchronous miss transfer, exact host-cache/SSD fallback, then row installation.
- No global synchronization.

But **no eviction policy guarantees that an unpredictable miss stays off the critical path**. Its latency is hidden only if lookup begins sufficiently ahead of consumption.

Also, if every “all-hit” step still requires the CPU to wait for a miss-count event and authorize the next graph segment, you have retained the very rendezvous you hoped to remove. A useful design must explain its all-hit execution path without hand-waving over that control dependency.

### Trace request

Record **raw IDs with lookup-batch and step boundaries**, not only a recency-stack summary:

- Table/layer identity.
- Row IDs and duplicates.
- Request/prompt class and phase.
- Step/MTP shape.
- Prefetch timing/order.
- Existing cache state or enough history to reconstruct warmup.

Start with **10,000 decode steps across representative traffic**. At 350–520 IDs/step, raw IDs occupy about **14–21 MB at 32 bits**, or **28–42 MB at 64 bits**, excluding metadata.

Simulate 64/128/256/512 MiB and report:

- Row hit rate.
- All-hit batch and step fractions.
- Misses per batch: median/p95/p99.
- Cold-start behavior.
- Transfer bytes and available prefetch lead time.

**Kill the cache project if simulation predicts mostly partial-hit steps and the implementation still pays essentially the same host rendezvous.** Reducing sub-millisecond row work alone probably does not justify it.

---

## Q4.6 — MTP=2 default: retain 10% as a policy threshold, not a universal constant

Prompt labels are not acceptance measurements. Long reasoning, code, and multilingual traffic can all be highly predictable locally.

For a deliberately challenging candidate pool, use:

- Code edits with unfamiliar identifiers, irregular literals, and alternatives sharing long prefixes.
- Tool/JSON outputs where supplied context selects among close alternatives.
- Exact quotation/extraction of irregular user-supplied strings or tables.
- Mixed-script/code-switching tasks, transliteration, and less-common languages relevant to actual traffic.
- Mathematical or symbolic continuations with frequent locally competing choices.
- Production-temperature creative continuations with many plausible next tokens.

Include ordinary prose, common code, and repetitive structured outputs for medium/high acceptance.

**Select low acceptance empirically on a calibration set, then evaluate on held-out prompts.** Do not manufacture the low class merely by changing temperature away from production settings.

### Default gate

I would use:

- **Mixed-set decode gain ≥10%.**
- Paired 95% confidence lower bound preferably **>5%**.
- No important class with a repeatable **>5% decode regression**, unless routing addresses it.
- No unacceptable TTFT, tail-latency, memory, or output-quality regression.

A robust 5–10% gain can justify conditional enablement. Below 10% is not “MTP=2 is bad”; it is “the evidence for an unconditional default needs more scrutiny.”

Measure:

\[
R_d=\frac{\sum \text{emitted tokens}_d}{\sum \text{decode time}_d},
\qquad
G=\frac{R_2}{R_1}-1.
\]

Do not average per-request throughput percentages. Weight prompt classes according to intended traffic, and also report each class separately.

For attribution:

\[
R=\frac{\text{emitted tokens per step}}{\text{wall time per step}}.
\]

Acceptance rate alone is insufficient, especially across different draft depths.

### “One run”

One **orchestrated campaign** can do this: a fixed, seeded mixed traffic list, both MTP configurations, paired request groups, and at least three distinct steady blocks. Separate prefill/TTFT/decode.

If depth is startup-fixed, one service initialization cannot compare both depths. Two configuration runs are the minimum; ABBA requires additional configuration instances. Do not claim initialization-level replication from several windows inside one initialization.

---

## Q4.7 — The first two protocol investments

### 1. Paired, balanced replication

Use matched request groups, balanced ordering, and at least three distinct steady blocks. Report median, spread, and paired differences.

This prevents a modest optimization from being “discovered” by favorable run order or cache drift.

### 2. Per-window workload accounting

At minimum:

- Emitted tokens and engine steps.
- Mean emitted tokens per step.
- Prompt class and sampling configuration.
- Decode time separately from prefill/TTFT.
- PLE rows, misses, and hit rate.
- Relevant memory/cache state.

This distinguishes faster execution from more favorable speculative acceptance.

### Is a manifest worth it on one card within one day?

**Yes—but make it minimal and automatic, not a tooling project.**

Record command/config, source revision plus dirty-diff identity, reader DLL hash, model/tokenizer identity, key wheel versions, driver, GPU1 UUID, and prompt/seed configuration. Capture once per configuration; reference it from every window.

Same-day A/B controls many hardware variables, but not accidental DLL changes, environment flags, dirty source changes, or differences between process configurations. Your DLL-provenance investigation already illustrates the value.

Continuous detailed clock telemetry can wait. The small manifest should not.

---

## Q4.8 — Next five actions, ranked by expected value per cost

Costs below are planning estimates for future authorized work, not actions taken here.

| Rank | Action | Expected value | Cost | Result that kills or redirects it |
|---:|---|---|---|---|
| **1** | **Mixed-traffic MTP=1 vs MTP=2 campaign** | Converts the existing approximately 24% traffic-specific gain into a defensible default or routing policy | Low–medium: harness work, configuration startups, several paired traffic blocks | Mixed gain <10% weakens unconditional default; repeatable class regressions imply routing |
| **2** | **MTP=2 PLE ablation screen, explicitly labeled by what it changes** | Establishes whether a large PLE-related opportunity plausibly remains | Low for existing trimmed run; medium for a matched replay | A *matched* intervention within 3–5% kills expensive PLE redesign; an unmatched trimmed result is only suggestive |
| **3** | **IDs/preservation/copy/consumer timeline** | Separates unavoidable producer latency from removable ordering and host turnaround | Approximately 0.5–1 day | Tight producer-to-copy timing and negligible exposed turnaround kill simple event substitution |
| **4** | **Raw row-ID trace and batch-aware cache simulation** | Cheap rejection or sizing of a potentially expensive GPU-cache design | Approximately 0.5 day once logging exists; tens of MB | Low all-hit-batch rate, inadequate lead time, or retained host rendezvous kills the cache |
| **5** | **Three-condition dispatch perturbation** | Closes a remaining causal question; limited immediate upside because prepared launches already lost | Approximately 0.5–1 day for a clean hook plus measurements | Flat response ends dispatch work; material slope justifies locating exposed eager launches, not automatically reviving prepared launches |

### Where I disagree with your ranking

1. **Move diverse-traffic MTP validation above the PLE-free run.** It governs a production decision with an already demonstrated large gain.
2. **Keep the cheap ablation near the top, but do not treat it as a ceiling or a decisive cache gate unless the workload is matched.**
3. **Move trace simulation above dispatch perturbation.** It cheaply rejects a costly proposal; dispatch already has a negative intervention result.
4. **Do not presume the event substitution is an optimization.** In the inspected source it is nearly the same dependency expressed differently.
5. **Keep true async outside the performance top five.** Its plausible benefit is small, while its lifetime and failure-path obligations are materially larger.

**Bottom line:** the highest-value question is no longer whether the port works or whether Triton dispatch can be shortened. It is whether the MTP=2 advantage generalizes, and whether PLE contains an exposed, removable dependency after accounting for the actual ID producer, graph boundaries, and caller-buffer lifetimes.
