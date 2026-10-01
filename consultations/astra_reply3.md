## Executive judgment

**MTP×4 is now the strongest production candidate, but not yet a justified unconditional default. Validate it before funding a PLE redesign.** The depth result is more actionable than the structural ablation: **+17.5% on this workload for a configuration change**, versus an unmatched approximately 10% PLE-related screen.

Three important qualifications:

1. **The IDs wait is a dependency wait, not necessarily removable CPU overhead.** Its duration includes waiting for GPU work. A small final `block` interval does not establish that GPU execution is unimportant.
2. **40% sampled GPU utilization does not establish 60% schedulable spare capacity.** Dependency gaps, serial kernels, and resource contention determine whether useful work can occupy those intervals.
3. **Your current traffic script measures request-level completion throughput, not isolated decode throughput.** In `_dev/probe/_c1_traffic.py`, the denominator includes the entire non-streaming HTTP request, including prefill and response handling. Consequently, acceptance divided by that rate is an *inferred effective step time*, not a directly measured engine-step duration.

I read both previous replies, section 19, the traffic script, and the implementation under `D:/code/vllm-windows/`. **I changed nothing, ran no benchmarks, and accessed neither GPU.**

---

## Q5.1 — Revised ranking: validate amortization first

Your result strengthens the case for deeper speculation. I would describe it as **amortizing a large per-step dependency cost that itself grows with depth**, rather than a strictly fixed CPU stall.

Using your approximate step times:

| Quantity | ×2 | ×4 | Change |
|---|---:|---:|---:|
| Effective step time | 18.3 ms | 22.7 ms | +24.0% |
| Implied emitted tokens/step | 2.74 | 3.99 | +45.8% |
| Throughput | 149.5 | 175.7 tok/s | +17.5% |
| IDs wait / emitted token | 3.98 ms | 3.49 ms | −12.5% |

The condition for ×4 to win is:

\[
\frac{A_4}{A_2}>\frac{T_4}{T_2},
\]

where \(A\) is emitted tokens per step and \(T\) is actual step duration.

At the reported timing ratio, ×4 needs **24% more tokens per step to break even**, or **36.4% more to deliver a 10% throughput improvement**. With \(A_2\approx2.74\), that means approximately **3.39** and **3.73** tokens/step respectively. These thresholds must be recomputed for each traffic class.

### Remaining actions, ranked by expected value per cost

| Rank | Action | Value | Approximate incremental cost |
|---:|---|---|---|
| **1** | **Mixed-traffic ×2/×3/×4 campaign, with correct latency and step accounting** | Validates an already demonstrated 17.5% opportunity; can yield a default or routing policy | Low engineering cost; planned configuration runs |
| **2** | **Small correlated PLE timeline plus corrected interval counters** | Determines whether early row availability could advance the consumer; fixes ambiguity in current accounting | Roughly 0.5–1 day |
| **3** | **Bounded, matched early-row oracle, primarily at ×4** | Prices the remaining opportunity at the likely production depth | Roughly 1–2 days if trace replay is straightforward; otherwise stop and reassess scope |
| **4** | **Row-ID trace and batch-aware GPU-cache simulation** | Rejects an expensive design cheaply | Approximately 0.5 day once tracing exists |
| **5** | **Depth-layout investigation beyond the arithmetic below** | Potential additional amortization, but diminishing acceptance and substantial layout risk | Cheap source audit; implementation potentially much more expensive |

**Dispatch optimization and true asynchronous SSD I/O remain below these.** The current evidence does not justify reopening either.

Add three things to the campaign:

- **Marginal yield:** extra accepted tokens versus extra step time for ×2→×3 and ×3→×4.
- **Short-output and low-acceptance traffic:** cases where startup/prefill or speculative waste can erase the long-decode benefit.
- **Client-visible streaming gaps:** deeper speculation can improve average throughput while delivering larger, less frequent token bursts.

The structural comparison’s similar acceptance statistics reduce one confound. They do not establish identical token trajectories, PLE access patterns, or computational workload.

---

## Q5.2 — The oracle: exact rows early, not arbitrary warm rows

### Is “background warm cache, IDs ignored” legitimate?

**Not for ordinary live generation.**

A warm host cache maps **IDs → rows**. It does not tell you which rows to consume, in which order, for the current step. Ignoring IDs can silently substitute incorrect embeddings even with a 100% cache hit rate.

It is legitimate only if another mechanism supplies the **exact ordered row payload for that exact lookup**. For an oracle, that mechanism can be a recorded trace.

### Smallest useful intervention

In the inspected implementation:

- `start_prefetch()` preserves IDs and queues their D2H copy.
- `_read_and_copy()` waits on `_ids_ready`, reads the rows, and queues H2D.
- `_finalize_prefetch()` waits for the coordinator, then enforces GPU copy completion before consumption.

The smallest *timing* intervention is to supply an already assembled, exact host payload to this path, while retaining the existing GPU destination and consumer.

Use a bounded recorded workload containing:

- Request and lookup identity.
- Ordered IDs, including duplicates and padding.
- Token counts and graph shapes.
- Actual input/draft tokens and acceptance/rollback metadata needed to reproduce subsequent state.
- Exact row bytes or a trace-specific immutable row store.

Then compare:

| Arm | Row payload | Release condition | Purpose |
|---|---|---|---|
| **Normal reference** | Normal lookup | Actual IDs ready | Checks relevance to current execution |
| **Late exact payload** | Preassembled exact rows | Actual IDs ready | Removes row assembly equally from the matched comparison |
| **Early exact payload** | Same preassembled exact rows | Earliest safe staging-buffer reuse | Prices the IDs-dependent rendezvous |

**The causal comparison is late versus early.** The normal reference quantifies how much preparing payloads changed the baseline.

Crucially, the early H2D must not remain queued behind the original side-stream `wait_stream(current_stream)` or diagnostic D2H. Otherwise the intervention nominally removes the host wait but retains the same device dependency.

Keep:

- `_copy_ready` protection for host staging reuse.
- `_previous_use` protection for device staging reuse.
- Consumer ordering after H2D.
- Full PLE GPU computation.
- The ID producer itself, preferably retained for validation.

### Your nondeterminism changes the replay requirement

**Replaying the same prompt and seed is insufficient.** A changed draft trajectory can produce different required rows.

Two legitimate options are:

1. **Controlled token/state-trace replay:** both timing arms execute the same recorded trajectory. This is a timing oracle, not an autonomous-generation correctness demonstration.
2. **Live replay with exact ID validation:** every consumed lookup must match the recorded identity and ordered IDs. Any mismatch invalidates the run; do not report only matching steps.

For an oracle-only timing experiment, comparisons may be collected asynchronously. For a deployable implementation, incorrect rows must never reach the consumer.

### Residual differences to enumerate

At minimum:

- Preassembly versus normal row lookup and cache locking.
- Extra trace/payload memory and its effect on host pressure.
- H2D submission timing and coordinator scheduling.
- Diagnostic ID-copy/validation overhead.
- Stream dependencies retained or removed.
- Graph segmentation, allocations, and staging-buffer count.
- Forced trajectory versus autonomous decoding.
- Cache warmup and inclusion/exclusion of preparation time.

This is **not** a one-line, correctness-preserving live-generation patch.

### Stop/go gate

Apply the gate at **×4**, and secondarily at ×2 for attribution.

| Matched throughput delta | Decision |
|---|---|
| **95% upper bound <5%** | Stop expensive PLE rendezvous redesign for this workload/depth |
| Point estimate <5%, interval crosses 5% | Economically reasonable to stop, but statistically inconclusive |
| 5–10%, repeatable | Consider only a small implementation with a credible path to retaining most of the oracle gain |
| >10%, with lower bound >5% | Meaningful opportunity; choose an implementation from the timeline and trace |

A weak oracle that accidentally preserves the rendezvous cannot kill the redesign. Verify that the intervention actually advances row readiness and downstream submission.

---

## Q5.3 — Production campaign: three configurations, with ×4 versus ×2 primary

### (a) Two or three configurations?

**Three.**

The extra ×3 startup costs **4.5 minutes**: initial startup cost is **13.5 minutes rather than 9 minutes**. That is inexpensive insurance against ×4 regressing on moderate-acceptance traffic while ×3 remains useful.

Predeclare:

- **Primary comparison:** ×4 versus ×2.
- **Secondary comparison:** ×3 versus ×2.
- **Selection comparison:** ×4 versus ×3.

This prevents retrospectively selecting the best noisy result.

Your proposed gate is reasonable:

- Mixed-set decode improvement **≥10%**.
- Paired 95% lower bound **>5%**.
- No important class with a repeatable **≥5% regression**.
- No unacceptable quality, memory, or latency regression.

Use a fixed request list, not “whatever number of requests fits 70 seconds.” The current script gives faster configurations more requests and therefore a somewhat different prompt mix.

For a practical first pass, use roughly **60–120 held-out requests across six relevant classes**, paired by prompt and sampling settings. Include short and long outputs and at least two prompt-length bands within the supported context envelope.

For confirmation, a reversed configuration order gives **six startups = 27 minutes** total startup cost. Several windows inside one initialization are not initialization-level replication. Bootstrap paired requests or blocks—not individual tokens—and qualify uncertainty that remains from few initializations.

### (b) Required metrics

| Priority | Metric | Why |
|---:|---|---|
| **1** | Decode time and time to last token | Actual user benefit; separates long-decode improvement from whole-request latency |
| **2** | TTFT and p95/p99 inter-chunk gap | Captures prefill effects and burstiness hidden by average tokens/s |
| **3** | Emitted tokens, engine steps, actual step duration | Establishes whether the gain comes from amortization |
| **4** | Drafted, accepted, and discarded tokens **per emitted token** | Measures efficiency across depths |
| **5** | GPU busy time or energy per emitted token, when cheaply available | Detects extra resource consumption that sampled utilization conceals |
| **6** | KV/state memory, remaining capacity, errors and quality checks | Guards against deployment regressions |

Record discarded drafts both per step and per emitted token. The latter is more useful across depths.

Illustratively, if emitted tokens/step equal accepted drafts plus one target token:

\[
W_d=d-(A_d-1).
\]

Your inferred averages imply approximately:

- ×2: **0.26 discarded drafts/step**, or **0.10 per emitted token**.
- ×4: **1.01 discarded drafts/step**, or **0.25 per emitted token**.

Those are illustrative, not substitutes for actual counters; termination and partial draft batches affect the identity.

**I accept that extra work can be free in single-stream wall-clock terms. I do not accept that it needs no accounting.** It may consume energy, lengthen streaming gaps, or become costly under concurrency. If production is strictly single-stream, do not add a large concurrency campaign merely for completeness—but scope the default accordingly.

### Default decision matrix

| Result | Decision |
|---|---|
| ×4 passes all primary gates | Make ×4 the default for the validated serving regime |
| ×4 fails particular classes; ×3 passes | Consider ×3 default or simple class-level routing |
| ×4 and ×3 are within about 2% with overlapping uncertainty | Prefer ×3 if it has better tails or lower work/token |
| Benefit confined to repetitive/high-acceptance traffic | Retain ×2 general default; enable deeper speculation selectively |
| Mean improves but streaming gaps or TTFT violate the SLO | Do not promote on tokens/s alone |

---

## Q5.4 — The layout arithmetic is decisive: ×6 and ×7 also fail

I checked the actual formula in:

`D:/code/vllm-windows/vllm/models/qwen4_exp/common/qsa_cache.py`

```python
span = self.compress_ratio + vllm_config.num_speculative_tokens
capacity = self.compress_ratio * cdiv(span, self.compress_ratio)
```

The checkpoint configuration specifies **`indexer_compress_ratio = 4`**. Therefore:

\[
C(d)=4\left\lceil\frac{4+d}{4}\right\rceil,
\qquad C(d)\mid1616.
\]

The divisors of \(1616=16\times101\) are:

\[
1,2,4,8,16,101,202,404,808,1616.
\]

Only the multiples of four can be ring capacities here:

\[
4,8,16,404,808,1616.
\]

### Nearby depths

| Speculative depth | Ring capacity | Divides 1616? |
|---:|---:|---|
| 0 | 4 | Yes |
| 1–4 | 8 | Yes |
| **5–8** | **12** | **No** |
| **9–12** | **16** | **Yes, for this assertion** |
| 13–16 | 20 | No |

The complete nonnegative depth set satisfying **this one assertion** is:

\[
\{0,\;1\!:\!4,\;9\!:\!12,\;397\!:\!400,\;801\!:\!804,\;1609\!:\!1612\}.
\]

The large ranges are arithmetic curiosities, not operational recommendations.

**Thus:**

- ×6 and ×7 are not reachable with the unchanged layout.
- “Depth four is the mathematical maximum” is incorrect.
- ×9–×12 pass this particular divisibility test, but that does not establish support, useful acceptance, memory feasibility, or correctness.

**Treat ×4 as the current validated operating envelope. Do not jump to ×9 merely to bypass the assertion.**

Do not delete the assertion or clamp capacity to eight. The source comment explains the correctness hazard: rejected draft rows can overwrite committed keys needed later.

Changing block size requires tracing the complete attention/Mamba/QSA alignment calculation. For example, retaining 1616 as a required factor and adding capacity 12 would give an LCM of **4848**, three times larger. That is not a recommendation; it illustrates why a blind adjustment is unsafe and why the real allocator constraints must be derived first.

---

## Q5.5 — Speculative IDs: yes to verified prefetch, no to consuming guesses

**I would not prioritize speculative ID production now.** The ×4 campaign and oracle are cheaper and more decisive.

A constrained version is legitimate:

1. Predict candidate IDs from currently available history.
2. Prefetch immutable rows keyed by those predicted IDs.
3. Once authoritative IDs exist, verify exact identity.
4. Consume only the rows corresponding to authoritative IDs.
5. On a miss or mismatch, use the existing exact path.

That can hide retrieval. **It does not automatically remove the CPU rendezvous** if the host still has to learn and validate the actual IDs before continuing.

A design that consumes guessed rows and “corrects later” is substantially different: all dependent computation and state must be invalidated and replayed. That includes target computation, KV/QSA/Mamba state, acceptance decisions, and output release. **I recommend against that design here.**

### Correctness gate despite nondeterministic greedy output

Do not use final token-string equality as the primary gate. Instead require exact local invariants:

- Authoritative integer IDs match the IDs used to select consumed rows.
- Consumed BF16 row bytes match the immutable checkpoint rows exactly.
- Ordering, duplicates, request identity and buffer generation are correct.
- A mismatched prediction cannot reach a consumer.
- Injected prediction failures take the exact fallback path.
- Exercise rejection/rollback, ring wraparound, request reuse, padding, EOS history and chunked-prefill boundaries.

Floating-point nondeterminism does **not** excuse incorrect integer IDs or embedding bytes.

For performance, measure full-lookup prediction success and fallback penalty, not merely per-row prediction accuracy. If a successful prediction saves \(H\), a failure adds \(M\), and prediction costs \(C\) per lookup, the first-order break-even condition is:

\[
pH>(1-p)M+C.
\]

Use measured values; sampled idle utilization supplies none of them.

---

## Q5.6 — Removing only the CPU wait: expect a fraction initially, not eleven milliseconds

**Do not subtract the 10.9 ms IDs wait from the 18.3 ms effective step time.** Much of that wait is upstream execution that remains necessary.

Also, the logged `block` interval is not pure GPU time: in the inspected `v1/engine/core.py`, it includes result waiting and scheduler output processing. Its p50 cannot be added to another p50 to derive an exact latency floor.

### Quantitative planning estimate

Let \(h\) be the actual critical-path saving while preserving the workload:

\[
R_{\text{new}}\approx149.5\frac{18.3}{18.3-h}.
\]

| Actual saving \(h\) | Effective step time | Approximate throughput | Gain |
|---:|---:|---:|---:|
| 0 ms | 18.3 ms | 149.5 tok/s | 0% |
| 0.5 ms | 17.8 ms | 153.7 tok/s | 2.8% |
| 1.0 ms | 17.3 ms | 158.1 tok/s | 5.8% |
| 1.5 ms | 16.8 ms | 162.8 tok/s | 8.9% |
| 1.8 ms | 16.5 ms | 165.8 tok/s | 10.9% |

**My budgeting prior—not a measured confidence interval—is approximately 17–18.3 ms at ×2, or 150–161 tok/s, with a central working expectation around 3–6% improvement.**

That assumes a narrow intervention removing exposed host turnaround while retaining PLE GPU work and roughly the current execution organization. A stronger restructuring could do better; an intervention that only relocates the wait could yield zero.

There is **no established hard 16 ms floor**. The structural run has different weights, graphs and allocations. Conversely, there is no basis for promising its full approximately 10% gain while retaining PLE.

### Price the opportunity at ×4

At 22.7 ms:

- **5% throughput gain requires approximately 1.08 ms saved per step.**
- **10% requires approximately 2.06 ms saved per step.**
- A 1 ms saving gives approximately **184 tok/s**, or **+4.6%**.

At ×2, the corresponding savings are about **0.87 ms** and **1.66 ms**.

Thus deeper speculation both delivers an immediate gain and raises the bar for the remaining redesign: **the same absolute saving buys a smaller percentage improvement.** This is why the oracle should primarily price ×4.

---

## Q5.7 — The higher cache hit rate strengthens the “not SSD-bound” judgment, but the current counter needs correction

There is a concrete instrumentation issue in the inspected `ple_ssd.py`:

- `self._stat_wait`, `_stat_ids`, `_stat_read` and `_stat_steps` reset after each report.
- **`table.hits` and `table.reads` do not reset.**
- Hits/misses are counted after `np.unique(flat)`: they describe **unique rows within each lookup**, not all row occurrences.
- Prompt prefetch uses the same table and updates those counters too.

Therefore, unless the running source differs, **“93.1–94.1% from cache” is a cumulative, mixed-phase, per-lookup-unique-row statistic—not necessarily the hit rate of the adjacent 200-step decode window.**

This also means the apparent ×4 cache improvement could partly reflect warmup/history and denominator differences.

### What changes in the diagnosis?

The measured row-read interval of **0.21–0.38 ms/step** is the more direct evidence.

At an 18.3 ms effective step time, eliminating all of it would yield only approximately:

\[
1.2\%\text{ to }2.1\%
\]

throughput improvement, **if all of it is exposed**. SSD optimization alone should recover less, because that interval also includes cache lookup, locking and assembly.

So:

- **Further demote SSD throughput and worker-count tuning.**
- Do not attribute the historic 77–80% → 93–94% change without matched interval data.
- Do not infer a GPU-cache hit rate or all-hit-batch probability from the host-cache percentage.

### Trace changes I would request

| Record | Purpose |
|---|---|
| Per-window counter deltas, separated into decode demand / prefill / read-ahead | Establish the actual hit rate |
| Total row occurrences, unique rows, hits and misses | Make denominators explicit |
| All-hit lookup fraction; misses/lookup p50/p95/p99 | Determine whether a GPU cache could avoid a rendezvous |
| Producer completion → preservation → D2H → host wakeup → rows ready → H2D → consumer | Separate upstream dependency from exposed turnaround |
| `_copy_ready.synchronize()` and `_previous_use` delays | Catch staging-buffer serialization outside the currently emphasized IDs timer |
| Shared lookup/step identifiers | Correlate timings rather than divide unrelated averages and medians |

The reported **10.9/17.34 ≈63%** is a useful scale comparison, but not an exact accounting partition: it mixes coordinator-thread averages with submit medians from another scope.

---

## Bottom line

1. **Run the mixed ×2/×3/×4 campaign; ×4 is the leading candidate.**
2. **Fix measurement semantics before interpreting another percentage point:** request versus decode throughput, cumulative versus interval cache hits, and dependency wait versus removable overhead.
3. **Price exact early row availability at ×4.** Below a credible 5% opportunity, stop expensive PLE redesign.
4. **Do not spend startups on ×6 or ×7:** both require capacity 12 and fail the same assertion.
5. **Do not consume guessed embeddings.** Verified prefetch is acceptable; speculative target-state rollback is not justified here.

The strongest discovery is not that eleven milliseconds can be removed. It is that **more useful tokens can be obtained per dependency cycle without first solving that dependency**. That deserves the next production-validation effort.
