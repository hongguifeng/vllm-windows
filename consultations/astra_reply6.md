## Bottom line

**The n-gram hash has no dependency on current-step hidden states, but “the host already holds all its inputs before replay” is false for the deployed ordinary MTP path.** There are two separate problems:

1. **The next verification step’s draft token values normally remain on the GPU.** The scheduler has their count and `-1` placeholders, not their values.
2. **Your measured engine uses the batch-queue path.** That path can submit step N before processing step N−1’s output on the host. Single-request serving and `VLLM_ENABLE_V1_MULTIPROCESSING=0` do not disable that overlap.

I inspected `D:/code/vllm-windows/vllm/` and verified that the key files below are byte-identical to their deployed `.venv/Lib/site-packages/vllm/` copies. I made no changes and ran no GPU work.

This does **not** kill earlier PLE lookup. It changes the proposal from “derive already-known IDs on the host” into **“make a complete, correctly versioned input snapshot available earlier, then derive IDs.”** That extra dependency must be priced before implementation.

## 1. What the Triton path reads—and what is missing on the host

All source paths below are relative to `D:/code/vllm-windows/vllm/`.

### The hash itself is entirely token-derived

`models/qwen4_exp/nvidia/ops/ple.py:25–116` reads:

- Flattened `input_ids`.
- `query_start_loc`.
- Per-request `ngram_context`.
- Layer-specific multipliers, head vocabulary sizes, and head offsets.
- EOS ID and shape/configuration constants.

**It does not read hidden states, KV cache, QSA state, acceptance decisions from the current verification, or PLE convolution history.** PDL synchronization is execution ordering, not another semantic input.

Thus, given the *exact* token/layout/context snapshot, CPU computation is possible.

### But decode `input_ids` are not generally known at schedule time

`v1/worker/gpu/input_batch.py:398–455` constructs generated-token inputs by reading:

```text
req_states.last_sampled_tokens
req_states.draft_tokens
```

on the GPU. In particular, it places the draft values into the verification suffix.

Where do those drafts come from?

- `v1/worker/gpu/model_runner.py:2152–2178` runs the proposer and stores its results in GPU request state.
- `v1/worker/gpu/spec_decode/utils.py:40–79` explicitly **skips draft D2H for ordinary, non-structured-output requests**.
- Its host return in that case is:

```python
draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
```

Consequently:

> Knowing the previously accepted tokens does not imply knowing the new draft candidates being verified in this step.

Those candidates affect IDs even if this step later rejects them.

### The host-history ordering assumption also fails

Your graph-run log contains `core.py:869` **`block`** statistics, emitted by `step_with_batch_queue`, rather than the ordinary `step` path’s `wait` statistics.

In `v1/engine/core.py:718–820`, the order is explicitly:

1. Schedule and submit a new batch.
2. Then retrieve an older queued result.
3. Then call `update_from_output`.

The configuration also defaults to async scheduling when compatible (`config/vllm.py:1456–1505`). This is internal iteration overlap, **not a request-concurrency experiment**.

Two additional warnings in the source reinforce this:

- `v1/core/sched/async_scheduler.py:15–44` maintains speculative and output placeholders.
- `v1/worker/gpu/states.py:61–62` labels the CPU computed-token mirror an **optimistic upper bound**, not an exact mirror.

Therefore there are two independent counterexamples to section 3. Even under synchronous scheduling, ordinary draft values are missing; with the measured batch queue, the latest committed history may also be unavailable at submission.

### What I accept from the control experiment

I retire the prescribed prefix anchor: it did not measure the intended quantity.

The control strongly supports enqueue cadence as the reason the dependency is much more exposed under graphs. However, `ready=False` at wait entry establishes only that the event is not complete then. It does not by itself attribute every microsecond to current-step execution rather than previous-step tail, submission gaps, or observation delay.

Most importantly, **10.4 ms of host waiting is not 10.4 ms of removable step latency**.

## 2. Appropriate implementation scope

**Yes: a guarded CPU decode path with GPU fallback is a sensible eventual scope—but eligibility must mean “exact host snapshot available,” not merely “decode.”**

For a first implementation I would require:

- Pure target-model decode, including its verification tokens.
- Exact sampled-token and draft-token values for this iteration.
- Exact committed context ending at the GPU-equivalent computed-token position.
- Supported request mapping and padding.
- No GPU-only adaptive layout changes.
- GPU fallback for prefill, mixed/transition batches, dummy execution, and unsupported cases.

Do not enable a global “decode uses host IDs” switch that also intercepts any PLE calls made during drafting. Each invocation needs its own input provenance.

### Smallest useful interface

Use one immutable, versioned descriptor containing:

```text
iteration/invocation ID
request ID + request-generation ID
packed request order and token boundaries
active and padded token/request counts
exact current input tokens
exact ngram context
layer/hash-parameter version
```

The CPU function consumes that descriptor plus frozen hash parameters and writes IDs into a leased buffer. The existing GPU function remains the fallback and comparison reference. Row lookup and consumption should be shared.

Two existing pieces reduce the arithmetic work:

- `ngram_embedding.py:771–851` already has a CPU fallback.
- `ple_ssd.py:PLEPromptPrefetcher` already creates a CPU copy of the hash buffers.

Reuse those semantics for a prototype. **Copy the loaded hash buffers**, rather than regenerating them from configuration: `load_weights` can overwrite multipliers, sizes, and offsets.

The main engineering burden is obtaining the descriptor without serializing the pipeline, not writing the hash.

## 3. Concrete correctness invariants and comparison plan

### Required invariants

| Area | Local invariant to prove |
|---|---|
| **Snapshot identity** | Every token, boundary, context row, and parameter set belongs to the same request generation and target invocation. “Latest available” is not a sufficient key. |
| **MTP acceptance** | Only committed sampled outputs extend persistent history. Rejected draft suffixes never extend it. The current verification input nevertheless contains **all scheduled draft candidates**, including candidates that this verification later rejects. |
| **Computed-token boundary** | Context ends at the exact GPU `num_computed_tokens`, not total host output length or the optimistic CPU bound. In ordinary decode, the newest sampled token is generally the next input, not already-computed context. |
| **Packing** | CPU row `j` maps to the same request and within-request offset as GPU row `j`. Preserve duplicates and head order. No predecessor can leak across requests. |
| **Padding** | Distinguish active rows, repeated query boundaries, dummy requests, and token padding. Either reproduce every row the current lookup reads or explicitly prove excluded rows cannot affect live outputs. Never silently interpret `-1` placeholders as tokens. |
| **EOS** | Walk predecessors newest to oldest. Once a predecessor is EOS, every older predecessor contributes EOS. The **current token being EOS does not itself erase its preceding history for that token’s hash**. Missing left context is EOS-filled. |
| **Hash arithmetic** | Products/XOR use the same 64-bit bit patterns, including overflow; interpret the mixed value as signed int64 for nonnegative remainder, then add the head offset. Python unbounded integers and unsigned modulo are not interchangeable with this. |
| **Ring/offset wrap** | Host-history ring positions refer to absolute token positions. Rejection/rollback and wrap cannot leave discarded candidates in the context. Buffer-slot generations must also survive wrap without accepting stale futures or events. |
| **Request reuse** | Reusing a numerical request-state index does not reuse history or a pending lookup. Key state by request generation; initialize short history with EOS; reject stale completions. |
| **Chunked prefill** | Context is the suffix immediately preceding the chunk’s computed-token boundary—not the end of the known prompt. First decode after prefill must neither skip nor duplicate the prompt tail or first sampled token. |

For MTP, the GPU update is visible in `input_batch.py:553–610`: it appends valid sampled outputs and adjusts computed tokens using `query_len - num_rejected`. Mirror the **net transition**, including any separate optimistic increment; do not apply that increment twice.

A useful ordinary-decode example:

```text
Previously committed history: H
Last sampled, not-yet-computed token: b
New draft candidates: d0, d1

Current verification inputs: [b, d0, d1]
Current context: suffix(H)
```

If verification accepts `d0` but rejects `d1`, the next committed history includes `b,d0`, and the correction token becomes the next input. It must not retain `d1`.

### Bitwise comparison in three layers

**A. Hash-only equivalence**

For identical token/layout/context inputs, compare:

1. An independent scalar reference.
2. The existing CPU implementation or proposed implementation.
3. Captured outputs from the production Triton implementation.

Require exact integer equality for every token/head, not an aggregate checksum alone.

Extend the nearby `tests/models/qwen4_exp/test_ple.py` cases. They already cover empty requests, trailing request padding, EOS, int64 overflow, and large token IDs. Its reference needs care for token padding beyond the final real boundary.

**B. State-reconstruction equivalence**

Freeze the host candidate **at the proposed early-submission point**. Later compare it with version-matched GPU:

- Input tokens.
- Query boundaries.
- Context.
- Computed-token positions.
- Resulting IDs.

This distinguishes a correct hash fed stale inputs from an incorrect hash.

**C. Lifetime/row equivalence**

Compare looked-up row bytes in packed order, and validate slot-generation ownership through repeated reuse. Correct IDs are insufficient if a late completion writes into a reused rows slot.

Exercise all acceptance lengths `0…k` for x2/x3/x4, rejection followed by acceptance, prompt-to-decode transitions, EOS at every relevant predecessor distance, request reuse, and repeated ring wraps. Packing cases can be synthetic local tests; no concurrent serving run is needed.

**Acceptance criterion: zero mismatches, with explicit coverage counters.** A long run of all-accepted drafts does not validate rejection handling.

## 4. Minimal correct rows-buffer ordering

The present code already separates these responsibilities:

- `_copy_ready.synchronize()` protects pinned **host rows** from being overwritten while H2D still reads them.
- `_stream.wait_event(_previous_use)` protects **GPU rows** from overwrite before their previous consumer finishes.
- `wait_stream(current)` orders the **new GPU IDs** before D2H.

So it is not necessary to retain the broad main-stream wait merely to protect rows.

For slot `s`, generation `g`, I would accept:

```text
Host:
    acquire slot/generation
    ensure previous CPU task no longer reads ids[s]
    ensure previous H2D has finished reading host_rows[s]
    fill ids[s]
    read/assemble host_rows[s]

Copy stream:
    wait on consumed[s, g-1]
    H2D host_rows[s] -> gpu_rows[s]
    record ready[s, g]
    publish that this event record has been enqueued

Consumer stream:
    wait on ready[s, g]
    copy gpu_rows[s] into graph-owned output
    record consumed[s, g]
```

Important details:

- The future must not become “ready” until the **current generation’s event record is enqueued**. Waiting on an old or not-yet-recorded event can otherwise pass incorrectly.
- CUDA stream waits do not protect CPU writes into pinned memory. Host staging needs an actual completion check or another leased buffer.
- Record `consumed` after the **last GPU read of staging**. With the current `output.copy_`, that is after the copy; with aliasing, it could be much later.
- The previous `consumed` event must already have been recorded/enqueued before the copy stream waits on it. Otherwise slot reuse can wait on the wrong generation.
- Graph replay must use the intended slot address. Rotating Python references does not update captured pointers.
- Capture/dummy execution must not create real I/O work or consume production slot generations.
- Cancellation, fallback, and exceptions must not leave an old task able to overwrite a newly leased slot.

**One slot is sufficient for correctness under serialized use.** More slots are justified only if measured reuse waits prevent useful overlap.

Even with host IDs, `_pending.result()` only collapses if rows and the event publication get ahead of the consumer. CPU assembly, dispatch delay, H2D, and slot reuse can still block it.

## 5. Ranking by expected value per unit cost

The source audit above is already done and costs **zero restarts**. After that, my ranking is:

| Rank | Move | Value and cost |
|---|---|---|
| **1** | **Fixed-request, code-weighted x2/x3/x4 battle** | Existing evidence offers x4 **+17.9% over 149.0** and x3 +4.6%, with almost no implementation risk. Three launches plus three counted blocks each cost **23.25 minutes**, excluding warm-up/gates. Adding two convergence blocks and one discarded block per launch gives a best-case **34.5 minutes**, assuming those satisfy the other warm-up requirements. |
| **2** | **Bounded early-rows oracle on x4** | Prices the remaining opportunity without first building a host-state mirror. Keep it bounded and explicitly non-production; it must not borrow future IDs silently or omit rows-lifetime protection. One launch with three baseline and three oracle blocks costs **11.5 minutes before gates**, if both arms can safely run within that launch. Add restarts if not. |
| **3** | **Earlier snapshot export plus guarded CPU decode IDs** | Do only after correctness/provenance checks and a material x4 oracle gain. Cost includes state plumbing, lifetime tests, fallback, and repeated validation—not just one hash function. Its benefit is currently unmeasured. |

Thus: **full host-ID implementation after both the depth battle and the x4 oracle.** A lightweight availability audit can be attached to either earlier run.

### Quantitative decision thresholds

At unchanged accepted tokens per step, x4’s 22.7 ms step requires approximately:

- **0.445 ms saved** for +2% throughput.
- **1.08 ms saved** for +5%.
- **2.06 ms saved** for +10%.

I would not fund substantial state-mirroring work for an oracle benefit below roughly 2%, given your drift gate. A repeatable ≥5% opportunity is more compelling.

At x2, 165/149 is **+10.7%**, equivalent to roughly **1.78 ms** of step reduction at unchanged acceptance. That is a useful scale from the PLE-free structural run—not a strict upper bound for an exact implementation. It does not establish a larger opportunity at x4.

Submit time alone is not the objective: its reduction can reappear as output blocking, as your PLE-free measurements already demonstrate.

## 6. Cheapest decisive experiment

### For the literal claim: no restart is necessary

The deployed code already falsifies it:

- Ordinary drafts are GPU-only and represented by host placeholders.
- Batch-queue execution does not guarantee `update_from_output(N−1)` precedes submission of N.

I would not spend 5.25 minutes rediscovering that.

### For the repaired hypothesis: one restart plus one diagnostic block

If you want to determine whether an earlier snapshot is practical, add a **non-intervening availability witness**, leaving production PLE unchanged.

At the proposed early-prefetch point, record:

1. Target invocation/iteration and request generation.
2. Latest host output version already processed.
3. Whether exact last-sampled and draft values are present—or only placeholders.
4. Exact versus upper-bound computed-token position.
5. Packed layout and padding.
6. Timestamp when every required host field first becomes available.

Optionally retain a small bounded set of versioned GPU input/context snapshots for deferred comparison. Do not insert `.cpu()` or synchronization at the early point and then call that “already available.”

Use the existing IDs D2H as the later output reference. Any extra snapshot copies are diagnostic overhead, not a throughput result.

Interpretation:

| Result | Conclusion |
|---|---|
| Missing drafts or unprocessed prior output at submission | Original claim false; export/provenance work required. |
| Complete candidate but token/context mismatch | Host reconstruction wrong; stop before row-path changes. |
| Inputs match but IDs differ | Hash/EOS/overflow/layout implementation wrong. |
| Inputs and IDs match, but become available only near the existing ID arrival | Semantically feasible, little demonstrated timing advantage. |
| Complete exact snapshot arrives substantially earlier | Proceed to price row overlap; still not proof of end-to-end speedup. |

One such semantic block need not establish steady-state throughput. If you use it to claim a timing gain, the existing warm-up and stability gates still apply.

**Recommendation:** keep the CPU-ID idea as a conditional optimization, but do not build it under the premise that all inputs are already on the host. The missing draft values and queued-output ordering are concrete, deployed-code obstacles—not hypothetical edge cases.
