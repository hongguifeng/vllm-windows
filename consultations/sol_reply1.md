The host attack is worth one bounded pass, but the two candidates have different roles:

- **B is the serious candidate.** It can plausibly recover enough exposed time to clear the 5% promotion bar.
- **A is an opportunistic candidate.** Its likely standalone gain is below the bar, but its cost is low enough to try if the fused branch is present in the deployed artifact.
- The cheapest missing experiment is a **host-slack probe plus CUDA-event boundary instrumentation**. It can tell whether B is attacking the pacer before spending the 35 minutes on production blocks.

One source-level detail matters first: in this checkout, [`mamba_get_block_table_tensor`](D:/code/vllm-windows/vllm/v1/attention/backends/utils.py:1131) still contains the five PyTorch operations, and `rg` finds no `VLLM_MAMBA_ALIGN_FUSED` symbol. Verify that the deployed copy really contains A before restarting. Otherwise A is not currently an environment-only experiment.

**Q1 — Ordering**

My ranking is:

| Rank | Action | Cost | Expected exposed saving | Decision |
|---|---|---:|---:|---|
| 0 | Verify deployed artifact and run the semantic prefix-mask checks | <5 min | None | Required before A or B |
| 1 | Add event-boundary logging and a temporary host-slack probe | 10–20 min plus one restart | Diagnostic | Cheapest way to decide whether B is attacking the pacer |
| 2 | Run A if the fused branch is actually deployed | About 15 min | Probably 0.2–0.4 ms/step, roughly +1–2% | Worth trying opportunistically |
| 3 | Run B after the fast-path proof | About 35 min | Roughly 0.8–1.5 ms/step if the sync is exposed, roughly +4–8% | Main host candidate |

So “A, then B” is reasonable when A is genuinely free in the deployed artifact. A’s result does not validate or invalidate B: A removes arithmetic and dispatch work; B removes device synchronization caused by CPU-mask indexing.

If the only objective is the highest probability of a promoted change, skip A and do the slack probe followed by B. A cannot normally reach the 5% bar alone. At a 20 ms cadence, even exposing its full 0.42 ms CPU saving gives only about a 2.1% rate improvement:

\[
20 / (20 - 0.42) - 1 \approx 2.1\%.
\]

The practical gain will be lower after the Triton launch and kernel cost.

The upstream cross-group sharing idea remains below the promotion threshold on the supplied numbers: 0.57 ms is approximately 2.8% of a 20 ms step. It can be revisited only as part of a larger change.

**Q2 — Does the build occupy the GPU idle gap?**

Use two complementary measurements.

First, put CUDA events on the same stream around the relevant boundaries:

```text
previous_forward_end.record()

host_build_start = monotonic()
previous_forward_end.query()
build_begin.record()

build_attn_metadata()

host_build_end = monotonic()
previous_forward_end.query()
build_end.record()

forward_begin.record()
enqueue_forward()
forward_end.record()
```

Collect these event intervals:

- `previous_forward_end -> build_begin`
- `build_begin -> build_end`
- `build_end -> forward_begin`
- `forward_begin -> forward_end`

Also record:

- host build wall time;
- whether `previous_forward_end.query()` is false at build entry;
- whether it is false or true when the build returns;
- cadence, device span, and idle gap.

The query results classify overlap cheaply:

| At build entry | At build return | Interpretation |
|---|---|---|
| Previous event already complete | Complete | The build starts after the preceding device work; most of its wall time is capable of pacing the next launch |
| Incomplete | Complete | The build overlaps the tail of the preceding device work; only its post-completion tail can pace the next step |
| Incomplete | Incomplete | Most of the build is hidden behind preceding GPU work; B’s likely gain is limited unless it changes later host scheduling |
| Complete | Incomplete | The event arrangement or stream selection is wrong; do not interpret the result |

For the second measurement, add a temporary delay at the **model-side build call site**, before or at entry to the builder, with values such as 0, 0.5, 1.0, and 2.0 ms. This is a diagnostic delay, not a production change.

Fit the cadence response:

\[
s = \frac{\Delta \text{cadence}}{\Delta \text{injected delay}}.
\]

Interpretation:

- `s ≈ 0` for a delay range: that delay is absorbed by preceding GPU work; the build location has hidden slack.
- `s ≈ 1`: the host is already pacing the next step; an equivalent build saving can reduce cadence almost one-for-one.
- `0 < s < 1`: only part of the build is exposed.
- The transition point estimates the available overlap slack.

The decisive production result for B is then:

- B reduces model-side build wall time;
- device span stays within normal variation;
- idle gap decreases by approximately the exposed part of that reduction;
- cadence decreases by the same amount;
- throughput rises with acceptance length, memory, and request mix held fixed.

If build wall time falls but the device event span and idle gap do not move, the build was overlapped or its time was reabsorbed elsewhere.

The current evidence that the build occurs before forward enqueue is necessary but insufficient. The event query and delay response supply the missing temporal evidence.

**Q3 — Conditions for replacing the mask with a slice**

The current GDN implementation uses boolean row indexing in the two relevant places in [`gdn_attn.py`](D:/code/vllm-windows/vllm/v1/attention/backends/gdn_attn.py:308):

```python
block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]
num_accepted_tokens[spec_sequence_masks_cpu]
```

A slice is equivalent only under a stronger condition than `num_spec_decodes == num_reqs`.

The exact safe condition is that the CPU mask is a prefix mask:

```text
mask[0:num_spec_decodes] == True
mask[num_spec_decodes:] == False
```

That permits full CUDA graph padding at the end. Requiring `mask.all()` is unnecessarily strict; requiring only `num_spec_decodes == num_reqs` can be unsafe if the mask has more entries than `num_reqs` or contains padded rows.

The fast path should assert, at minimum:

```python
assert spec_sequence_masks_cpu.dtype == torch.bool
assert 0 <= num_spec_decodes <= spec_sequence_masks_cpu.numel()
assert bool(spec_sequence_masks_cpu[:num_spec_decodes].all())
assert not bool(spec_sequence_masks_cpu[num_spec_decodes:].any())
```

For the steady-state path, also assert:

```python
query_lens_cpu = query_start_loc_cpu.diff()
assert bool((query_lens_cpu[:num_spec_decodes] > 0).all())
assert bool((query_lens_cpu[num_spec_decodes:] == 0).all())
```

The second assertion describes the expected trailing CUDA graph padding. If the system permits a nonzero non-spec tail, then the mask condition must be checked without assuming zero query lengths.

The semantic traps are:

- **Ordering.** Boolean indexing returns selected rows in increasing original index order. A prefix slice has the same order only when the true indices are exactly `0..num_spec_decodes-1`.
- **Padded rows.** Full CUDA graph buffers can contain trailing rows that are not real speculative requests. The current code explicitly filters them and later fills static buffers with `NULL_BLOCK_ID`, false masks, repeated query offsets, and neutral acceptance values. A slice must select the same real prefix before that staging.
- **Zero-length rows.** A zero-length row should not silently become a speculative row. Check query lengths and the CPU draft-token mask. The existing code has explicit zero-length handling in the mixed path.
- **Capture-time builds.** [`build_for_cudagraph_capture`](D:/code/vllm-windows/vllm/v1/attention/backends/gdn_attn.py:521) calls the ordinary builder with `num_decode_draft_tokens_cpu` derived from `torch.diff(m.query_start_loc)`. The optimization must be valid during capture as well as ordinary replay preparation.
- **Static metadata buffers.** The full graph path copies selected metadata into persistent buffers around lines 430–500. The replacement must preserve the copy lengths and trailing fills. It must not change the shape expected by replay.
- **GPU mask shape and values.** Keep `spec_sequence_masks` unchanged. The model kernels may use it independently of `spec_state_indices_tensor` and `num_accepted_tokens`. An optimization of the CPU indexing must not replace the GPU mask with a shorter tensor.
- **View versus copy.** Boolean indexing returns a new compact tensor. A slice returns a view. For `num_accepted_tokens`, this is normally harmless because it is one-dimensional. For `block_table_tensor[:, :k]`, the slice can have a non-contiguous row stride when the source has wider rows. Verify the consumer’s stride assumptions. If `.contiguous()` is required, its copy may consume much of the gain.
- **Value of `.sum().item()`.** The `.item()` result itself is exact for a CPU boolean tensor. It cannot numerically disagree with `num_reqs`; the possible failure is that the mask length, padding convention, or mask contents differ. Assert against `mask.numel()`, not only `m.num_reqs`.
- **Aliasing and mutation.** The current full-CUDA-graph staging copies from the selected tensors and then fills the persistent destination buffers. Confirm that no later `fill_`, copy, or kernel mutates the slice’s source storage in a way that changes `block_table_tensor`.

The decisive correctness test should construct cases with:

- all true, no padding;
- true prefix plus trailing false padded rows;
- a false row in the middle;
- a zero-length row;
- non-contiguous block-table storage;
- capture-sized tensors with `num_reqs < max capture size`.

For every case, compare the existing masked result and the candidate slice result for shape, dtype, device, values, strides, and the metadata consumed by the forward path. The middle-hole case must reject the fast path.

**Q4 — Is A likely to help?**

A replaces approximately 0.32–0.42 ms of host CPU work per step with one Triton launch.

The expected arithmetic is:

```text
gross host saving:       0.32–0.42 ms
Triton dispatch cost:    about 0.03–0.06 ms
kernel execution/copy:   measure; likely tens of microseconds
net likely saving:       about 0.2–0.35 ms
```

If it is invoked more than once per step, multiply the launch penalty by the number of invocations. The known tensor-descriptor overhead makes the high end of the launch estimate more plausible, and it may add a device-side cost that is invisible in py-spy.

The measurement should record, for baseline and A:

- `mamba_get_block_table_tensor` wall p50/p95;
- per-step model-side build wall time;
- one-launch count per call and per step;
- CUDA-event duration of the fused kernel;
- device span;
- idle gap;
- cadence;
- fixed-table throughput.

A is useful if it reduces the builder wall time and the idle gap. It is not useful as a standalone optimization if:

- the builder p50 falls by less than roughly 0.15–0.20 ms/step;
- device span rises by a similar amount;
- cadence and idle gap are unchanged;
- or two stable production blocks show less than about 2% throughput movement.

A result between 2% and the 5% promotion bar can still be retained as a component of B+A, but it should not receive independent promotion funding.

A should be killed immediately if the fused path adds more than about 0.1 ms/step of device span or if its total dispatch cost consumes most of the measured 0.32–0.42 ms saving. The exact cutoff should come from the event measurement, because the current instrumentation cannot infer Triton kernel duration from py-spy.

**Q5 — How to detect re-absorption**

Track the identity:

\[
\text{cadence} = \text{device span} + \text{idle gap}.
\]

For each variant, compare:

- model-side build wall time;
- total metadata-build wall time;
- scheduler wall time;
- PLE row-assembly wall time;
- Triton dispatch wall time;
- cadence;
- device span;
- idle gap;
- throughput.

A real gap closure looks like this:

```text
model build wall time:  down
device span:            unchanged
idle gap:               down by approximately the exposed saving
cadence:                down by approximately the same amount
throughput:             up
```

Re-absorption looks like this:

```text
model build wall time:  down
idle gap:               unchanged or only slightly down
another host category:  up by a similar amount
cadence:                unchanged
throughput:             unchanged
```

If the model build falls by 1.0 ms and the idle gap falls by 0.9 ms, the work was exposed and the gap closed. If the model build falls by 1.0 ms but the scheduler or Triton-dispatch interval rises by 0.8–1.0 ms, the host work moved elsewhere.

The already-established PLE evidence makes PLE row assembly a less likely re-absorption source, but it does not eliminate scheduler and dispatch work. Use wall timers for those categories; py-spy percentages alone cannot locate the movement on the timeline.

**Q6 — Portfolio ranking**

The correct ranking depends on the still-unknown implementation costs for the engine and kernel changes. The supplied numbers give useful break-even thresholds.

1. **B: remove the two masked-select synchronizations.**

   This is the highest-confidence local host candidate. The relevant code path is identified, the possible saving is comparable to the full idle gap, and the cost is about 35 minutes. If it exposes even 1.0 ms of a 20 ms step, the theoretical cadence gain is about 5.3%, which is near the promotion threshold.

2. **D: raise acceptance length from 2.35 to 2.75, if the drafting change is under roughly 1–1.5 hours.**

   The supplied +17% value dominates every host optimization. Its break-even cost against B is:

   \[
   17\% / C_D > (5\text{–}8\%) / 35\text{ min}.
   \]

   Therefore D has better value per minute than B if it costs less than approximately 75–120 minutes. If the engine-level drafting change is several hours, B has the better immediate return. The implementation must preserve the measured acceptance parity and be evaluated on the same fixed table.

3. **A: fused Mamba alignment helper.**

   Low risk and low cost, but its likely standalone gain is only 1–2%, below the 5% bar. Run it when the deployed artifact is confirmed, especially if a restart is already required. Do not let it displace B or a cheap acceptance prototype.

4. **C: drafter work, 2.36 ms or 13.7% of the step.**

   The ceiling is attractive, but the expected value depends on what can actually be removed. If a change saves fraction \(r\) of the drafter time, its approximate cadence gain is:

   \[
   2.36r / 20 \approx 11.8r\%.
   \]

   A 25% drafter improvement is only about 3% end-to-end; a complete removal would approach 12–14% before other effects. Unless there is a known, small change, it is lower priority than B.

5. **GDN attention kernel work, 8.0 ms or 54% of forward device work.**

   This has the largest absolute ceiling, but it is the least predictable and probably the most expensive. A 10% improvement in that 8 ms is only about 0.8 ms per step, approximately 4–5% end-to-end. A 20% improvement is approximately 9–10%. Before committing engineering time, use a small CUDA-event correctness/timing screen at the production shapes. Broad kernel work should proceed only if that screen shows a credible improvement large enough to clear the cost break-even.

6. **Other host metadata reuse or PLE work.**

   Do not fund these as standalone changes under the current evidence. Cross-group metadata sharing is estimated at +2.8%, and the PLE row path has already failed the critical-path tests. If B changes the idle gap but throughput is still below 5%, then attack whichever measured host interval remains exposed.

A useful practical rule is:

- prototype D first if its implementation estimate is under 90 minutes;
- otherwise run the slack probe and B;
- run A opportunistically;
- screen C and GDN with event measurements before allocating substantial work.

**Q7 — Decision matrix**

Let:

- \(T\) = fixed-table throughput;
- \(C\) = cadence;
- \(D\) = device span;
- \(G=C-D\) = idle gap;
- \(M\) = model-side build wall time.

Use only warm blocks satisfying the existing two-block gate, and reject any block with changed memory behavior, acceptance length, or weight-load activity.

| Observed result after A or B | What it means about the model-side build | Action |
|---|---|---|
| \(M\downarrow\), \(D\) unchanged, \(G\downarrow\) by roughly the same amount, \(C\downarrow\), \(T\uparrow\) | Strong evidence that the changed build work was exposed in the host pacing gap | Promote if \(T\) rises at least 5%; otherwise combine with the other candidate |
| \(M\downarrow\), \(D\) unchanged, \(G\) unchanged, \(C\) unchanged, \(T\) unchanged | The work was overlapped or reabsorbed | Do not promote this change alone; inspect scheduler/dispatch timers |
| \(M\downarrow\), \(D\downarrow\) by approximately the same amount, \(G\) unchanged, \(C\downarrow\) | Throughput improved because device work changed, not because the host gap closed | Attribute the gain to the device side; the pacer hypothesis is not proven |
| \(M\downarrow\), \(G\downarrow\) partially, \(C\downarrow\) partially | The build was only partly exposed | Keep the change if the gain is material; use the event/query data to estimate the remaining exposed tail |
| \(M\downarrow\), \(G\downarrow\), but \(C\) does not fall | Another host interval grew and reabsorbed the saving | Identify the growing interval before doing more metadata work |
| \(M\) unchanged, \(C\downarrow\), \(G\downarrow\), \(T\uparrow\) | The candidate did not touch the measured builder, or the timer is scoped incorrectly; another change caused the gain | Check call-site coverage and timer boundaries; do not credit the candidate |
| \(M\) unchanged, \(C\) unchanged, \(G\) unchanged, \(T\) unchanged | No useful effect or inactive code path | Kill the candidate |
| \(T\uparrow\) but \(C\), \(D\), and \(G\) are unchanged | Acceptance, request mix, or measurement noise changed | Reject the result unless those controls explain it |
| \(T\downarrow\), \(G\downarrow\), \(D\downarrow\) | The optimization may have altered metadata semantics or acceptance behavior | Stop the candidate and run output/acceptance equivalence checks |
| A helps, B does not | Alignment arithmetic/dispatch was exposed; the masked-select path was overlapped, inactive, or semantically guarded | Keep A only if combined value justifies it; kill B |
| B helps, A does not | The synchronization was the exposed pacer; A’s arithmetic was overlapped or launch-bound | Keep B; kill A as a standalone change |
| Both help and combined gain is close to the sum | The two costs are mostly independent | Promote the combined change if it clears 5% |
| Both help individually but the combined gain is much smaller than the sum | They consume the same idle slack or one exposes the other’s replacement cost | Keep the cheaper/better measured option |
| Neither helps | The measured 2.9 ms build is mostly overlapped or another fixed host interval is pacing | Stop this metadata line and move to D, C, or measured GDN work |

The key acceptance criterion for B is therefore not merely “the builder timer got shorter.” It is:

```text
builder wall time down
+ device span stable
+ idle gap down
+ cadence down
+ fixed-table throughput up
```

That combination would establish that the model-side metadata build was acting as the pacer and would justify spending the next iteration on the remaining exposed host interval or on the larger acceptance/device-side opportunities.
