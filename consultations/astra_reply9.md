# Buffered IOCP review — reply 9

The disk counters establish that buffered IOCP is effective for device misses in this workload:

- Fresh random buffered passes read about 133.7 MiB in 31,717 operations and complete in 0.241–0.270 s.
- Identical offsets in a new process perform essentially no device I/O: 0.004 MiB and one operation in 0.123–0.124 s.
- Direct `NO_BUFFERING` reads of the same hot offsets still take 3.448–3.697 s and issue 65,539 operations.

Because buffered device-miss runs issue almost one operation per row, request coalescing explains little. The evidence supports a large advantage from the buffered path and its cache behavior. It does not isolate whether the gain comes from IOCP, buffering, alignment, cache interaction, or other filesystem-path effects. It also does not justify declaring `NO_BUFFERING` slow because a minifilter is absent; no such trace was collected.

The smallest experiment with a plausible path to at least 3% overall improvement is an instrumented restart that separates GPU step time, host gaps, and PLE-read completion time for this exact single-stream workload. The current `pyspy` capture shows 61% of active-step samples blocked in `Future.result` under the PLE graph break and 16% in attention backends, but that does not prove critical-path ownership. Existing CUDA events are installed but require a restart. Without that timing split, changing the DLL or IO request shape risks optimizing a component that is already overlapped or too small to matter.

An exact 320-byte buffered request versus the current rounded 4 KiB/8 KiB request is a reasonable secondary experiment. It can determine whether page rounding adds measurable CPU or cache traffic, but the observed near-one-operation-per-row behavior gives no current evidence that it can deliver 3% end-to-end. Comparing IOCP completion batching is similarly useful for attribution, with an even weaker benefit case from the present counters.

Interpretation matrix:

| Result | Interpretation |
| --- | --- |
| GPU step time dominates; PLE waits are outside the critical path | DLL/request changes are unlikely to reach 3% overall |
| Host gaps or `Future.result` waits overlap PLE work poorly | Source audit and IO scheduling may have ≥3% potential |
| Rounded requests materially increase measured critical-path time | Test exact 320-byte buffered requests |
| Exact requests change only disk counters, not TTFT or step cadence | Keep the current path |
| Completion batching changes CPU overhead but not step cadence | No meaningful end-to-end gain |
| Windows GPU-step timing matches the reference while acceptance differs | Separate acceptance behavior from step cost |

Windows medians of 0.719/2.053/9.475 s versus historical WSL best-of-three values of 0.711/2.172/12.690 s indicate comparable magnitude, including better Windows results at longer input. They are not an exact contemporary head-to-head because the WSL runs used a different random token-ID protocol and best-of-three selection. The natural-language probe likewise shows near parity, while its 53–60% acceptance versus historical throughput does not by itself identify step-cost differences.
