I’ll treat the corrected 48–53.6-row window as the operative measurement and separate the cache question from the 10.7 ms rendezvous question. The key diagnostic is a timestamp decomposition whose intervals sum to the observed latency without adding per-step synchronization.
The corrected numbers change the emphasis. At 48–53.6 rows per lookup, with 83.5–98.5% of lookups requiring no disk read and a median of zero missing rows, the PLE storage path is not a plausible explanation for the 10.7–10.9 ms rendezvous. The device row cache remains worth testing as a small optimization, but it is no longer the main causal question.

### Q6.1

I accept the separation:

- A device row-cache demotion can plausibly affect the 0.33–0.67 ms row assembly or transfer component.
- That is approximately **1.8–3.7% of an 18.3 ms step**.
- The 10.7–10.9 ms IDs rendezvous is approximately **58–60% of the step** and should be largely independent of whether the rows arrive from disk, host memory, or a device cache.
- The corrected hit structure does not reopen the original disk-I/O interpretation. A lookup with zero missing rows still has to pass through the rendezvous and row-publication path, but it has no PLE read to explain that wait.

The 16 KB payload estimate makes the oracle easier to construct. A preallocated pinned buffer or ring slot can hold one approximately 50-row payload without allocation or meaningful bandwidth pressure. At 10 GB/s, transferring 16 KB is about **1.6 microseconds of raw payload time**. Even allowing for launch, synchronization, and visibility overhead, a 10.7 ms interval cannot be explained by the bytes themselves.

I would retain the three arms. Keep **late versus early** as the causal pair, with identical lookup IDs, row contents, payload sizes, and cache state. The third arm remains useful as the current-path or implementation-control arm. The early arm should preassemble its payload outside the measured critical interval; otherwise the oracle measures assembly rather than publication timing. The late arm should perform the same amount of preparation and differ only in when the payload becomes usable.

The stop gate should move from a per-row hit-rate requirement to an end-to-end effect and attribution requirement:

| Result | Interpretation | Decision |
|---|---|---|
| Early arm saves at least **0.25 ms median** and the saving appears in the row-path timestamps | Device residency or earlier publication has a measurable practical effect | Continue the cache/oracle investigation |
| Row-path timestamps improve by 0.33–0.67 ms, but end-to-end latency improves by less than 0.25 ms | The 10.7 ms rendezvous dominates the benefit | Stop treating device caching as the main optimization target |
| Early arm improves only windows with missing-row tails | The effect is a miss-tail effect rather than a normal-step effect | Report median and tail results separately; do not generalize it to all steps |
| No row-path improvement | The cache arm did not change the intended path, or the measurement is not reaching it | Inspect arm validity before drawing a performance conclusion |
| End-to-end improvement exceeds roughly 0.67 ms | More than the estimated row component changed | Check for an unintended scheduling or synchronization difference between arms |

The 0.25 ms threshold is a practical gate, equal to about **1.4% of an 18.3 ms step**. It is deliberately below the estimated 0.33–0.67 ms opportunity. The all-hit rate should be reported as a workload descriptor, not used as a requirement for accepting or rejecting the oracle.

### Q6.2

I accept the proposed warm-up protocol, with one operational definition added:

1. Wait until `load_weights` is absent from repeated py-spy samples and does not reappear.
2. Confirm that request throughput has stopped trending materially.
3. Discard one complete traffic block.
4. Use subsequent blocks for medians.

For reproducibility, I would define “stopped trending” as two consecutive complete blocks whose median throughput differs by no more than **2%**, with no `load_weights` frame observed between them. If the discarded block and the next block differ by more than 2%, discard additional blocks until that condition holds.

The loader-frame condition is the hard gate. Throughput alone is insufficient because it can stabilize while page-in or other startup work is still progressing. Conversely, a temporary absence from one py-spy sample is insufficient, so the zero-frame observation should be repeated.

The rise from **144.4 to 149.5 to 152.2** is plausibly partly page-in contamination. The evidence is strong for the first request: it took 72.5 seconds, py-spy showed the EngineCore thread inside `load_weights`, and device memory was still rising. The warm OS page cache reducing a later startup to 4.5 minutes also supports that explanation. The exact fraction of the later 144.4-to-152.2 increase attributable to page-in cannot be estimated from throughput alone. After the loader frames disappear and the stabilization gate passes, remaining changes should be treated as ordinary runtime warm-up unless another source of drift is observed.

The ordinary buffered reads for memory-mapped weight loading and the `NO_BUFFERING` PLE reads should be treated as separate cache domains. A warm OS page cache for weights does not establish that PLE rows are warm.

### Q6.3

The 10.7 ms should be decomposed into intervals that follow the actual dependency chain. Record host timestamps with one monotonic clock and CUDA events in preallocated per-step rings. Read or query the event rings after the block; do not call a per-step `cudaEventSynchronize` on the measured path.

I would record these marks:

| Mark | Clock and placement | Role |
|---|---|---|
| `h_ids_enqueue_begin`, `h_ids_enqueue_end` | Host monotonic clock immediately around the IDs launch | Measures CPU enqueue cost and anchors the timeline |
| `e_ids_reached` | CUDA event immediately after the IDs producer work | Measures how long the GPU takes to reach and complete that point |
| `e_producer_done` | CUDA event after the producer has finished writing the preserved data | Separates producer completion from later preservation work |
| `e_preserve_done` | CUDA event after the preservation copy | Shows when the data needed by the side stream is actually available |
| `e_wait_passed` | CUDA event immediately after the side stream’s wait on `_previous_use` | Shows when the side stream has passed the dependency |
| `e_d2h_start`, `e_d2h_end` | CUDA events immediately before and after the asynchronous D2H copy | Separates queueing delay from copy duration |
| `h_wait_begin`, `h_wakeup` | Host monotonic clock around the host wait or polling interval | Measures host visibility and wake-up latency |
| `h_rows_ready` | Host timestamp at the release/publication point after row assembly | Measures assembly plus publication delay |
| `h_h2d_enqueue` | Host timestamp when H2D is enqueued | Connects host row publication to device transfer |
| `e_h2d_done` | CUDA event after H2D | Establishes when the consumer can receive the payload |
| `e_consumer_arrival` | CUDA event immediately before the first consumer operation that uses the rows | Measures downstream ordering delay |

The most useful derived intervals are:

```text
IDs enqueue → producer done
producer done → preservation done
preservation done → side-stream wait passed
wait passed → D2H start
D2H start → D2H end
D2H end → host wakeup
host wakeup → rows ready
rows ready → H2D enqueue
H2D enqueue → H2D done
H2D done → consumer arrival
```

The first interval needs care: a CUDA event placed after a kernel measures when the GPU reaches that event, not when the host submitted the kernel. That is why the host enqueue pair and the post-producer CUDA event are both needed.

For the first diagnostic campaign, all of these are load-bearing. I would classify them as follows:

- **Essential:** IDs enqueue, producer completion, preservation completion, wait passed, D2H start/end, host wake-up, rows ready, H2D completion, and consumer arrival.
- **Conditionally mergeable:** producer completion and preservation completion can be merged only if preservation is structurally part of the producer and there is no independent copy or queue.
- **Potentially removable after validation:** H2D completion can be removed in a later reduced trace if the consumer stream has an explicit dependency directly on H2D completion and the first trace proves there is no intervening queueing delay.
- **Do not remove initially:** D2H start, host wake-up, rows-ready publication, or consumer arrival. Each distinguishes a different explanation for the 10.7 ms.

The decision matrix is:

| Dominant measured interval | Likely location of the 10.7 ms | Conclusion |
|---|---|---|
| IDs enqueue → producer done | Main-stream queued work or GPU scheduling before the IDs result is produced | The wait is upstream of row I/O |
| Producer done → preservation done | Preservation copy or its stream ordering | The producer finished, but the preserved result was delayed |
| Preservation done → wait passed | Side-stream dependency or `_previous_use` ordering | The side stream is waiting behind an ordering edge |
| Wait passed → D2H start | Side-stream queueing after the dependency | The dependency passed, but the copy was not issued promptly |
| D2H start → D2H end | Actual transfer or a hidden pageable-memory path | A 10.7 ms value here would indicate a copy-path problem |
| D2H end → host wake-up | Host polling, callback, or event-visibility delay | The GPU finished, but the host did not observe it promptly |
| Host wake-up → rows ready | Row assembly, lock contention, or publication delay | The delay is on the host after visibility |
| Rows ready → H2D done or consumer arrival | H2D ordering or consumer-stream queueing | The consumer is not receiving the ready payload promptly |
| Every individual interval is small, but the total is large | Missing mark, mismatched event ring slot, or clock-domain interpretation error | Fix instrumentation before interpreting the result |

My prior expectation is that the 10.7 ms will appear **before the D2H transfer itself**, most likely in the path from IDs enqueue to producer completion or in the dependency from preservation to the side-stream wait. A true 16 KB transfer duration should be measured in microseconds to at most a small fraction of a millisecond on a functioning pinned-buffer path. A multi-millisecond D2H interval would therefore be evidence of hidden synchronization, pageable memory, or queueing—not payload bandwidth.

Stratify these intervals by the two observed step shapes. The 48-row and 53.6-row cases should not be pooled until their producer and consumer timing distributions are shown to be equivalent.

### Q6.4

I have no objection to the protocol:

- lazy loading enabled;
- loader frames gone;
- throughput stabilization confirmed;
- one complete traffic block discarded;
- medians computed only from later blocks.

The additional requirement is to record the warm-up gate and the discarded block’s throughput in the campaign log. If the post-discard block still differs from the discarded block by more than the predefined 2% criterion, discard further blocks rather than mixing them into the measured median.

The timeout behavior belongs in the campaign cost model. Every run should finish below the relay timeout with margin, and the real engine log should be taken from `_dev/out/logs/flashnext_<timestamp>.log`. A timed-out relay must be treated as an incomplete run; its surviving Python process and device allocation make it unsafe to interpret as a normal failed measurement.
