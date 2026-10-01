# Round 5: the trace says 1.55 ms device chain under an 11.1 ms host wait — and my reading of it was wrong in a specific way

Session facts: Windows 11, RTX 5090 (GPU1) for our engine; GPU0 belongs to the user's
own WSL2 engine and is never loaded. Server now stopped, GPU1 shows 0 MiB.
Model: full 143 GiB checkpoint via `lazy` mmap, PLE (4.03 B x 8192 BF16) offloaded to
NTFS via a CPython extension using `NO_BUFFERING` + per-request events.
MTP depth 2, `-Graphs` (breakable CUDA graphs), `VLLM_ENABLE_V1_MULTIPROCESSING=0`
so engine core, worker and the PLE coordinator are threads in one process.
Single-stream decode only; do not ask us to test concurrency.

## 1. What was built

`VLLM_PLE_SSD_TRACE=1` places 8 preallocated CUDA events per step in a ring of 1600
(200-step windows). Events are read only when a window closes, so no synchronization is
introduced on the timed path; one `torch.cuda.synchronize()` per 200 steps. Intervals are
reported stratified by step shape (never pooled). Device intervals use `elapsed_time`;
host intervals use one monotonic clock. The two scales are compared by totals, not bridged.

Overhead measured at essentially zero: with tracing on, three steady-state blocks gave
149.5 / 149.0 / 147.8 tok/s (median 149.0), matching the same-day uninstrumented
149.5 / 148.8. The warm-up gate was enforced (two blocks 145.5 / 146.9, 0.97% apart,
no `load_weights` frames between, then one block discarded).

## 2. The numbers (decode steps, tokens=3, n ~= 200 per window, p50/p95 microseconds)

| Interval | Where recorded | p50 | p95 |
|---|---|---:|---:|
| `p->w` | main: after ids preserve -> side: after `wait_event(_previous_use)` | 318 | 336 |
| `w->j` | side: after the previous wait -> side: after `wait_stream(current)` | **2.94** | 3.07 |
| `j->s` | side: to just before the ids D2H | 3.65 | 3.74 |
| `copy` | around the ids D2H | **45** | 58 |
| `s->e` | D2H start -> D2H recorded | 3.68 | 3.97 |
| `e->h` | ids landed -> rows H2D enqueued | 1115 | 1570 |
| `h->c` | rows H2D enqueued -> consumer point | 53.8 | 66.5 |
| **chain** | preserve -> consumer | **1548** | 2007 |

Host, same steps: `gate` (host wait on previous rows H2D) **3.2 us**; **ids wait 11117 us**;
row read 150 us; `pending` (wait start -> handed back to the submit thread) 11375 us.
Repeated in later windows: chain 1.55-1.95 ms, ids 9.98-10.91 ms.

## 3. My inference last round, and why it was invalid

I argued from `w->j = 2.94 us` that the main stream was empty at that moment, therefore
the 11 ms could not be GPU work, and placed the result in your "GPU finished, host did not
observe it" row. **That inference is unsound: both bracketing events are queued behind the
same main-stream backlog**, so a small `elapsed_time` between them only says the main-stream
work between the two record points was short. It says nothing about how much work preceded
the first record point. `gate = 3.2 us` also does not prove the current step's GPU work was
done; it only proves the previous step's side-stream H2D had completed.

## 4. A second error, found by reading the code we quoted

I claimed the ids round trip CPU->GPU->CPU was "provably redundant". It is not.
`ngram_embedding.compute_ngram_ids` is device work: `ngram_embedding.py:795-801` builds
`positions = torch.arange(num_tokens, device=input_ids.device)`, `packed = torch.full(...,
device=input_ids.device)`, then `torch.searchsorted` and index math, and there is a Triton
path just above it. `ple_ssd.py` has no `_compute_ngram_ids`. `self._ids` is
`torch.empty(..., device="cpu", pin_memory=True)` (`ple_ssd.py:568`) filled by the D2H, and
the SSD table lookup needs ids on the host, so the D2H is load-bearing.

## 5. The dependency structure, verbatim from the deployed copy

`ple_ssd.py:676 start_prefetch` (decorated `eager_break_during_capture(always=True)`,
raises if `torch.cuda.is_current_stream_capturing()`, returns early when
`BreakableCUDAGraphCapture.current() is not None`):

```python
699        self._copy_ready.synchronize()          # host: previous rows H2D
702        self._stream.wait_event(self._previous_use)
704        self._device_ids[:tokens].copy_(ngram_ids)      # main stream, preserves ids
708        self._stream.wait_stream(torch.cuda.current_stream())
714            self._ids[:tokens].copy_(self._device_ids[:tokens], non_blocking=True)
715            self._ids_ready.record(self._stream)
717        self._pending = self._coordinator.submit(self._read_and_copy, tokens)
```

`ple_ssd.py:639 _read_and_copy` waits `_ids_ready`, reads rows into pinned host staging,
then `_copy_to_gpu` (`:665`) enqueues `_gpu[:tokens].copy_(_host[:tokens], non_blocking=True)`
and records `_copy_ready` on the side stream.

Consumer, `ple_ssd.py:852-854` inside `_finalize_prefetch`:

```python
852        torch.cuda.current_stream().wait_event(self._copy_ready)
853        output.copy_(self._gpu[: output.shape[0]].flatten(-2))
854        self._previous_use.record()      # on the main stream
```

Call path: `model.py:469 _start_layer_ple_prefetch` -> `ple_layer.py:154 start_prefetch` ->
`ngram_embedding.py:858 start_prefetch` -> `compute_ngram_ids` -> `ple_ssd.start_prefetch`.
The docstring says "Start the pinned lookup **while the preceding decoder layer runs**", so
at the moment `start_prefetch` executes, the main stream already holds the current step's
work for every layer before this PLE layer, and `wait_stream(current)` makes the side stream
wait for all of it.

## 6. The reading I now hold, and want tested

The 11 ms is the current step's GPU work that precedes this PLE layer (including the
kernels that produce the ids), which `wait_stream(current)` forces the side stream to wait
behind. Everything measured on the device's own clock is short because those marks all sit
in the same backlog; the host clock starts before that backlog executes. If this is right,
the multi-buffer rotation I was about to build does not help, because the dependency is on
real GPU output, not on buffer reuse.

**The number we lack is how much of the step's GPU work precedes the PLE layer.** We can get
it for about four lines: record an event on the main stream at the top of `execute_model`
(or at the first `_start_layer_ple_prefetch`) and keep `e_preserve`; their `elapsed_time` is
that preceding work, measured on the device clock. Would you accept that as the decisive
measurement, and is there a better four-line version?

## 7. The A/B we ran, and what it does and does not show

Blocking `Event.synchronize()` versus `query()` + `sleep(0.2 ms)`, both with tracing on:

| Arm | Blocks | Median | ids wait |
|---|---|---:|---:|
| A blocking | 149.5 / 149.0 / 147.8 | **149.0** | 9.98-10.91 ms |
| B polling | 148.0 / 145.0 | **146.5** | **11.42 ms** |

I reported that this partially falsifies "the GIL is held while waiting". Two caveats I now
see: B cannot distinguish late-event from GIL-starvation, because `query()` and resuming from
`sleep()` both require the interpreter lock; and A beating B is weak evidence that
`synchronize()` releases the lock. Do you agree that B is not a clean discriminator, and if
so, what is the cheapest clean one? A GIL canary thread (increment a counter in a loop, check
whether it stalls during the ids wait) is one candidate we could afford.

## 8. Questions, ranked by how much they cost us to answer

1. Given the corrected dependency structure, is reading 6 (the 11 ms is this step's preceding
   GPU work) the best explanation of all 15 numbers, or does something still not fit?
2. Confirm or refute reading 3's invalidation: is `w->j` uninformative for the reason above,
   and does any interval in our table still constrain the host-lag hypothesis?
3. Design the four-line decisive measurement for "GPU work preceding the PLE layer", including
   where exactly the anchor event should be recorded in vLLM's `execute_model` for a
   breakable-graph step, and what its p50 would have to come out as to support or kill reading 6.
4. Under breakable CUDA graphs, is `elapsed_time` between events recorded in eager segments
   trustworthy when the work between them executes inside a captured graph? If not, how do we
   tell without rebuilding the graph path?
5. If reading 6 survives: is there any way to get this step's ids earlier than the layers
   before this PLE layer complete (host-side derivation from previous-step state, a cheaper
   producer, splitting the producer from the layer), and what would each cost?
6. Given all of the above, what should we do next: the measurement in (3), the mixed-code
   depth battle (x2/x3/x4, code-weighted, fixed request table), or the bounded early-rows
   oracle priced on x4? Rank three moves by expected value per unit cost, including the cost
   of a server restart (~4 minutes) and a measurement block (~75 seconds).

## 9. Standing numbers, for continuity

Depth ladder, same day, byte-identical launch except `num_speculative_tokens`:
x2 median 149.0-149.5 tok/s (step 18.3 ms, submit p50 17.34 ms, GPU util 40.2%),
x3 155.8 (+4.2%), **x4 175.7 (+17.5%)** with a working ceiling at 4; x5 fails at startup
with `AssertionError: QSA ring capacity 12 must divide the attention block size 1616`.
PLE-free structural view at x2: 165.0 tok/s, submit p50 5.08 ms, block p50 10.85 ms,
accept length and rate aligned with the full-weight run, GPU util 45.2%.
Corrected PLE window counters: 48.0/53.6 rows per lookup (not 350-520), unique-row hits
95.8-99.6%, all-hit lookups 83.5-98.5%, **misses per lookup p50 = 0**, lookahead 0, so the
independence arithmetic you derived last round is void and a GPU hot-row cache is dead.
Our reader is synchronous (`NO_BUFFERING | RANDOM_ACCESS`, no `FILE_FLAG_OVERLAPPED`);
there is no I/O deadline in production; the only real parallelism is a Python thread pool.

Measurement protocol you set and we are holding to: >= 2500 decode tokens of warm-up before
a block counts, hard gate that `load_weights` frames are absent in repeated `py-spy` samples
and never return, two consecutive blocks within 2% to declare the drift stopped, discard one
whole block after stopping, and only later blocks enter the median.
