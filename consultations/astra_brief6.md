# Round 6: the ids wait is an enqueue-cadence artifact, and the host already holds everything the ids need

Same session as round 5. Windows 11, RTX 5090 on GPU1; GPU0 is the user's engine and is
never loaded. Server stopped, GPU1 at 0 MiB. MTP depth 2, breakable CUDA graphs (the
capture log shows FULL), `VLLM_ENABLE_V1_MULTIPROCESSING=0`. Single-stream decode only;
concurrency experiments are out of scope by the user's instruction.

## 1. The control experiment, and what it settled

Same instrumentation, same launch parameters, the only difference being `-Graphs`:

| Quantity | graphs (FULL), 147 tok/s | eager (`enforce_eager=True`), 17.3 tok/s |
|---|---:|---:|
| `dispatch -> preserve` p50 | 36.86 us | **30.72 us** |
| **ids wait p50** | **10393 us** | **250 us** |
| previous consumer -> this ids landed p50 | 16385 us | 116180 us |
| device chain p50 | 1548 us | 3763 us |
| `e->h` p50 | 1115 us | 3306 us |
| ready before waiting | 0 of 200 | 0 of 199 |
| seen true by main thread | 200 of 200 | 199 of 199 |

Two conclusions:

1. **The prefix anchor is unusable.** In eager it is also ~30 us, the same magnitude as in
   graph mode. In both modes the main stream is drained at those two record points, so both
   events execute on enqueue and the interval measures the record overhead of the second
   event. Your prescribed anchor placement cannot produce the quantity you wanted here.
2. **The wait waits for the device to reach this step's ids producer, and its price comes
   from enqueue cadence.** `_start_layer_ple_prefetch` appears only inside the model forward
   (`model.py:516` and `:528`); during capture it is preserved as an eager segment at its
   original position and executes at that position on replay. In eager the host enqueues
   kernel by kernel and the device keeps up, so by the time the host reaches the prefetch the
   preceding kernels are done and `wait_stream(current)` is satisfied immediately: 250 us.
   Under a full-graph replay the whole step enters the queue at once, the host races a full
   step ahead of the device, and the ids must wait for the device to climb to the producer:
   10.4 ms. Both wait for the same thing; only how far ahead the host enqueued differs.

So reading 6 was right about what is being waited for, and the readiness test from round 5
(`ready before waiting in 0 of 200`) says it is a real device-side dependency rather than
scheduling. What is not inherent is its cost.

## 2. Two routes you ranked are now closed

- **Early production on a separate stream**: `_prepare_ngram_context` runs inside
  `prepare_inputs`, before dispatch (`model_state.py:98-125`), reading
  `req_states.num_computed_tokens.gpu` and `req_states.all_token_ids.gpu`. Those are updated
  by the previous step's output processing, so an early producer queues behind the same tail.
- **Eager execution**: 17.3 vs 147 tok/s, 8.5 times slower. Not a route.

## 3. The observation we want you to test

The host appears to already hold every input the ids need, and to hold it before the graph
runs. Step N's ids depend on: `input_ids` (known at schedule), `query_start_loc` (known at
schedule), and the ngram context, which is the last `ngram_size - 1` tokens of each request.
Under MTP the accepted set for step N-1 is decided by the target verify on the device, but the
host learns it from the sampler output, and that arrives before the host schedules step N:
`update_from_output` then `schedule` then `prepare_inputs` then replay. So at the moment step
N's prefetch is issued, the host has step N-1's accepted tokens in hand.

If that is right, step N's ids could be computed on the host directly into the pinned `_ids`
buffer at schedule or prepare time, the SSD read could begin at submit and overlap the whole
step, and `_finalize_prefetch`'s `self._pending.result()` wait would collapse. The GPU ids
producer, the preserve copy and the D2H round trip would all leave the critical path. The
current decode step is about 18.4 ms with submit p50 17.34 ms, so this is worth up to roughly
10 percent on depth 2, and more at depth 4 where submit p50 is 21.8 ms.

Questions:

1. Is the claim "the host holds all ids inputs before the replay" true, or does some part of
   the ids depend on device state that only exists inside the step? Specifically: what does
   the Triton `ple_ngram_ids` path read that the host does not already have?
2. If it holds, is the right scope **host ids for decode, GPU ids for prefill**, rather than a
   full replacement? What is the smallest surface that keeps the two paths agreeing?
3. What must be proven, concretely, for MTP acceptance (accepted and rejected draft tokens),
   request packing and padding, EOS history, ring/offset wraparound, request reuse, and
   chunked prefill boundaries? We want local invariants and a bitwise comparison plan, not a
   final-token-string test.
4. If the ids round trip disappears, what is the minimal correct structure that still protects
   the rows buffer against write-after-read reuse? Today `wait_stream(current)` appears to
   provide that; without the ids hop we would keep per-slot events instead. Give us the
   ordering you would accept, and what could still go wrong.
5. Would you do this before or after the fixed-request code-weighted x2/x3/x4 battle, and
   before or after the bounded early-rows oracle priced on x4? Rank three moves by expected
   value per unit cost, counting a restart at ~4 minutes and a measurement block at ~75
   seconds.
6. Is there a cheaper experiment that would falsify or confirm section 3 without building it?
   We would accept one restart plus one block if the payoff is real.

## 4. Standing numbers

Depth ladder, byte-identical except `num_speculative_tokens`: x2 median 149.0 tok/s (step
18.4 ms, submit p50 17.34 ms, block p50 0.10 ms, GPU util 40.2%), x3 155.8 (+4.2%), x4 175.7
(+17.5% over x2, step 22.7 ms, submit p50 21.8 ms, util 39.8%); x5 fails at startup on
`QSA ring capacity 12 must divide the attention block size 1616`. PLE-free structural view at
x2: 165.0 tok/s, submit p50 5.08 ms, block p50 10.85 ms, accept length and rate aligned with
the full-weight run, util 45.2%.

Corrected PLE window counters: 48.0/53.6 rows per lookup, unique-row hits 95.8-99.6%,
all-hit lookups 83.5-98.5%, misses per lookup p50 = 0, lookahead 0. So a GPU hot-row cache is
dead and the independence arithmetic from round 4 is void.

Warm-up discipline we are holding to: at least 2500 decode tokens before a block counts; hard
gate that `load_weights` frames are absent in repeated `py-spy` samples and never return; two
consecutive blocks within 2 percent to declare the drift stopped; discard one whole block
after stopping; only later blocks enter the median.

Tracing overhead is measured at about zero: with tracing on, blocks gave 149.5 / 149.0 / 147.8
and 146.5 / 148.1 / 148.5 against uninstrumented 149.5 / 148.8.
