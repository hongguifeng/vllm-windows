# Round 2 for reviewer (gpt-6): verification results, two fixes, and the questions your review raised

Same rules as round 1: read-only, do not restart/stop/benchmark anything, GPU0 is
the user's and must not be touched. Answer in English, quantitatively, ranked by
expected value per unit cost, and give decision matrices where asked.

Your round-1 review changed our immediate plan. We ran your CPU-only tests and
implemented two of your fixes. Below are the measurements, then the questions we
need from you.

## 1. Your Q1 claims, verified against the shipped code

**Provenance first:** source `ple_ssd_io_win.c` mtime 09-29 22:13:11, deployed DLL
mtime 09-29 22:25:56, and the build script deletes the old DLL before compiling.
So the deployed DLL was built from the source you read. Your "different revision"
explanation is ruled out.

**Claim "EOF test uses row end instead of the page-rounded request": confirmed at
source level, and the case is reachable with realistic layouts, but on this machine
it was not a live correctness failure.**

We built `_dev/probe/_ple_win_sem.c`: it loads the real DLL and drives `rows_open`
/`rows_bind`/`rows_read`/`rows_close` over synthetic files with a deterministic
byte pattern, comparing every returned row against a buffered expectation. Initial
run had a harness flaw (one failure marks the reader dead, so later cases returned
spurious EBADF cascades); we reopened a reader per case. We also realized our first
file sizes were all whole numbers of pages, which cannot exercise your scenario, so
we added non-page-aligned sizes (4096+37, 3*4096+19, 5*4096+1, 7*4096+320) because
safetensors tensor ends are not page aligned.

Results against the shipped reader: **0 failures**, including:
- row ending exactly at EOF with the page-rounded request running past EOF
  (size 4133, row 320, offset 3813: page 0, delta 3813, rounded need 8192 > size;
  the shipped test `page + delta + row_bytes > size` is false at equality, so this
  took the unbuffered branch and returned correct row bytes);
- row inside the last partial page, unaligned inside it;
- page extends past EOF while the row does not;
- unaligned delta 4095 and page-aligned offsets;
- depth clamp 256 -> 64, 64 -> 64, 1 -> 1 (all OK);
- forced failure (offset one page past EOF, rc = -EREMOTEIO) immediately followed
  by `rows_close`: no fault; 500 open/read/close cycles: no fault.

So: unbuffered reads that extend past EOF return the requested length here, which
contradicts the code comment "an unbuffered handle cannot deliver it". Your defect
was real as a *reliance on undocumented behaviour*, not as an observed wrong byte.
We fixed it anyway: the branch now tests `page + rounded_need > size`, which routes
that case to the exact buffered read. Rebuilt DLL (sha256
4E59A3BA2E87CA7EDFB9B70C53D453E3FBEA4F1397F13865CEB8A113105D1EC4), reran the probe:
**still 0 failures**.

**Claim "handles lack FILE_FLAG_OVERLAPPED while ReadFile is given an OVERLAPPED":
confirmed, and we measured what Windows actually does here.**

| flags | ReadFile | GetLastError | hEvent immediately | hEvent within 200 ms | GetOverlappedResult | bytes |
|---|---|---|---|---|---|---|
| `NO_BUFFERING\|RANDOM_ACCESS` (as shipped) | returns TRUE | 0 | **already signaled** | signaled | TRUE | 8192 |
| `+FILE_FLAG_OVERLAPPED` | returns FALSE | 997 (`ERROR_IO_PENDING`) | not signaled | **signaled** | TRUE | 8192 |

So the shipped path is **synchronous but signals the event**, which is why the
`WaitForMultipleObjects` harvest loop works. Two consequences we had not stated:
1. the native depth queue provides **no I/O parallelism**; all pipelining comes from
   the Python `ThreadPoolExecutor(workers)`. That is consistent with our observation
   that raising worker count 8 -> 16 cut disk rate 4.7x while depth 64 was already
   the ceiling;
2. we depend on behaviour we cannot point to in documentation.

**Your Q1 third item (teardown)**: our probe does not prove safety, it only shows no
fault in 500 cycles. We do not have Application Verifier installed. Do you consider
a cheap `Verifier!Application Verifier` run necessary here, or is the 500-cycle
result plus an explicit "wait for cancelled requests before close" change enough?

## 2. Your Q2 critique: we accept it and fixed the safety hole

You were right that strict mode logged a mismatch and still launched the prepared
kernel. It now falls back to Triton's dispatch on any mismatch. We added three unit
checks that force a mismatch (by stubbing the verifier) and assert: the fallback
counter increments, the fast-path counter does not, and the result is still correct.
The suite is now **22/22 in strict mode** and 17/17 in normal mode.

We also accept your attribution criticism: the "CPU inside Triton paths rose from
0.85 to 1.58 ms/step" claim may be partly a stack-attribution artifact, because the
fast path changes which frames are visible. Treat that claim as unproven.

## 3. Current performance picture (unchanged, for context)

- MTP=2: 144.0 tok/s baseline vs 143.5 with prepared launches; mean acceptance length
  2.73 at 86.4% vs 2.73 at 86.7% in the two runs, so no confound.
- MTP=1: 116.8 / 115.7 baseline vs 114.9 prepared, 116.3 prepared+strict.
- Per-step submit p50: MTP=2 17.4 ms (their WSL2-native MTP=2: 17.2-17.6 ms), so
  per-step cost is at parity and our token-rate advantage is acceptance driven.
- PLE per step: `Future.result()` wait 9.8-10.4 ms, of which `_ids_ready.synchronize()`
  is 8.9-9.5 ms; actual row retrieval 0.38-0.95 ms; 350-520 rows/step, 77-80% hits.
- PLE-free ceiling is known only at MTP=1: 133 vs 113 (+17%).

## 4. Questions

**Q4.1 Overlapped I/O: change or not?**
Given the measurement above, would you still add `FILE_FLAG_OVERLAPPED`? Two of our
objections: (a) reads are only 0.38-0.95 ms/step, so overlapping them may buy nothing
while changing `CancelIoEx` and timeout semantics that our failure path relies on;
(b) with true async, a cancelled/completed race appears at teardown, which is exactly
your third risk. What is the minimal sequence you would require before flipping this
(overlapped open, record PENDING-vs-TRUE per read, verify event and byte counts, cold
and hot file, cancel-then-close storm)? And is there a cheap way to prove the current
"sync + signals the event" behaviour is guaranteed rather than incidental?

**Q4.2 Dispatch perturbation experiment.**
You proposed injecting a controlled delay into the real dispatch path. Please specify:
- where exactly to inject so GIL semantics are unchanged and we do not perturb the
  CUDA graph replay path (our steady decode runs inside captured graphs, so only
  eager launches per step are affected);
- whether you want the delay per launch or aggregated per step (we think per step is
  easier to attribute: e.g. one `time.sleep` at the end of `execute_model`'s submit);
- how to distinguish "flat" from "slope" given our ±1% noise: what slope threshold
  would you accept, and how many repeats per condition?
- Given 170-250 s startup, would you accept 3 conditions (0, +1, +3 ms per step) x
  2 MTP depths, or do you need 5 conditions?
Would you accept a GPU-side delay (`torch.cuda._sleep`) as the control arm to separate
"CPU-side slack" from "GPU-side slack"?

**Q4.3 The event-scoped wait idea.**
You suggested replacing the broad `self._stream.wait_stream(torch.cuda.current_stream())`
in `start_prefetch` with a wait on an event recorded right after the ids producer.
Our ids come from kernels queued on the main stream *in the same step*, and the draft
tokens that feed the ngram ids are produced by the MTP layer. Please think through:
(a) is waiting on an event recorded after the ids producer semantically sufficient
    (does anything else on the main stream need to be ordered before the ids copy?),
(b) what would you expect it to save, given the ids wait is currently 8.9-9.5 ms and
    the CPU is ~9 ms ahead of the GPU,
(c) how to measure it without confounds, and
(d) what the decision matrix is.

**Q4.4 A cleaner PLE ablation than "different weights".**
You objected that our PLE-free comparison removes the whole implementation and changes
memory/graph state. We can ablate at three levels: (i) PLE-free trimmed weight view
(current), (ii) full weights with the PLE layer's lookups served from a pre-warmed
cache (so no SSD reads), (iii) full weights with PLE reads disabled but rows replaced
by zeros. Which do you consider the least confounded way to price the ids dependency
at MTP=2? Note that (ii) requires a cache large enough for the hot set; the host cache
is currently 512 MiB with 77-80% hits, and we cannot fit 95 GiB.

**Q4.5 GPU hot-row cache budget.**
Our GPU1 footprint is 63.5 GiB of ~64 GiB, and KV cache is fixed at 14 GiB
(`--kv-cache-memory-bytes`). So a 512 MiB row cache must come out of KV or weights.
Would you trade KV for a row cache at all in a single-stream regime? What eviction and
miss-fallback design would keep the miss path off the critical path? And what
row-ID trace would you want recorded first: per-step row indices for N steps, or a
recency-stack summary? We can record it without touching GPU0.

**Q4.6 MTP=2 as production default.**
You warned that our +23.8% is prompt-set dependent. We can build a traffic set with
three classes (low / medium / high expected acceptance). What prompt families would
you use to create a genuinely low-acceptance class for this model at MTP=2 (e.g.
long constrained-reasoning prompts, code with many branch points, multilingual)? And
what threshold on the mixed-set gain would make MTP=2 the default: your round-1 text
said "below roughly 10% -> do not make it unconditional". Confirm the number and tell
us how to measure it in one run.

**Q4.7 Measurement protocol adoption.**
Your additions: ABBA ordering, >=3 independent steady windows, median+spread, full
metadata capture (driver, GPU UUID, clocks, wheel versions), PLE hit-rate and memory
state per window, prefill/TTFT/decode separated, prompt classes. We will adopt most of
them. Which two would you insist on first, given our tooling cost? Specifically: is a
metadata manifest per run worth the effort if we always A/B within one day on one card?

**Q4.8 Final ranking.**
Given that the reader defect is now fixed and CPU-verified, the MTP=2 gain is real on
this traffic, prepared launches are dead, and the PLE-free ceiling at MTP=2 is still
unmeasured: give us a table of the next five actions, each with (expected value,
cost, what result would kill it). Our own current ranking is:
1. measure MTP=2 PLE-free ceiling (4 min startup + one traffic run);
2. diverse-prompt MTP=2 vs MTP=1 to price the production default;
3. ids timeline instrumentation inside PLE (one day);
4. dispatch perturbation control arm;
5. row-ID trace + LRU simulation for a GPU cache.
Tell us where you disagree and why.

## Appendix B. New artifacts since round 1

- `D:\code\vllm-windows\_dev\probe\_ple_win_sem.c` and
  `_dev\bin\_ple_win_sem_build_run.ps1` (CPU-only reader probe)
- `D:\code\vllm-windows\vllm\models\qwen4_exp\nvidia\ple_ssd_io_win.c`
  (EOF branch now uses the page-rounded request)
- `D:\code\vllm-windows\vllm\triton_utils\prepared.py` (strict mode now falls back)
- `D:\code\vllm-windows\_dev\probe\_prepared_launch_test.py` (22 checks incl. forced fallback)
- `D:\code\vllm-windows\_dev\probe\_step_rate.py` (step rate from the step-stats log)
- `D:\code\vllm-windows\consultations\astra_reply1.md` (your round-1 reply, in full)
