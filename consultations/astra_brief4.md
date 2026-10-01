# Round 4 for reviewer (gpt-6): a correction - the denominator we gave you was wrong by 7x

Same rules: read-only, no restarts, no benchmarks, GPU0 untouched (still 62,277 MiB
of the user's own engine). Answer in English, quantitatively.

First: all three of your round-3 source claims were confirmed against the deployed
copy, and we fixed the first one.

- `table.hits` and `table.reads` were never cleared while the timing counters were,
  so the logged percentage described the whole process. Now every counter resets with
  the report.
- `np.unique(flat)` became `np.unique(flat, return_counts=True)`, so we report
  distinct-row and row-occurrence hit rates separately.
- Read-ahead now counts separately from demanded lookups.
- Worktree `f33da54841`; deployed copy synced and compiling; the `.orig` backup is
  untouched.

## 1. Corrected per-window numbers (MTP x2, full weights + PLE SSD)

Warm block discarded, then two measured blocks: 147.1 / **146.5 / 151.1**, median
148.8, which matches the 149.5 in section 19 - the timing accounting is unchanged.
What changed is what the PLE line means:

```
PLE decode window: 200 lookups, 53.6 rows per lookup, 99.6% unique hits,
99.6% row hits, 98.5% all-hit lookups, miss p50/p95/p99 0/0/8, 0 lookahead lookups
```

| Per window of 200 steps | Range observed |
|---|---|
| Rows per lookup | **48.0 / 53.6**, alternating between two step shapes |
| Distinct-row hit rate | 95.8-99.6% |
| Row-occurrence hit rate | same, to within 0.1 pt |
| **Lookups needing no disk read** | **83.5-98.5%** |
| Missing rows per lookup: p50 / p95 / p99 | **0** / 0-16 / 8-40 |
| Read-ahead lookups in the window | 0 |
| ids wait / row assembly | 10.7-10.9 ms / 0.33-0.67 ms |

The "350-520 rows per step" we gave you in round 3 was cumulative counters divided by
the report interval, and it is wrong by roughly a factor of seven. Your conclusion
that "a 90% all-hit probability requires 99.97-99.98% per-row hits" was computed for
350-520 rows and does not apply at 50 rows. Measured directly: the median step needs
**no** disk read at all, and 83.5-98.5% of lookups need none.

## 2. Questions

**Q6.1 Does the device row-cache demotion change, and does the oracle design change?**
Our reading is that a device cache attacks 0.33-0.67 ms per step - about 3% at 18.3 ms -
while the 10.7-10.9 ms ids rendezvous does not care where rows come from. Do you accept
that, or does the corrected hit structure reopen it? Separately: at ~50 rows of 320 bytes,
a preassembled payload for one lookup is roughly 16 KB. Does that change your oracle
design (three arms, late versus early as the causal pair) or your stop gate?

**Q6.2 What warm-up do you require, given memory-mapped weights?**
Default `safetensors_load_strategy=None` is memory-mapped lazy loading
(`config/load.py:70`); the shard progress in the startup log is only the header pass.
The 143 GiB page-in is paid on first touch: our first request after startup ran 72.5 s
at about 12 tok/s, and py-spy showed the EngineCore main thread still inside
`load_weights`, moving from `fc_embedding.qweight` to fused_moe routed experts, with GPU
memory climbing a few hundred MiB per minute. It was progressing, not deadlocked. The
next startup took 4.5 minutes because the OS page cache was warm - weight loading uses
ordinary buffered reads, unlike our PLE reads which use NO_BUFFERING.

Two things we can observe cheaply: py-spy `load_weights` frame count reaching zero, and
request throughput stabilizing. Would you accept "warm until the loader frames reach zero,
then discard one traffic block" as the campaign's warm-up rule? And do you accept that
today's x2 blocks rising 144.4 to 149.5 to 152.2 were partly page-in rather than cache
warming?

**Q6.3 Tell us exactly which timestamps to record so the 10.7 ms is unambiguous.**
Your round-3 matrix said that if producer to preservation to D2H is already tight, the
event substitution is dead. We now know the median step reads nothing from disk, so the
wait is unlikely to be disk time. Where do you expect the 10.7 ms to sit: waiting for the
GPU to reach the ids kernel, the side-stream ordering behind the main stream's queued
work, or somewhere else? Specify the marks you want - we propose: ids kernel enqueue,
producer completion, preservation copy completion, side-stream becoming waitable after
`_previous_use`, D2H start and end, host wakeup, rows ready, H2D completion, consumer
arrival - using preallocated CUDA event rings with no per-step synchronization to read
them. Say which of these you consider load-bearing and which you would drop.

**Q6.4 One protocol confirmation.**
We intend to run the campaign with lazy loading plus a discarded warm block, and to
report medians only from blocks after page-in completes. Any objection?

## 3. One operational fact you should know for the campaign's cost model

A command that exceeds the tool timeout kills the PowerShell log relay while leaving the
Python process alive holding 64 GiB of device memory: the server becomes a zombie that
answers `/v1/models` and hangs every completion. Every cleanup goes through
`_dev/bin/_stop_vllm.ps1`, and every command must finish well inside the timeout. The
real engine log is `_dev/out/logs/flashnext_<timestamp>.log`, written by the serve script
itself; the redirected stdout/stderr files are only a relay.
