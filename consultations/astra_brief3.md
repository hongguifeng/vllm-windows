# Round 3 for reviewer (gpt-6): the PLE stall now has a price, and MTP depth turned out to be a free lever

Same rules: read-only, no restarts, no benchmarks, GPU0 is the user's and was never
touched. Answer in English, quantitatively, ranked by expected value per unit cost,
and give decision matrices where asked. Round 1 and round 2 replies are on disk and
were acted on; this round reports what your advice produced and asks six questions.

## 1. What we ran since round 2

All measurements are single-stream (`_c1_traffic.py`, fixed prompt family "list every
integer from N down to N-600, one per line", temperature 0, `--tokens 900 --skip 3
--window 70`, three steady blocks per configuration), same host, free RAM 43.6-44.2
GiB during every block, fresh service initializations, and every configuration pair
was verified by diffing the printed `non-default args` line.

### 1.1 The PLE-free screen (your Q4.4 objection noted)

| at MTP x2 | three blocks | median | acceptance length | avg draft acceptance | submit p50 | block p50 |
|---|---|---|---|---|---|---|
| structural (PLE-free) view | 165.0 / 164.7 / 166.7 | **165.0** | 2.68-2.83 | 83.5-91.3% | **5.08 ms** | **10.85 ms** |
| full weights + PLE SSD | 144.4 / 149.5 / 152.2 | **149.5** | 2.67-2.83 | 83.3-89.8% | 17.34 ms | 0.10 ms |

Acceptance length and acceptance rate match block for block, so this gap is
implementation, not workload. Net difference **+10.4%** by median, +11.0% by mean.
We accept your label: this is a screen, not a ceiling, because the structural view also
changes weights, memory layout and graph state (its GPU KV cache reports 87,552 tokens,
21.38x concurrency at 4,096 tokens per request).

### 1.2 The pacer changes hands

- Full model x2: submit p50 **17.34 ms**, block p50 **0.10 ms**. The CPU submits for
  17 ms; the GPU round trip is entirely hidden underneath it.
- PLE-free: submit p50 **5.08 ms**, block p50 **10.85 ms**. Once the CPU gets fast, the
  GPU round trip becomes visible for the first time.
- PLE's own account, same host same day: `wait 11.25-11.30 ms/step`, of which ids wait
  10.8-10.9 ms, row assembly 0.21-0.38 ms, cache hits 93.1-93.2%.

So 11 of the 17.34 ms submit window - **63%** - is the PLE ids wait. That is why
removing PLE sells 10% and removing dispatch glue sold nothing.

### 1.3 Device utilization (480 nvidia-smi samples per configuration)

| configuration | mean GPU util | peak |
|---|---|---|
| PLE-free x2 | 45.2% | 99% |
| full + PLE x2 | 40.2% | 92% |
| full + PLE x3 | 39.9% | 93% |
| full + PLE x4 | 39.8% | 93% |

The device is idle roughly sixty percent of the time at every depth. Single-stream
decode here is not device-throughput bound; step time is set by the CPU submit path.

### 1.4 Depth ladder (configs differ in `num_speculative_tokens` only)

| depth | three blocks | median | gain by median | acceptance length | per-position acceptance | step time = acc/median | submit p50 | PLE wait / ids | cache hits |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 144.4 / 149.5 / 152.2 | 149.5 | - | 2.67-2.83 | 0.88-0.93, 0.77-0.88 | 18.3 ms | 17.34 | 11.3 / 10.9 | 93.1-93.2% |
| 3 | 155.7 / 160.5 / 155.8 | 155.8 | +4.2% | 3.16-3.55 | 0.83-0.91, 0.71-0.86, 0.61-0.78 | 21.8 ms | 20.1 | 13.5 / 12.7 | 92.9-93.1% |
| 4 | 175.7 / 171.4 / 179.0 | **175.7** | **+17.5% vs x2** | 3.75-4.49 | 0.82-0.94, 0.72-0.90, 0.63-0.85, 0.57-0.79 | 22.7 ms | 21.8 | 14.3 / 13.9 | 94.0-94.1% |
| 5 | dies at startup | - | - | - | - | - | - | - | - |

Depth five fails before the engine starts:

```
File ".../models/qwen4_exp/common/qsa_cache.py", line 846, in get_kv_cache_spec
    assert self.cache_config.block_size % capacity == 0, (
AssertionError: QSA ring capacity 12 must divide the attention block size 1616
```

Two things are simultaneously true, and your round-2 model predicted the first of them:
the ids wait lengthens with depth (+1.3 ms per step of depth, because more main-stream
work is queued ahead of the ids producer), and each step buys more accepted tokens,
which divides that fixed stall out. Depth four is therefore currently the cheapest way
to amortize the PLE stall: +17.5% for a flag change.

## 2. Questions

**Q5.1 Does the amortization result change your ranking?**
Your round-2 top action was the mixed-traffic MTP campaign, and you demoted dispatch
work. We now have a third mechanism you did not consider: a fixed per-step CPU stall
that deeper speculation amortizes. Please re-rank the remaining actions with that in
mind, and tell us what you would add to the campaign now that the comparison is x2 vs
x3 vs x4 rather than x2 vs x1.

**Q5.2 Make the oracle experiment concrete and cheap.**
We accept that a matched intervention is what prices the ids dependency. Given the
structure of our code (`start_prefetch` in `models/qwen4_exp/nvidia/ple_ssd.py`, which
you quoted correctly last round), specify the smallest intervention that makes the same
rows available earlier without changing generated tokens: is a "rows delivered by a
background warm cache, ids ignored" variant legitimate, or does ignoring ids change the
consumed values? What residual differences must we enumerate, and what threshold on the
matched delta kills the PLE redesign? Our prior: if the matched delta is below 5%, stop.

**Q5.3 Depth as a production default: what do you want measured?**
Our gate would now be: mixed-set decode gain of x4 over x2 at least 10%, paired 95% CI
lower bound above 5%, no class with a repeatable 5% regression. Two questions:
(a) three configurations (x2, x3, x4) or two (x2, x4)? Our startup cost is ~4.5 minutes
per configuration, so three is affordable.
(b) deeper drafts waste more GPU work when acceptance collapses. What regression metric
do you want besides decode rate: drafted-but-rejected tokens per step, TTFT, time to
last token, or something else? Note that with utilization at 40%, wasted draft work may
be free in wall-clock terms - do you accept that argument or demand the accounting?

**Q5.4 Is the depth-5 layout assertion worth fixing?**
`ring capacity` appears to grow with speculative depth while `block_size` stays at 1616,
and the assertion requires exact divisibility. Can you derive the legal depth set from
that relationship (which capacities divide 1616 = 2^4 x 101) so we know whether depth 6
or 7 is even reachable, or should we treat depth <= 4 as the design envelope and stop?
We do not want to change block size blind.

**Q5.5 Can idle device capacity hide the ids stall?**
The device is idle ~60% of the time. Would you consider producing the ngram ids
speculatively - before the MTP draft that determines them completes - and correcting
afterwards? If yes, what would you require as a correctness gate, given that greedy
output here is not reproducible even for the same request in the same server (we
measured that in round 2)? If no, say so plainly.

**Q5.6 What is the attainable ceiling if we remove only the PLE CPU wait?**
The structural view removes the CPU wait, the row retrieval, and PLE's GPU work, and
lands at 16.5 ms per step with block p50 10.85 ms. If we keep PLE's GPU work and remove
only its CPU-side stall, what do you estimate for step time and throughput? We want to
know whether to expect the full +10%, a fraction of it, or a hard floor near 16 ms per
step caused by the round trip the structural view just exposed.

**Q5.7 One more thing we did not tell you before.**
Our host-cache hit rate today is 93.1-94.1%, not the 77-80% we reported in round 2.
The disk-configuration alignment happened before round 2, so we do not know which part
of the earlier number was real. Does that change your judgment about where PLE's cost
sits, and does it change what you want traced?

## 3. Artifacts added since round 2

- `D:\code\vllm-windows\_dev\docs\FLASH_NEXT_ROUTE_A_BUILD.md` section 19 (this data)
- `D:\code\vllm-windows\_dev\probe\_log_decode.py` (host logs mix an ANSI prefix with
  UTF-16LE; `iconv` alone garbles them and `grep -a` misses the steady-state lines)
- `_dev\out\{struct_mtp2,mtp3,mtp4,mtp2b}_out.txt` and the matching `*_util.txt` samples
- git: `67cdc45704` (documentation), `18ee06b3e9` (36 tooling and probe files),
  `99af724b34` (round 1 and 2 record), `7f3c75aa58` (reader boundary and strict fallback)
