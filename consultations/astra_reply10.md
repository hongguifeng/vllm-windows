# Single-stream stopping review — reply 10

No. Under the explicit “stop if the remaining opportunity is about 1 ms” rule, another instrumented restart or kernel rewrite is not justified.

The practical result is already strong:

- Buffered IOCP reduced large-prefill latency from 4.23/15.31/67.60 seconds to 0.719/2.053/9.475 seconds.
- Whole-shard reads are correct and fast after warmup.
- Current Windows decode is broadly comparable to historical WSL: 18.0–19.0 ms versus 17.3–18.6 ms chunk intervals, with overlapping throughput and acceptance ranges.
- The comparison is not an exact paired experiment, so it cannot prove parity, but it also does not establish a material regression.
- The tested low-level opportunities are either already enabled, below the decision threshold, or negative in production-like controls. GDN mask slicing reduced metadata time by about 0.135 ms while lowering throughput; deeper MTP helped synthetic text but only 0.3% on mixed production data.

The gap to historical WSL is therefore uncertain and probably small. The best apparent difference is roughly 1 ms per 18.5 ms step, equivalent to about 5.7% throughput or 0.23 seconds over 512 output tokens at the stated rate. That is a useful theoretical sensitivity, but the current measurements do not show that Windows has a stable, recoverable 1 ms deficit. The workloads and configuration snapshots differ.

The theoretical improvement ceiling is also unknown, but the available evidence does not support a cheap path to a larger gain. Existing profiler results lack trustworthy CUDA activity, some prior timings were scaled or mixed across generations, and recorded metadata waits overlapped GPU execution. Those facts make a kernel rewrite speculative rather than evidence-led.

Recommended order:

1. **Stop optimization work now.** Keep the buffered IOCP and prefill-root fixes.
2. **Optionally run one clean C1 replication without a profiler** if a final paired Windows-versus-WSL decision is needed. Match prompt set, configuration snapshot, warmup, output length, and acceptance measurement. This is for confidence, not optimization.
3. **Reopen investigation only if C1 shows a repeatable gap exceeding about 1 ms per step or 5–6% throughput**, with acceptance and workload held constant, or if a new trace directly identifies a critical-path operation with at least 1 ms recoverable cost and low implementation risk.

Do not spend a restart, cache flush, driver change, new profiler setup, deeper MTP experiment, or kernel rewrite on the current evidence.
