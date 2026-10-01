#!/usr/bin/env python3
"""Summarise vLLM's per-iteration detail lines out of a serve log.

``--enable-logging-iteration-details`` prints one line per engine step with the
intra-step wall time. That is the number the throughput rate does not give us:
rate tells us the cadence, these lines tell us how much of a step happens
inside ``execute_model`` and how much is left over between steps.

Decode-only iterations (no context tokens) are separated from prefill
iterations, and the leftovers are reported as percentiles.
"""
import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms")


def percentile(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(int(round(q * (len(s) - 1))), len(s) - 1)
    return s[k]


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_bytes().decode("utf-16-le", errors="replace")
    decode_ms: list[float] = []
    decode_tok: list[int] = []
    prefill_ms: list[float] = []
    mixed_ms: list[float] = []
    best_idx = best_ms = 0.0

    for m in PATTERN.finditer(text):
        idx, ctx_req, ctx_tok, gen_req, gen_tok, elapsed = m.groups()
        elapsed = float(elapsed)
        ctx_tok, gen_tok = int(ctx_tok), int(gen_tok)
        if ctx_tok == 0 and gen_tok > 0:
            decode_ms.append(elapsed)
            decode_tok.append(gen_tok)
        elif ctx_tok > 0 and gen_tok == 0:
            prefill_ms.append(elapsed)
        else:
            mixed_ms.append(elapsed)
        if elapsed > best_ms:
            best_ms, best_idx = elapsed, int(idx)

    n = len(decode_ms)
    print(f"decode-only iterations: {n}")
    if n:
        mean = sum(decode_ms) / n
        mean_tok = sum(decode_tok) / n
        print(f"  intra-step ms: p50={percentile(decode_ms, .5):.2f} "
              f"p90={percentile(decode_ms, .9):.2f} "
              f"p99={percentile(decode_ms, .99):.2f} max={max(decode_ms):.2f} "
              f"mean={mean:.2f}")
        print(f"  tokens/step: mean={mean_tok:.2f} "
              f"min={min(decode_tok)} max={max(decode_tok)}")
        implied = mean_tok / mean
        print(f"  implied decode rate from intra-step time alone: "
              f"{implied:.1f} tok/s")
    if prefill_ms:
        print(f"prefill iterations: {len(prefill_ms)} "
              f"p50={percentile(prefill_ms, .5):.2f} ms "
              f"max={max(prefill_ms):.2f} ms")
    if mixed_ms:
        print(f"mixed iterations: {len(mixed_ms)} "
              f"p50={percentile(mixed_ms, .5):.2f} ms")
    print(f"slowest single step: {best_ms:.2f} ms at iteration {best_idx}")


if __name__ == "__main__":
    main()
