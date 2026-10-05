# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Decode-throughput probe for a running Flash-Next service.

Measures tokens/second and time-per-output-token for a long greedy generation,
both alone (batch 1) and with the full batch (max_num_seqs) in flight, which is
where an acceptance-adaptive draft count is supposed to show a difference: a
step that verifies fewer drafts does less attention and sampling work.

Usage:
    python probe_decode_tps.py --base http://127.0.0.1:9394/v1 [--tokens 512]
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _ask(
    base: str, prompt: str, min_tokens: int, stream: bool
) -> tuple[int, float, float]:
    """Return (token count, time-to-first-token, total time) in seconds."""
    payload = {
        "model": "Qwen3.8-Flash-Next",
        "prompt": prompt,
        "max_tokens": min_tokens + 64,
        "min_tokens": min_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": stream,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=600) as fh:
        if not stream:
            body = json.loads(fh.read().decode("utf-8"))
            total = time.perf_counter() - t0
            usage = body.get("usage") or {}
            return int(usage.get("completion_tokens", 0)), total, total
        for raw in fh:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_chunks += 1
    return n_chunks, ttft or 0.0, time.perf_counter() - t0


def _run(base: str, prompt: str, min_tokens: int, concurrency: int) -> dict:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        t0 = time.perf_counter()
        results = list(
            pool.map(lambda _: _ask(base, prompt, min_tokens, True), range(concurrency))
        )
    wall = time.perf_counter() - t0
    tokens = sum(r[0] for r in results)
    return {
        "requests": concurrency,
        "tokens": tokens,
        "wall_s": wall,
        "tok_per_s": tokens / wall,
        "tpot_ms": 1000.0 * (results[0][2] - results[0][1]) / max(results[0][0] - 1, 1),
        "ttft_s": results[0][1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:9394/v1")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    prompt = (
        "List every reason a shipment could be delayed, one per line, "
        "as thoroughly as possible. "
    )

    for trial in range(args.trials):
        for concurrency in (1, 4):
            r = _run(args.base, prompt, args.tokens, concurrency)
            print(
                f"[trial {trial} batch {concurrency}] tokens={r['tokens']} "
                f"wall={r['wall_s']:.2f}s tok/s={r['tok_per_s']:.1f} "
                f"TPOT={r['tpot_ms']:.1f}ms TTFT={r['ttft_s']:.2f}s"
            )


if __name__ == "__main__":
    main()
