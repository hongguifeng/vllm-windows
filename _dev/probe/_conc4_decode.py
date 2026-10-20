"""Steady-state c=4 decode probe with exact token counts.

Four concurrent short-prompt requests, 128 new tokens each, ignore_eos. Reads
usage.completion_tokens so MTP's multi-token chunks cannot inflate the count.
Prefill is ~200 tokens, so the wall time is decode-dominated.

  python _dev/probe/_conc4_decode.py --port 9401 --runs 3
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request
import uuid

FILLER = "benchmark filler sentence about graphics cards and memory bandwidth "


def post(port: int, model: str, prompt: str, max_tokens: int) -> tuple[float, int]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
    return time.perf_counter() - t0, usage.get("completion_tokens", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9401)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    base = (FILLER * 40)[:1600]
    post(args.port, args.model, "warmup", 8)  # one cold engine pass

    for run in range(args.runs):
        results: list[tuple[float, int]] = []
        lock = threading.Lock()

        def worker() -> None:
            prompt = f"[{uuid.uuid4().hex[:8]}] " + base
            out = post(args.port, args.model, prompt, args.max_tokens)
            with lock:
                results.append(out)

        threads = [threading.Thread(target=worker) for _ in range(args.conc)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total = sum(n for _, n in results)
        per_req = [n / dt for dt, n in results]
        print(
            f"[c{args.conc} run{run + 1}] wall={wall:.3f}s tokens={total} "
            f"aggregate={total / wall:.1f} tok/s per-req={[round(x, 1) for x in per_req]} "
            f"median-per-req={statistics.median(per_req):.1f} tok/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
