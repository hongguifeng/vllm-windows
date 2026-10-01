#!/usr/bin/env python3
"""Drive one stream at a time for a fixed wall-clock window.

A single request stops on EOS after ~500 tokens, which is too short to sample a
30 s profiling window, and 256-token bursts carry +-5% run-to-run noise. This
issues requests back to back on one stream only, then reports the steady-state
rate over the whole window, which is far less noisy than per-request numbers.
"""
import argparse
import json
import time
import urllib.request

MODEL = "qwen3.8-flash-next-full"


def ask(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(base, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        ch = data.get("choices", [{}])[0]
        return {
            "finish": ch.get("finish_reason"),
            "completion": data.get("usage", {}).get("completion_tokens"),
            "prompt": data.get("usage", {}).get("prompt_tokens"),
            "wall": time.time() - t0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc), "completion": 0, "wall": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--window", type=float, default=30.0,
                    help="seconds of steady-state traffic to produce")
    ap.add_argument("--tokens", type=int, default=700,
                    help="max_tokens per request")
    ap.add_argument("--skip", type=int, default=2,
                    help="requests excluded from the steady-state numbers")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}/v1/chat/completions"

    results = []
    t_start = time.time()
    k = 0
    while time.time() - t_start < args.window:
        r = ask(base, args.model,
                f"List every integer from {9000 - k * 7} down to "
                f"{9000 - k * 7 - 600}, one per line.", args.tokens)
        results.append(r)
        k += 1

    total_tokens = sum(r.get("completion", 0) for r in results)
    total_wall = sum(r.get("wall", 0) for r in results)
    print(f"requests={len(results)} tokens={total_tokens} "
          f"sum(wall)={total_wall:.1f}s span={time.time() - t_start:.1f}s")
    steady = results[args.skip:]
    steady_tokens = sum(r.get("completion", 0) for r in steady)
    steady_wall = sum(r.get("wall", 0.0) for r in steady)
    if steady_tokens and steady_wall:
        print(f"steady-state (dropping first {args.skip}): "
              f"{steady_tokens / steady_wall:.1f} tok/s "
              f"over {steady_tokens} tokens / {steady_wall:.1f}s")
    for r in results:
        if r.get("error"):
            print("  error:", r["error"])
        else:
            print(f"  finish={r['finish']} completion={r['completion']} "
                  f"prompt={r['prompt']} wall={r['wall']:.2f}s "
                  f"rate={r['completion'] / r['wall']:.1f}")


if __name__ == "__main__":
    main()
