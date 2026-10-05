# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Streaming decode ITL probe (clean C=1 reference for this rig: ITL50 ~20 ms).

Usage:
    python _dev/probe/_itl_canary.py --base http://127.0.0.1:9393/v1 -n 64
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:9393/v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument(
        "--prompt", default="List every integer from 1 to 200, one per line."
    )
    parser.add_argument("-n", type=int, default=64)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.n,
        "min_tokens": args.n,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    first = None
    stamps: list[float] = []
    with urllib.request.urlopen(req, timeout=600.0) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.perf_counter()
                if first is None:
                    first = now - t0
                stamps.append(now)

    itl = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
    total = (stamps[-1] - t0) * 1000.0 if stamps else float("nan")
    print(
        f"[itl] ttft={first * 1000.0:.1f} ms  chunks={len(stamps)}  e2e={total:.0f} ms"
    )
    if itl:
        itl_sorted = sorted(itl)
        p = lambda q: itl_sorted[min(len(itl_sorted) - 1, int(q * len(itl_sorted)))]  # noqa: E731
        print(
            f"[itl] median={statistics.median(itl):.1f} ms  p90={p(0.9):.0f}  "
            f"p99={p(0.99):.0f}  max={max(itl):.0f}  (clean C=1 ref: ~20 ms)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
