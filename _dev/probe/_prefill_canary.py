# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Allocator-poisoning canary: repeated fresh N-token prefills on one engine.

Healthy one-chunk (2048 tok) prefill on this box is ~0.13-0.20 s. The poisoned
signature is 1.2-2.5 s per chunk (engine-reported prefill in the hundreds of
tok/s) that does not bounce back.

Usage:
    python _dev/probe/_prefill_canary.py --base http://127.0.0.1:9393/v1 \
        --tokens 2048 -n 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

WORDS = [f"tok{i}" for i in range(4096)]


def post(url: str, payload: dict, timeout: float = 1800.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def build(nwords: int, nonce: str) -> str:
    words = WORDS[:nwords]
    return f"[{nonce}]\n" + " ".join(words) + "\nEnd."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:9393/v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base

    def ntok(text: str) -> int:
        body = post(f"{root}/tokenize", {"model": args.model, "prompt": text}, 120.0)
        return body.get("count", len(body.get("tokens", [])))

    nwords = args.tokens
    for _ in range(3):
        got = ntok(build(nwords, "calib"))
        if got == args.tokens:
            break
        nwords = max(1, int(nwords * args.tokens / got))

    print(f"[canary] target={args.tokens} filler_words={nwords}")

    rates: list[float] = []
    for i in range(args.n):
        prompt = build(nwords, f"c{i}-{time.time_ns() % 10**9}")
        actual = ntok(prompt)
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        t0 = time.perf_counter()
        out = post(f"{base}/completions", payload)
        dt = time.perf_counter() - t0
        ptok = out.get("usage", {}).get("prompt_tokens", actual)
        rate = ptok / dt
        rates.append(rate)
        print(
            f"[canary {i}] tokens={ptok:6d}  ttft={dt:7.3f}s  prefill={rate:9.1f} tok/s"
        )
    print(
        f"[canary] median={statistics.median(rates):.1f} tok/s  "
        f"min={min(rates):.1f}  max={max(rates):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
