# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fresh-prefill TTFT probe against a running service (cache-busting).

Usage:
    python _dev/probe/_prefill_ttft_probe.py --base http://127.0.0.1:9393/v1
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def post(url: str, payload: dict, timeout: float = 1800.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def count_tokens(root: str, model: str, text: str) -> int:
    body = post(f"{root}/tokenize", {"model": model, "prompt": text}, timeout=120.0)
    return body.get("count", len(body.get("tokens", [])))


def make_prompt(target: int, nonce: str) -> str:
    """Deterministic filler with a unique nonce so prefix cache cannot hit."""
    head = f"[{nonce}] Read the catalog and answer with one word.\n"
    body = " ".join(f"row{i}x{nonce[:4]}" for i in range(target * 2))
    return head + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:9393/v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--sizes", default="256,2048,8192")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base

    print(f"[probe] target {base}")
    for size in (int(s) for s in args.sizes.split(",")):
        nonce = f"n{time.time_ns() % 10**9}"
        prompt = make_prompt(size, nonce)
        ntok = count_tokens(root, args.model, prompt)
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
        prompt_tokens = out.get("usage", {}).get("prompt_tokens", ntok)
        rate = prompt_tokens / dt if dt > 0 else float("nan")
        print(
            f"[prefill] requested={size:6d} actual={prompt_tokens:6d} tok  "
            f"ttft={dt:8.3f}s  prefill={rate:9.1f} tok/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
