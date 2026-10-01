#!/usr/bin/env python3
"""Greedy output parity across server restarts.

Sends a fixed set of temperature-0 requests and writes the completions to a
JSON file. Two runs of this against servers that differ only in configuration
should produce byte-identical text; anything else means the configuration
changed the numbers actually being computed. The long prompt is deliberately
repetitive so it crosses prefill chunk boundaries and leans on the PLE path.
"""
import argparse
import json
import sys
from pathlib import Path

import requests

MODEL = "qwen3.8-flash-next-full"

SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the compiler optimizes "
    "the attention kernel and the scheduler batches the tokens into blocks. "
)


def build_cases() -> list[dict]:
    cases = [
        {"tag": "short", "prompt": "What is the capital of France?", "max": 48},
        {"tag": "code", "prompt": "Write a Python function that reverses a "
                                  "singly linked list, with type hints.", "max": 96},
        {"tag": "count", "prompt": "List the prime numbers below 50 as a "
                                   "comma separated line.", "max": 128},
        {"tag": "repeat", "prompt": SENTENCE * 20, "max": 64},
        {"tag": "long", "prompt": SENTENCE * 120, "max": 64},
        {"tag": "qa", "prompt": "Explain in one short paragraph why radix "
                                "attention caches are hard to evict.", "max": 80},
    ]
    return cases


def post(port: int, case: dict) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "temperature": 0,
        "max_tokens": case["max"],
    }
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    return {
        "tag": case["tag"],
        "text": data["choices"][0]["message"]["content"],
        "finish": data["choices"][0].get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    rows = []
    for case in build_cases():
        row = post(args.port, case)
        rows.append(row)
        print(f"{row['tag']:7s} finish={row['finish']} "
              f"prompt={row['prompt_tokens']} completion={row['completion_tokens']} "
              f"chars={len(row['text'])}", flush=True)

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
