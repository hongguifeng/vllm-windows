# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Functional probe against a running Flash-Next service.

Measures TTFT for a repeated long prompt (prefix caching + the A1 prompt-tail
hash fix), an image prompt twice (the A4/A5 multimodal hashing fix), and a plain
chat for output sanity (A2 PLE buffer + A3 vocab clamp).

Usage (pybase64 comes from the vLLM install, so the bare interpreter cannot run
this):
    .venv-flashnext/Scripts/python.exe _dev/fork_cmp170hx/probe_service.py \
        --base http://127.0.0.1:9394/v1
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pybase64 as base64

REPO = Path(__file__).resolve().parents[2]


def post(base: str, payload: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def streamed_ttft(base: str, payload: dict) -> tuple[float, int, str]:
    """Return (seconds to first content chunk, completion tokens, text)."""
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    t0 = time.perf_counter()
    first = None
    text = []
    completion_tokens = 0
    with post(base, payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            if delta.get("content"):
                if first is None:
                    first = time.perf_counter() - t0
                text.append(delta["content"])
            completion_tokens = chunk.get("usage", {}).get(
                "completion_tokens", completion_tokens
            )
    if first is None:
        first = time.perf_counter() - t0
    return first, completion_tokens, "".join(text)


def long_prompt(words: int = 900) -> str:
    filler = " ".join(f"item{i}" for i in range(words))
    return (
        "Summarize the inventory list below in one short sentence. "
        "Do not add commentary.\n\n" + filler + "\n\nEnd of list."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default="http://127.0.0.1:9394/v1", help="OpenAI-compatible root"
    )
    parser.add_argument("--image", default=str(REPO / "_dev/test/_vision_sample.png"))
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"[probe] target {base}")

    # 1. Plain chat: output sanity after the PLE buffer and vocab-clamp changes.
    payload = {
        "messages": [{"role": "user", "content": "Reply with exactly: PING-7341"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    ttft, ntok, text = streamed_ttft(base, payload)
    print(f"[chat ] ttft={ttft:.3f}s tokens={ntok} text={text!r}")

    # 2. Long prompt repeated: prefix caching (A1 restores the prompt-tail block).
    prompt = long_prompt()
    results = []
    for i in range(3):
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 24,
            "temperature": 0.0,
        }
        ttft, ntok, text = streamed_ttft(base, payload)
        results.append((ttft, ntok, text))
        print(f"[long {i}] ttft={ttft:.3f}s tokens={ntok} text={text[:60]!r}")
    same = all(r[2] == results[0][2] for r in results)
    print(f"[long ] identical output across repeats: {same}")

    # 3. Image prompt twice: multimodal block hashing (A4/A5).
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[image] skipped, no {img_path}")
        return 0
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    img_url = f"data:image/png;base64,{b64}"
    img_results = []
    for i in range(2):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {
                            "type": "text",
                            "text": "What is in this image? One sentence.",
                        },
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        ttft, ntok, text = streamed_ttft(base, payload)
        img_results.append((ttft, ntok, text))
        print(f"[image{i}] ttft={ttft:.3f}s tokens={ntok} text={text[:80]!r}")
    print(
        f"[image] identical output across repeats: "
        f"{all(r[2] == img_results[0][2] for r in img_results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
