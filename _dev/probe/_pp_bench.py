#!/usr/bin/env python3
"""Decode benchmark for the PP engine: c1/c2/c4/c8 + cold prefill.

Streams each request, so decode rate excludes prefill: (n_tokens - 1) /
(last_token_time - first_token_time). A random salt keeps prefix caching from
turning later runs into cache hits.
"""
import json
import random
import string
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next-w4a16"
TOKENS = 400

PROMPT = (
    "Write a detailed technical explanation, in plain prose, of how a modern "
    "GPU executes a matrix multiplication kernel: describe the memory "
    "hierarchy, tiling, the role of shared memory, and why occupancy matters. "
    "Do not use lists, do not stop early, and keep going for a long time."
)


def run(salt: str, tokens: int = TOKENS) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{PROMPT}\n\n[run {salt}]"}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    t_first = None
    t_last = None
    n = 0
    usage_n = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
            u = obj.get("usage")
            if u and u.get("completion_tokens"):
                usage_n = u["completion_tokens"]
    t_end = time.time()
    n = usage_n or n
    decode = (n - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first else 0.0
    return {
        "tokens": n,
        "ttft": (t_first - t0) if t_first else None,
        "wall": t_end - t0,
        "decode": decode,
    }


def level(concurrency: int, salt: str) -> None:
    results = [None] * concurrency
    def worker(i: int) -> None:
        results[i] = run(f"{salt}-{concurrency}-{i}")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = [r for r in results if r]
    total = sum(r["tokens"] for r in ok)
    agg = total / wall
    per = sum(r["decode"] for r in ok) / len(ok)
    ttfts = [r["ttft"] for r in ok if r["ttft"]]
    print(
        f"c{concurrency}: aggregate {agg:7.1f} tok/s | per-request decode "
        f"{per:7.1f} tok/s | ttft {min(ttfts)*1000:6.0f}-{max(ttfts)*1000:6.0f} ms | "
        f"tokens {total}"
    )


def prefill(prompt_tokens: int = 4096, salt: str = "") -> None:
    filler = " ".join(
        "".join(random.choices(string.ascii_lowercase, k=6)) for _ in range(prompt_tokens)
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{filler}\n\n[run {salt}] What is 2+2?"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    first = None
    n_prompt = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ch = (obj.get("choices") or [{}])[0]
            if (ch.get("delta") or {}).get("content") and first is None:
                first = time.time()
            u = obj.get("usage")
            if u and u.get("prompt_tokens"):
                n_prompt = u["prompt_tokens"]
    if first:
        dt = first - t0
        print(
            f"prefill: {n_prompt or prompt_tokens} tokens in {dt:.2f} s = "
            f"{(n_prompt or prompt_tokens)/dt:6.0f} tok/s"
        )


if __name__ == "__main__":
    salt = "".join(random.choices(string.ascii_lowercase, k=8))
    print(f"salt {salt}")
    level(1, salt)
    level(2, salt)
    level(4, salt)
    prefill(4096, salt)
