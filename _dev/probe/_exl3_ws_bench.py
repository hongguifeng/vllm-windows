"""Throughput probe for the Windows EXL3 server (OpenAI-compatible API).

    python _dev/probe/_exl3_ws_bench.py --concurrency 4 --tokens 128

Sends `concurrency` identical greedy requests, waits for all of them, and
reports aggregate completion tok/s plus per-request latency.
"""

import argparse
import json
import threading
import time
import urllib.request

PROMPT = (
    "Write a detailed technical explanation of how pipeline parallelism works "
    "in large language model inference. Cover micro-batching, bubble time, and "
    "why the schedule matters. Keep writing until you are cut off."
)


def request(base: str, model: str, tokens: int, results: list, index: int) -> None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as response:
        payload = json.loads(response.read())
    elapsed = time.perf_counter() - started
    results[index] = (elapsed, payload["usage"]["completion_tokens"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:9400")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-EXL3")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()

    results: list = [None] * args.concurrency
    threads = [
        threading.Thread(
            target=request,
            args=(args.base, args.model, args.tokens, results, index),
        )
        for index in range(args.concurrency)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    total = sum(tokens for _, tokens in results)
    print(f"concurrency={args.concurrency} tokens={args.tokens} wall={wall:.2f}s")
    for elapsed, tokens in results:
        print(f"  request {elapsed:.2f}s  {tokens / elapsed:.1f} tok/s")
    print(f"aggregate {total / wall:.1f} tok/s over {total} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
