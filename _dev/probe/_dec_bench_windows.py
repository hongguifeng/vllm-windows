#!/usr/bin/env python3
"""Run the reference engine's decode benchmark against our own engine.

Derived read only from the reference project's ops/bench/dec_bench.py. The prompt
string, sampling settings, streaming mode, metric keys and printed columns are
kept identical so the numbers can be lined up against the numbers that project
recorded for itself. Only the endpoint and model name change, and the warm up is
longer because both projects concluded that a cold engine under host memory
pressure can read twenty five percent slow.

Standard library only. Point --port at our engine.

Usage:
    python _dec_bench_windows.py --port 8111
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request

PROMPT = "请详细说明 vLLM 中 PagedAttention 的工作原理。"
SHORT_PROMPT = "你好，介绍一下你自己。"
KEYS = [
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
    "time_per_output_token_seconds_sum",
    "time_per_output_token_seconds_count",
]


def metric_values(base: str) -> dict:
    """Read the four engine counters this benchmark needs."""
    text = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    values = {}
    for key in KEYS:
        match = re.search(
            "^vllm:%s\\{[^}]*\\}\\s+([\\d.eE+-]+)" % key, text, re.M)
        values[key] = float(match.group(1)) if match else -1.0
    return values


def run(base: str, model: str, n: int, tag: str, prompt: str = PROMPT) -> None:
    """Stream one completion and report step intervals and acceptance."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": n,
        "temperature": 0.6,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        base + "/v1/completions", body, {"Content-Type": "application/json"})
    before = metric_values(base)
    start = time.time()
    arrivals = []
    completion = None
    for line in urllib.request.urlopen(request, timeout=600):
        text = line.decode(errors="ignore")
        if text.startswith("data: ") and "[DONE]" not in text:
            try:
                payload = json.loads(text[6:])
            except ValueError:
                continue
            if payload.get("usage"):
                completion = payload["usage"]["completion_tokens"]
            elif payload.get("choices"):
                arrivals.append(time.time() - start)
    after = metric_values(base)
    gaps = [(arrivals[i] - arrivals[i - 1]) * 1000
            for i in range(1, len(arrivals))]
    drafted = after[KEYS[0]] - before[KEYS[0]]
    accepted = after[KEYS[1]] - before[KEYS[1]]
    count = after[KEYS[3]] - before[KEYS[3]]
    total = after[KEYS[2]] - before[KEYS[2]]
    tpot = (total / count) if count > 0 else float("nan")
    decode_span = arrivals[-1] - arrivals[0]
    rate = ((completion - 1) / decode_span) if decode_span else float("nan")
    tokens_per_step = (completion / len(arrivals)) if arrivals else float("nan")
    print(f"  [{tag}] steps={len(arrivals)} completion={completion} "
          f"accept={(accepted / drafted * 100 if drafted else 0):.1f}% "
          f"tok/step={tokens_per_step:.2f}")
    engine_tpot = (f"  engine TPOT={tpot * 1000:.2f} ms (n={count:.0f})"
                   if tpot == tpot and count > 0 else "")
    print(f"      TTFT={arrivals[0] * 1000:.0f} ms  decode={decode_span:.2f} s "
          f"⇒ {rate:.1f} tok/s{engine_tpot}")
    print(f"      step interval p50={statistics.median(gaps):.1f} "
          f"p95={sorted(gaps)[int(len(gaps) * 0.95) - 1]:.1f} "
          f"max={max(gaps):.1f} ms")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--model", default="qwen3.8-flash-next-full")
    ap.add_argument("--warm-tokens", type=int, default=1024)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    # Both projects concluded a decode comparison needs at least 2500 decoded
    # tokens of warm up, so warm longer than the reference script did.
    for index in range(3):
        run(base, args.model, args.warm_tokens, f"warm{index + 1}")
    run(base, args.model, 64, "warm4")

    for index in range(2, 5):
        run(base, args.model, 256, f"run{index}")
    run(base, args.model, 256, "run5-short-prompt", prompt=SHORT_PROMPT)


if __name__ == "__main__":
    main()
