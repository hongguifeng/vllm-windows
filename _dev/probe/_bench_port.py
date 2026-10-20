"""Tiny OpenAI-endpoint bench: prefill TTFT and concurrent decode.

Written for the PP-vs-TP comparison so every arm is measured by the same code.
No dependencies beyond the standard library.

  python _dev/probe/_bench_port.py --port 9402 --model qwen38-27b
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--prefill-len", type=int, default=4096)
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--decode-tokens", type=int, default=128)
ap.add_argument("--label", default="")
args = ap.parse_args()

URL = f"http://127.0.0.1:{args.port}/v1/completions"


def post(prompt, max_tokens):
    """Streaming request: returns (ttft, total, completion_tokens, inter-token latencies)."""
    body = json.dumps(
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    marks = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                marks.append(time.perf_counter())
    end = time.perf_counter()
    ttft = (marks[0] - t0) if marks else float("nan")
    itls = [b - a for a, b in zip(marks, marks[1:])]
    return ttft, end - t0, len(marks), itls


# A longer filler so --prefill-len is actually reached: the tokenizer turns this
# into roughly one token per 3.5 characters.
prompt = ("benchmark filler sentence about graphics cards and memory bandwidth " * 1200)[
    : args.prefill_len * 4
]

print(f"[{args.label}] warmup", flush=True)
post("warmup", 4)

print(f"[{args.label}] prefill {args.prefill_len} tok, single request", flush=True)
ttft, total, n_out, _ = post(prompt, args.max_tokens)
print(
    f"[{args.label}] prefill: ttft={ttft:.3f}s total={total:.3f}s "
    f"prompt_tok/s={args.prefill_len / ttft:,.0f} (assumed {args.prefill_len} in, {n_out} out)",
    flush=True,
)

print(f"[{args.label}] decode {args.conc} concurrent, {args.decode_tokens} tok each", flush=True)
results = []


def worker(i):
    results.append(post(prompt[:4000], args.decode_tokens))


threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.conc)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - t0
gen = sum(r[2] for r in results)
all_itl = [x for r in results for x in r[3]]
tpot = [statistics.median(r[3]) * 1000 for r in results if r[3]]
print(
    f"[{args.label}] decode: {gen} tok in {wall:.2f}s -> {gen / wall:,.1f} tok/s aggregate, "
    f"median ITL {statistics.median(all_itl) * 1000:.2f} ms, per-req median ITL "
    f"{[round(x, 2) for x in tpot]} ms",
    flush=True,
)
