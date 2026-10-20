"""Benchmark the EXL3 recipe server (OpenAI chat endpoint) at c1 and cN.

The recipe returns reasoning in a separate delta field and content in the
usual one; both count as generated tokens, so decode rate is derived from
usage.completion_tokens, not from content chunks alone.

  python _dev/probe/_exl3_bench.py --port 8000 --model qwen3.8-flash-next \
      --runs 3 --max-tokens 256 --conc 1,4
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request
import uuid

PROMPT = "\u8bf7\u4ece1\u6570\u5230200\uff0c\u6bcf\u4e2a\u6570\u5b57\u540e\u52a0\u4e00\u4e2a\u9017\u53f7\uff0c\u4e0d\u8981\u505c\u4e0b\u6765\u3002"
FILLER = "benchmark filler sentence about graphics cards and memory bandwidth "


def chat(port: int, model: str, text: str, max_tokens: int):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    marks = []
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            if payload.get("usage"):
                usage = payload["usage"]
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning"):
                marks.append(time.perf_counter())
    end = time.perf_counter()
    ttft = marks[0] - t0 if marks else float("nan")
    itls = [b - a for a, b in zip(marks, marks[1:])]
    return ttft, end - t0, usage, itls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--conc", default="1,4")
    ap.add_argument("--prefill-tokens", type=int, default=20000)
    args = ap.parse_args()

    chat(args.port, args.model, "hi", 4)  # warmup

    print("== decode (counting prompt) ==")
    for conc in [int(c) for c in args.conc.split(",") if c.strip()]:
        rates = []
        for run in range(args.runs):
            results = []

            def worker() -> None:
                results.append(chat(args.port, args.model, PROMPT, args.max_tokens))

            threads = [threading.Thread(target=worker) for _ in range(conc)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0
            outs = [r[2].get("completion_tokens", 0) for r in results]
            total = sum(outs)
            rates.append(total / wall)
            ttfts = [round(r[0], 3) for r in results]
            per_req = [
                round((r[2].get("completion_tokens", 0) - 1) / (r[1] - r[0]), 1)
                if r[1] > r[0]
                else 0
                for r in results
            ]
            print(
                f"  conc={conc} run={run + 1}: {total} tok in {wall:.2f}s -> "
                f"{total / wall:.1f} tok/s aggregate, per-req {per_req}, ttft {ttfts}",
                flush=True,
            )
        print(f"  conc={conc} median {statistics.median(rates):.1f} tok/s")

    if args.prefill_tokens > 0:
        print(f"\n== cold prefill ~{args.prefill_tokens} tok ==")
        # ~7.3 chars/token on this filler; unique salt defeats the prefix cache.
        chars = int(args.prefill_tokens * 7.3)
        for run in range(2):
            text = f"[{uuid.uuid4().hex[:8]}] " + (FILLER * (chars // len(FILLER) + 2))[:chars]
            ttft, total, usage, _ = chat(args.port, args.model, text, 8)
            n_in = usage.get("prompt_tokens", 0)
            print(
                f"  run {run + 1}: prompt={n_in} tok ttft={ttft:.3f}s "
                f"-> {n_in / ttft:,.0f} tok/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
