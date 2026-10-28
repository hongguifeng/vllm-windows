"""Long-context decode A/B between two ports.

The recipe-protocol numbers use ~60-token prompts and 400-token replies, so they
say nothing about agent workloads at 32k-200k context. This prefills a prompt of
a target token count from the repository's own text, then streams a fixed reply
and reports TTFT and decode rate.

    python _dev/probe/_longctx_ab.py --ports 9393,9394 --lengths 32768,100000
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen3.8-Flash-Next"
REPO = Path(__file__).resolve().parents[2]


def post(port: int, path: str, body: dict, timeout: float = 1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(port: int, text: str) -> int:
    with post(port, "/tokenize", {"model": MODEL, "prompt": text}, 300) as r:
        return json.load(r)["count"]


def source_text(min_chars: int) -> str:
    chunks, total = [], 0
    for sub in ("docs", "agent_bench", "vllm/models/qwen4_exp"):
        for f in sorted((REPO / sub).rglob("*")):
            if f.suffix not in (".md", ".py", ".jinja"):
                continue
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(t) < 2000:
                continue
            chunks.append(f"\n# {f.name}\n{t}")
            total += len(t)
            if total >= min_chars:
                return "\n".join(chunks)
    return "\n".join(chunks)


def stream_decode(port: int, prompt: str, max_tokens: int) -> dict:
    prompt = f"[{time.time_ns()}]\n{prompt}"  # fresh salt: no prefix-cache hit
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with post(port, "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            delta = ((d.get("choices") or [{}])[0].get("delta")) or {}
            if delta.get("content") or delta.get("reasoning"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
    n = usage["completion_tokens"]
    return {
        "ttft_s": first - t0,
        "decode_tok_s": (n - 1) / (last - first) if last > first else None,
        "completion_tokens": n,
        "prompt_tokens": usage["prompt_tokens"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="9393,9394")
    ap.add_argument("--lengths", default="32768,100000")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]

    text = source_text(max(lengths) * 5)
    print(f"source text: {len(text)} chars")

    rows = []
    for target in lengths:
        # One rescale from the measured token count, then reuse for every port.
        approx = text[: int(len(text) * target / max(1, tokenize(ports[0], text)))]
        exact = tokenize(ports[0], approx)
        prompt = f"[{time.time_ns()}]\n{approx}\n\nReply with OK."
        n_tok = tokenize(ports[0], prompt)
        print(f"\n== target {target}: prompt {n_tok} tokens ==")
        for port in ports:
            rates, ttfts = [], []
            for _ in range(args.reps):
                r = stream_decode(port, prompt, args.tokens)
                if r["decode_tok_s"] is not None:
                    rates.append(r["decode_tok_s"])
                ttfts.append(r["ttft_s"])
                print(
                    f"  :{port}  prompt {r['prompt_tokens']:6d}  TTFT {r['ttft_s']:7.2f}s "
                    f"({r['prompt_tokens'] / r['ttft_s']:6.0f} tok/s prefill)  "
                    f"decode {(r['decode_tok_s'] or 0.0):7.1f} tok/s",
                    flush=True,
                )
            rows.append((target, port, statistics.median(ttfts), statistics.median(rates)))

    print("\n| context (tokens) | port | TTFT s | decode tok/s |")
    print("|---|---:|---:|---:|")
    for target, port, ttft, rate in rows:
        print(f"| {target:,} | :{port} | {ttft:.2f} | {rate:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


