"""Long-context decoding with concurrency, normalized by acceptance and step time.

The daily scenario is several long-context requests at once, not short prompts.
Aggregate tok/s alone is confounded by draft acceptance (the engine is not
deterministic), so this reports the decode window, steps (from /metrics), tok/step
and step time for each port.

    python _dev/probe/_longctx_conc.py --ports 9393,9394 --length 32768 --users 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
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
            chunks.append(t)
            total += len(t)
            if total >= min_chars:
                return "\n".join(chunks)
    return "\n".join(chunks)


def metrics(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        txt = r.read().decode("utf-8", "replace")
    out = {}
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = float(line.split()[-1])
    return out


def one(port: int, prompt: str, tokens: int, out: dict, idx: int) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"[{time.time_ns()}-{idx}] {prompt}"}],
        "max_tokens": tokens,
        "min_tokens": tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    first = last = None
    with post(port, "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            delta = ((d.get("choices") or [{}])[0].get("delta")) or {}
            if delta.get("content") or delta.get("reasoning"):
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
    out[idx] = {"ttft": (first or t0) - t0, "end": (last or t0) - t0}


def run(port: int, prompt: str, users: int, tokens: int) -> dict:
    m0 = metrics(port)
    results: dict[int, dict] = {}
    threads = [
        threading.Thread(target=one, args=(port, prompt, tokens, results, i))
        for i in range(users)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    m1 = metrics(port)
    steps = max(1.0, m1.get("drafts", 0) - m0.get("drafts", 0))
    acc = m1.get("accepted", 0) - m0.get("accepted", 0)
    window = max(r["end"] for r in results.values()) - max(
        r["ttft"] for r in results.values()
    )
    if window <= 0:
        window = wall
    return {
        "wall": wall,
        "ttft_max": max(r["ttft"] for r in results.values()),
        "decode_window": window,
        "tok_s": users * tokens / window,
        "steps": steps,
        "tok_per_step": users * tokens / steps,
        "step_ms": window / steps * 1e3,
        "accept_per_step": acc / steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="9393,9394")
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]

    text = source_text(args.length * 5)
    approx = text[: int(len(text) * args.length / max(1, tokenize(ports[0], text)))]
    rows = []
    print(f"prompt target {args.length} tokens, {args.users} users, {args.tokens} out")
    for port in ports:
        for rep in range(args.reps):
            r = run(port, approx, args.users, args.tokens)
            print(
                f"  :{port} rep{rep}  prompt? TTFT {r['ttft_max']:6.1f}s  "
                f"decode {r['tok_s']:6.1f} tok/s  tok/step {r['tok_per_step']:4.2f}  "
                f"acc/step {r['accept_per_step']:4.2f}  step {r['step_ms']:6.1f} ms",
                flush=True,
            )
            rows.append((port, r["tok_s"], r["tok_per_step"], r["step_ms"]))

    print("\n| port | decode tok/s | tok/step | step ms |")
    print("|---|---:|---:|---:|")
    for port in ports:
        sel = [x for x in rows if x[0] == port]
        print(
            f"| :{port} | {statistics.median(x[1] for x in sel):.1f} | "
            f"{statistics.median(x[2] for x in sel):.2f} | "
            f"{statistics.median(x[3] for x in sel):.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
