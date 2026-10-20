# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact single-card decode bench: non-streaming requests, token counts from
`usage`, plus spec-decode acceptance read from /metrics around the run.

  python _dev/probe/_tpot_bench.py --port 9402 --conc 1 --tokens 200
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
ap.add_argument("--prompt-tokens", type=int, default=512)
ap.add_argument("--tokens", type=int, default=200)
ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--reps", type=int, default=1)
# Defeats the prefix cache, so the wall time of a 1-token request is a cold
# prefill instead of a hit.
ap.add_argument("--unique", action="store_true")
# Real text instead of the repeated filler clause: draft acceptance on
# repetitive text flatters MTP, so arms need a prompt that is not self-similar.
ap.add_argument("--text-file", default="")
# Chat-shaped input: a drafter can be sensitive to the prompt distribution, so
# the same measurement is repeated through /v1/chat/completions.
ap.add_argument("--chat", action="store_true")
ap.add_argument("--label", default="")
args = ap.parse_args()

BASE = f"http://127.0.0.1:{args.port}"


def post(prompt: str, max_tokens: int) -> dict:
    if args.chat:
        body_dict = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        path = "/v1/chat/completions"
    else:
        body_dict = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        }
        path = "/v1/completions"
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


def metrics() -> dict:
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=60) as resp:
        text = resp.read().decode()
    wanted = (
        "vllm:spec_decode_num_accepted_tokens_total{",
        "vllm:spec_decode_num_drafts_total{",
        "vllm:spec_decode_num_draft_tokens_total{",
        "vllm:generation_tokens_total{",
        "vllm:iteration_tokens_total_count{",
        "vllm:iteration_tokens_total_sum{",
    )
    out = {}
    for line in text.splitlines():
        for key in wanted:
            if line.startswith(key):
                out[key.split("{")[0]] = float(line.rsplit(" ", 1)[1])
    return out


# A prompt of roughly the requested token count, built from a repeated clause.
clause = "The quick brown fox jumps over the lazy dog while the engineer reads logs. "
filler = (clause * ((args.prompt_tokens * 6) // len(clause) + 2))[
    : args.prompt_tokens * 6
]
if args.text_file:
    with open(args.text_file, encoding="utf-8", errors="ignore") as fh:
        text = fh.read().replace("\r\n", "\n")
    start = 0 if args.prompt_tokens > 2000 else len(text) // 3
    filler = text[start : start + args.prompt_tokens * 6]
if args.unique:
    import uuid

    filler = f"[{uuid.uuid4()}] " + filler

# Warmup so kernels, graphs and the PLE row cache are in their steady state.
# A unique prompt must stay uncached, or the prefill measurement is a hit.
if not args.unique:
    post(filler, 8)

before = metrics()
latencies: list[float] = []
per_req: list[float] = []
emitted: list[int] = []
lock = threading.Lock()


def worker() -> None:
    start = time.perf_counter()
    out = post(filler, args.tokens)
    elapsed = time.perf_counter() - start
    tokens = out["usage"]["completion_tokens"]
    with lock:
        latencies.append(elapsed)
        emitted.append(tokens)
        per_req.append(tokens / elapsed)


for _ in range(args.reps):
    threads = [threading.Thread(target=worker) for _ in range(args.conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
after = metrics()

d_tokens = after.get("vllm:generation_tokens_total", 0) - before.get(
    "vllm:generation_tokens_total", 0
)
d_steps = after.get("vllm:iteration_tokens_total_count", 0) - before.get(
    "vllm:iteration_tokens_total_count", 0
)
d_accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get(
    "vllm:spec_decode_num_accepted_tokens_total", 0
)
d_drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - before.get(
    "vllm:spec_decode_num_drafts_total", 0
)
d_draft_tokens = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get(
    "vllm:spec_decode_num_draft_tokens_total", 0
)

label = args.label or f"c{args.conc}"
if d_tokens > 0 and d_steps > 0:
    print(
        f"[{label}] {d_tokens:.0f} tokens over {d_steps:.0f} steps "
        f"({d_tokens / d_steps:.2f} tok/step), wall {wall:.2f}s"
    )
else:
    print(f"[{label}] wall {wall:.2f}s, metrics deltas unavailable")
if d_drafts > 0:
    print(
        f"[{label}] spec: {d_drafts:.0f} drafting steps, "
        f"{d_draft_tokens / d_drafts:.2f} drafts/step, "
        f"{d_accepted / d_drafts:.2f} accepted/step, "
        f"acceptance {100 * d_accepted / max(d_draft_tokens, 1):.1f}%"
    )
# Aggregates come from this run's own requests only; the /metrics window can
# straddle a neighbouring request, so it is used for acceptance, not throughput.
agg = sum(emitted) / wall
print(
    f"[{label}] aggregate {agg:.1f} tok/s, per-request "
    f"{statistics.median(per_req):.1f} tok/s, "
    f"latency median {statistics.median(latencies):.2f}s, "
    f"tokens {sum(emitted)}"
)
