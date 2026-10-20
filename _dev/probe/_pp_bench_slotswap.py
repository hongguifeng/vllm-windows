"""PP benchmark with exact token accounting for the slot-swap A/B.

The older _bench_port.py assumes its 16k-character filler is 4000 tokens; this
one reads usage.prompt_tokens so TTFT per token is exact.  Every prefill gets a
unique salt so the prefix cache cannot make a later run look faster.

  python _dev/probe/_pp_bench_slotswap.py --port 9401 --prefill-tokens 4000
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request
import uuid

FILLER = "benchmark filler sentence about graphics cards and memory bandwidth "


def post(port: int, model: str, prompt: str, max_tokens: int):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    marks = []
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                marks.append(time.perf_counter())
    end = time.perf_counter()
    ttft = marks[0] - t0 if marks else float("nan")
    itls = [b - a for a, b in zip(marks, marks[1:])]
    return ttft, end - t0, usage, itls


def make_prompt(tokens: int, salt: str) -> str:
    # ~5.6 chars/token on this filler; overshoot and trim to a token target via
    # a calibration request in main().
    body = FILLER * (tokens * 6 // len(FILLER) + 2)
    return f"[{salt}] " + body[: tokens * 6]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9401)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--prefill-tokens", type=int, default=4000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conc", type=int, default=4)
    args = ap.parse_args()

    # Calibrate chars-per-token on this tokenizer, then build exact-ish prompts.
    salt = uuid.uuid4().hex[:8]
    probe = make_prompt(256, salt)
    _, _, usage, _ = post(args.port, args.model, probe, 1)
    cpt = len(probe) / usage["prompt_tokens"]
    print(f"calibration: {len(probe)} chars = {usage['prompt_tokens']} tok -> {cpt:.2f} chars/tok")

    target_chars = int(args.prefill_tokens * cpt)

    print(f"\nprefill {args.prefill_tokens} tok, streamed, unique salt per run:")
    ttfts = []
    for i in range(args.runs):
        prompt = f"[{uuid.uuid4().hex[:8]}] " + (FILLER * (target_chars // len(FILLER) + 2))[:target_chars]
        ttft, total, usage, _ = post(args.port, args.model, prompt, 8)
        ttfts.append(ttft)
        print(
            f"  run {i + 1}: prompt={usage.get('prompt_tokens')} tok ttft={ttft:.3f}s "
            f"total={total:.3f}s -> {usage.get('prompt_tokens', 0) / ttft:,.0f} tok/s",
            flush=True,
        )
    print(f"  ttft median {statistics.median(ttfts):.3f}s")

    print(f"\ndecode {args.runs} runs, {args.prefill_tokens} tok prompt, 256 new tokens:")
    rates = []
    for i in range(args.runs):
        prompt = f"[{uuid.uuid4().hex[:8]}] " + (FILLER * (target_chars // len(FILLER) + 2))[:target_chars]
        ttft, total, usage, _ = post(args.port, args.model, prompt, 256)
        n_out = usage.get("completion_tokens", 0)
        rate = (n_out - 1) / (total - ttft) if total > ttft else float("nan")
        rates.append(rate)
        print(
            f"  run {i + 1}: ttft={ttft:.3f}s out={n_out} total={total:.3f}s "
            f"decode={rate:.1f} tok/s (e2e {n_out / total:.1f})",
            flush=True,
        )
    print(f"  decode median {statistics.median(rates):.1f} tok/s")

    if args.conc > 1:
        print(f"\ndecode {args.conc} concurrent, 256 tokens each:")
        results = []

        def worker() -> None:
            salt = uuid.uuid4().hex[:8]
            prompt = f"[{salt}] " + (FILLER * (target_chars // len(FILLER) + 2))[:target_chars]
            results.append(post(args.port, args.model, prompt, 256))

        threads = [threading.Thread(target=worker) for _ in range(args.conc)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        gen = sum(r[2].get("completion_tokens", 0) for r in results)
        all_itl = [x for r in results for x in r[3]]
        per_req = [statistics.median(r[3]) * 1000 for r in results if r[3]]
        print(
            f"  {gen} tok in {wall:.2f}s -> {gen / wall:,.1f} tok/s aggregate, "
            f"median ITL {statistics.median(all_itl) * 1000:.2f} ms, per-req {[round(x, 1) for x in per_req]} ms"
        )


if __name__ == "__main__":
    main()
