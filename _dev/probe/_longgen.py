"""Long-generation throughput probe: TTFT + steady-state tok/s over a fixed prompt.

Usage: python _longgen.py --port 9401 --max-tokens 256 [--conc 1]
"""

import argparse
import json
import time
import urllib.request

PROMPT = "\u8bf7\u4ece1\u6570\u5230200\uff0c\u6bcf\u4e2a\u6570\u5b57\u540e\u52a0\u4e00\u4e2a\u9017\u53f7\uff0c\u4e0d\u8981\u505c\u4e0b\u6765\u3002"


def one_request(port: int, model: str, max_tokens: int) -> tuple[float, float, int, dict]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
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
    ttft = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
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
            delta = choices[0]["delta"].get("content") or ""
            if not delta:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += len(delta)
    return ttft or 0.0, time.perf_counter() - t0, chunks, usage or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9401)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    one_request(args.port, args.model, 8)  # warmup
    ttft, total, chars, usage = one_request(args.port, args.model, args.max_tokens)
    out_tokens = usage.get("completion_tokens") or 0
    decode_s = total - ttft
    print(
        f"[{args.label}] ttft={ttft:.3f}s total={total:.3f}s "
        f"out_tokens={out_tokens} chars={chars} "
        f"decode={out_tokens / decode_s:.1f} tok/s (e2e {out_tokens / total:.1f} tok/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
