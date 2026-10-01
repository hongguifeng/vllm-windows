# Minimal ITL probe: one streaming chat completion, median inter-chunk gap.
# Usage: python _itl_client.py --port 8000 --tokens 384 [--warm 64]
import argparse, json, time, urllib.request


def run(port, tokens, label):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content":
                      "Write a detailed story about a robot learning to paint."}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    gaps, last, n = [], None, 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                n += 1
                now = time.perf_counter()
                if last is not None:
                    gaps.append((now - last) * 1000)
                last = now
    total = time.perf_counter() - t0
    gaps.sort()
    p50 = gaps[len(gaps) // 2] if gaps else float("nan")
    p90 = gaps[int(len(gaps) * 0.9)] if gaps else float("nan")
    print(f"[{label}] tokens={n} total={total:.1f}s "
          f"p50_gap={p50:.2f}ms p90_gap={p90:.2f}ms "
          f"tok/s={n / total:.1f}", flush=True)
    return p50


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--warm", type=int, default=64)
    ap.add_argument("--label", default="itl")
    a = ap.parse_args()
    if a.warm:
        run(a.port, a.warm, a.label + "/warmup")
    run(a.port, a.tokens, a.label)
