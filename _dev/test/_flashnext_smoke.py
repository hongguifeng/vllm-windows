"""Chat smoke client for the Flash-Next structural probe.

The point is not quality -- the PLE layer is deliberately absent from the view
this runs against. The point is whether a full forward pass through qwen4_exp
finishes on Windows at all: GDN linear attention (36 layers), QSA sparse
attention (12 layers), hc_count=4 hyper-connections, the INC 2/3-bit
RoutedExperts MoE, and the sampler. Four signals answer that:

  1. /health turns green (engine came up, weights mapped, KV sized)
  2. a short prompt returns finish_reason=stop with non-empty content
  3. a long-ish prompt exercises chunked prefill across the 2048-token chunks
  4. two concurrent requests exercise the batching path (the 27B's known
     weak spot on this rig)

Run against a server started by _dev\\bin\\_flashnext_struct_serve.ps1:
  D:\\code\\vllm-windows\\.venv\\Scripts\\python.exe D:\\code\\vllm-windows\\_dev\\test\\_flashnext_smoke.py [--port 8111]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:{port}"
MODEL = "qwen3.8-flash-next-struct"


def post(port: int, path: str, payload: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        BASE.format(port=port) + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    out["_wall"] = time.perf_counter() - t0
    return out


def get_health(port: int, deadline: float) -> bool:
    url = BASE.format(port=port) + "/health"
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except urllib.error.HTTPError:
            pass
        except OSError:
            pass
        time.sleep(3)
    return False


def one(port: int, tag: str, prompt: str, max_tokens: int = 32) -> dict:
    r = post(port, "/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    ch = r["choices"][0]
    usage = r.get("usage", {})
    print(f"  {tag:14s} finish={ch['finish_reason']:10s} "
          f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
          f"wall={r['_wall']:6.2f}s  content={len(ch['message'].get('content') or '')} chars")
    body = (ch["message"].get("content") or "").strip()
    print(f"     -> {body[:120]!r}")
    return r


def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--model", default=MODEL, help="served model name")
    ap.add_argument("--long-sentences", type=int, default=1200,
                    help="sentences in the long-prefill prompt; keep it under max-model-len")
    ap.add_argument("--wait", type=float, default=900, help="seconds to wait for /health")
    args = ap.parse_args()
    MODEL = args.model

    print("==== flash-next structural smoke ====")
    ok = get_health(args.port, time.perf_counter() + args.wait)
    print(f"  /health        : {'READY' if ok else 'NEVER CAME UP'}")
    if not ok:
        return 1
    rc = 0
    try:
        one(args.port, "short", "What is the capital of France? Answer in one word.")
        long_prompt = " ".join(
            f"sentence {i} about the weather in city {i}."
            for i in range(args.long_sentences)
        )
        r = one(args.port, "long-prefill", long_prompt, max_tokens=16)
        pt = r.get("usage", {}).get("prompt_tokens", 0)
        print(f"     prompt_tokens={pt} (>2048 means chunked prefill ran)")
    except Exception as e:
        print(f"  single request FAILED: {type(e).__name__}: {e}")
        rc = 1

    try:
        t0 = time.perf_counter()
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(2) as ex:
            futs = [
                ex.submit(one, args.port, f"concurrent-{i}",
                          f"Say the word {'alpha' if i == 0 else 'beta'} only.")
                for i in range(2)
            ]
            [f.result() for f in futs]
        print(f"  two concurrent   : both returned in {time.perf_counter()-t0:.2f}s")
    except Exception as e:
        print(f"  concurrent FAILED: {type(e).__name__}: {e}")
        rc = 1
    print(f"\nverdict: {'ARCHITECTURE RUNS' if rc == 0 else 'DID NOT COMPLETE'}"
          " (PLE is intentionally absent here; this says nothing about PLE)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
