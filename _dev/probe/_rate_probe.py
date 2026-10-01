"""Measure decode step cost: per-stream rate with GPU utilisation sampled."""
import argparse
import json
import subprocess
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8111/v1/chat/completions"
MODEL = "qwen3.8-flash-next-full"
TOKENS = 256


def ask(i, prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": TOKENS,
        "temperature": 0,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(BASE, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        usage = data.get("usage", {})
        return {
            "i": i,
            "finish": data.get("choices", [{}])[0].get("finish_reason"),
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
            "wall": time.time() - t0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"i": i, "error": repr(exc), "wall": time.time() - t0}


def gpu_window():
    """Sample GPU utilisation and return the samples taken while open."""
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", "250"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        shell=False)
    return proc


def collect(proc):
    proc.terminate()
    out = proc.communicate()[0]
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return rows


def run_level(n):
    prompts = [f"Count down from {900 + 100 * k} to {800 + 100 * k}, "
               f"one number per line." for k in range(n)]
    proc = gpu_window()
    t0 = time.time()
    threads = {}
    workers = [threading.Thread(target=threads.__setitem__, args=(k, ask(k, p)))
               for k, p in enumerate(prompts)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    wall = time.time() - t0
    gpu = collect(proc)
    results = [threads[k] for k in range(n)]
    utils = [g[0] for g in gpu]
    total_completion = sum(r.get("completion", 0) for r in results)
    per = [r["completion"] / r["wall"] for r in results if r.get("completion")]
    print(f"n={n}: wall={wall:6.2f}s completion={total_completion:4d} "
          f"aggregate={total_completion/wall:6.1f} tok/s "
          f"per-stream={', '.join(f'{p:5.1f}' for p in per)}")
    if utils:
        print(f"   gpu util %: mean={sum(utils)/len(utils):5.1f} "
              f"min={min(utils):5.1f} max={max(utils):5.1f} "
              f"(samples={len(utils)})")
    for r in results:
        if r.get("error"):
            print(f"   req {r['i']}: {r['error']}")
    return results


def warm_up(tokens: int) -> None:
    """Generate ``tokens`` decoded tokens before timing.

    Their native-WSL2 notes are explicit that a decode measurement taken before
    the engine has decoded a few thousand tokens is not comparable; the same
    configuration measured 22.2 ms and 17.87 ms per step under those conditions.
    """
    t0 = time.time()
    got = 0
    k = 0
    while got < tokens:
        r = ask(k, f"Count down from {9999 - k} to 1, one number per line.")
        n = r.get("completion", 0)
        if not n:
            print(f"warm-up stopped early: {r.get('error') or r}")
            break
        got += n
        k += 1
    dt = time.time() - t0
    print(f"warm-up: {got} tokens in {dt:6.1f}s ({got / dt:.1f} tok/s)")


def main() -> None:
    global BASE, MODEL, TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--model", default="qwen3.8-flash-next-full",
                    help="served model name")
    ap.add_argument("--tokens", type=int, default=TOKENS)
    ap.add_argument("--warmup", type=int, default=2500,
                    help="decode tokens to generate before timing (0 disables)")
    ap.add_argument("--c1-only", action="store_true",
                    help="measure a single stream only")
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    MODEL = args.model
    TOKENS = args.tokens
    print(f"model={MODEL} tokens/request={TOKENS}")
    if args.warmup > 0:
        warm_up(args.warmup)
    levels = (1,) if args.c1_only else (1, 2, 4)
    for n in levels:
        run_level(n)


main()
