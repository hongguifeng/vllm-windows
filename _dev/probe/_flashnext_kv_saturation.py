"""Deliberately fill the KV cache and watch how the scheduler reacts.

K requests fire at the same time, each sized with the server's own /tokenize so
none gets rejected, and each with a distinct head so prefix caching cannot
collapse them onto one set of blocks. While they run this samples
vllm:kv_cache_usage_perc and free VRAM, then diffs the engine log for preemption
and alloc-heal lines. K*T is what decides whether the cache is merely full or
actually over capacity.

    python _flashnext_kv_saturation.py --port 9394 --requests 4 --tokens 131072
"""

import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexedly quick daft zebras jump. "
)
NEEDLE = "IMPORTANT NOTE: the secret passcode is 7391-QKXM. Keep it in mind."
QUESTION = ("\n\nAnswer with one line: the secret passcode from the text above.")


def tokenize(port, s):
    b = json.dumps({"model": "qwen3.8-flash-next-win", "prompt": s}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=b,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=180).read())["count"]


def head(i):
    """Different tokens at position 0 for every request. vLLM v1 block hashes
    chain to the previous block, so one differing head block means no reuse."""
    return " ".join(f"salt{i:03d}w{j:03d}" for j in range(96)) + "\n"


def size_text(port, budget, pre, with_question=True):
    pre_tok = tokenize(port, pre)
    tail = QUESTION if with_question else ""
    half = max((budget - pre_tok - 128 - len(tail)) // (2 * tokenize(port, FILLER)), 1)
    for _ in range(5):
        text = pre + FILLER * half + "\n" + NEEDLE + "\n" + FILLER * half + tail
        got = tokenize(port, text)
        if abs(got - budget) <= 120 or got > budget:
            break
        half = max(int(half * (budget - pre_tok) / got), 1)
    return text, got


def metrics(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as f:
            body = f.read().decode()
    except Exception:
        return None, None
    m = re.search(r"^vllm:kv_cache_usage_perc(\{[^}]*\})?\s+([0-9.]+)", body, re.M)
    p = re.search(r"^vllm:num_preemptions_total(\{[^}]*\})?\s+([0-9.]+)", body, re.M)
    return (float(m.group(2)) if m else None, float(p.group(2)) if p else None)


def free_mib(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def fire(port, content, max_tokens=16, salt=None, min_tokens=None):
    body = {"model": "qwen3.8-flash-next-win",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0,
            "cache_salt": salt or "sat"}
    if min_tokens:
        # Without this the model answers in ~8 tokens, releases its blocks and the
        # cache never fills no matter how many long prompts are in flight.
        body["min_tokens"] = min_tokens
    b = json.dumps(body).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=b,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=900) as f:
            d = json.loads(f.read())
        u = d.get("usage", {})
        ans = (d["choices"][0]["message"]["content"] or "").strip()
        return (round(time.time() - t0, 1), u.get("prompt_tokens"),
                "7391-QKXM" in ans, ans[:40], u.get("completion_tokens"),
                d["choices"][0].get("finish_reason"))
    except urllib.error.HTTPError as e:
        return ("HTTP %s" % e.code, e.read().decode()[:120].strip(), False, "")
    except Exception as e:
        return ("ERR", repr(e)[:120], False, "")


def count_lines(path, pat):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return len(re.findall(pat, f.read()))
    except Exception:
        return -1


def run(port, k, token_list, gpu, log, seconds, outs, canary_tokens, canary_delay, images):
    texts = []
    payloads = []
    for i in range(k):
        t0 = time.time()
        # The image placeholders cost real tokens in the prompt, so the text budget
        # has to give them up or every request comes back 400.
        budget = token_list[i] - (images * 1240 if images else 0)
        text, got = size_text(port, budget, head(i), with_question=not images)
        texts.append(text)
        if images:
            from _flashnext_multi_vision import make_image, STYLES
            parts = [{"type": "text", "text": text}]
            for j in range(images):
                c, s, n = STYLES[j % len(STYLES)]
                parts.append({"type": "image_url",
                              "image_url": {"url": "data:image/png;base64,"
                                            + make_image(2048, c, s)}})
            parts.append({"type": "text",
                          "text": "\n\nAnswer with one line: the passcode, then the "
                                  "colour and shape of each image in order."})
            payloads.append(parts)
        else:
            payloads.append(text)
        print(f"[sat] sized req {i}: text {got} tok"
              + (f" + {images} images" if images else "")
              + f" in {time.time()-t0:.1f}s")
    if len(outs) < k:
        outs = outs + [outs[-1]] * (k - len(outs))

    usage0, preempt0 = metrics(port)
    heal0 = count_lines(log, r"alloc-heal fired")
    print(f"[sat] before: kv_usage={usage0} preemptions={preempt0} "
          f"heal={heal0} free={free_mib(gpu)} MiB")

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            u, p = metrics(port)
            samples.append((round(time.time(), 1), u, p, free_mib(gpu)))
            time.sleep(0.5)

    th = threading.Thread(target=sampler)
    th.start()

    results = [None] * k
    def worker(i):
        results[i] = fire(port, payloads[i], outs[i], min_tokens=outs[i])
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    t_start = time.time()
    for t in threads:
        t.start()

    # A short request fired while the cache is supposed to be full: if the card
    # is going to turn pathological this is where it shows.
    canary = [None]
    def fire_canary():
        time.sleep(canary_delay)
        budget = canary_tokens - (1240 if images else 0)
        text, got = size_text(port, budget, head(999), with_question=not images)
        if images:
            from _flashnext_multi_vision import make_image, STYLES
            c, s, n = STYLES[0]
            content = [{"type": "text", "text": text},
                       {"type": "image_url",
                        "image_url": {"url": "data:image/png;base64,"
                                      + make_image(2048, c, s)}},
                       {"type": "text",
                        "text": "\n\nAnswer with one line: the passcode, then the "
                                "colour and shape in the image."}]
        else:
            content = text
        canary[0] = (got, fire(port, content, 16, salt="canary"))
    canth = threading.Thread(target=fire_canary)
    canth.start()

    for t in threads:
        t.join()
    canth.join()
    time.sleep(1.0)
    stop.set()
    th.join()

    print(f"[sat] demands {token_list} = {sum(token_list)} tok against the cache; "
          f"wall {time.time()-t_start:.1f}s")
    for i, r in enumerate(results):
        print(f"  req {i}: {r[0]}s  prompt={r[1]}  needle={r[2]}  out={r[4]} {r[5]}  {r[3]}")
    if canary[0]:
        print(f"[sat] canary {canary[0][0]} tok inside the run -> {canary[0][1][0]}s "
              f"needle={canary[0][1][2]}")
    usage = [s[1] for s in samples if s[1] is not None]
    frees = [s[3] for s in samples if s[3] > 0]
    preempts = [s[2] for s in samples if s[2] is not None]
    if usage:
        print(f"[sat] kv_usage max={max(usage):.3f} "
              f"(>=0.98 in {sum(1 for u in usage if u >= 0.98)} of {len(usage)} samples)")
    if preempts:
        print(f"[sat] preemptions_total: {preempts[0]} -> {preempts[-1]} "
              f"(delta {preempts[-1]-preempts[0]:.0f})")
    if frees:
        print(f"[sat] free MiB min={min(frees)} max={max(frees)} last={frees[-1]}")
    heal1 = count_lines(log, r"alloc-heal fired")
    print(f"[sat] heal fired {heal0} -> {heal1} (delta {heal1-heal0})")
    usage1, preempt1 = metrics(port)
    print(f"[sat] after : kv_usage={usage1} preemptions={preempt1} free={free_mib(gpu)} MiB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9394)
    ap.add_argument("--requests", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--log", required=True, help="engine stdout log for heal/preempt lines")
    ap.add_argument("--seconds", type=float, default=0.5, help="sample interval")
    ap.add_argument("--out-tokens", default="16",
                    help="comma list of max_tokens per request (repeats the last value)")
    ap.add_argument("--canary-tokens", type=int, default=0,
                    help="fire a short request this many tokens long mid-run")
    ap.add_argument("--canary-delay", type=float, default=40.0)
    ap.add_argument("--images", type=int, default=0,
                    help="capped 2048px images per request (max 4)")
    ap.add_argument("--tokens-list", default='',
                    help="comma-separated per-request budgets; overrides --requests/--tokens")
    a = ap.parse_args()
    tl = [int(x) for x in a.tokens_list.split(",")] if a.tokens_list else [a.tokens] * a.requests
    raise SystemExit(run(a.port, len(tl), tl, a.gpu, a.log, a.seconds,
                         [int(x) for x in a.out_tokens.split(",")],
                         a.canary_tokens, a.canary_delay, a.images))
