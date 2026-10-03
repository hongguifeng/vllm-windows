"""Grow a conversation until the KV cache is full and see what breaks.

Every turn appends a fresh ~turn_tokens user message plus a forced assistant
reply, so the next turn has to carry all of it. Two conversations run side by
side; their combined history passes the cache ceiling partway through, which is
what a real long chat session does -- the request itself never exceeds
--max-model-len, the *total* held across conversations does.

Each turn carries its own passcode, and the last turn asks for all of them, so a
recomputed or evicted history shows up as a wrong answer rather than a silent
one.

    python _flashnext_multiturn_grow.py --port 9394 --conversations 2 --turns 4 \
        --turn-tokens 60000 --reply-tokens 1000
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
    "How vexingly quick daft zebras jump. "
)
CODES = ["ALPHA-0001", "BRAVO-0002", "CHARLIE-0003", "DELTA-0004",
         "ECHO-0005", "FOXTROT-0006"]


def tokenize(port, s):
    b = json.dumps({"model": "qwen3.8-flash-next-win", "prompt": s}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=b,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=180).read())["count"]


def turn_text(port, budget, conv, turn, code):
    """Sized against the server's own tokenizer, with tokens at position 0 that
    no other conversation shares."""
    pre = " ".join(f"conv{conv:02d}w{j:03d}" for j in range(96)) + "\n"
    pre_tok = tokenize(port, pre)
    tail = (f"\n\nIMPORTANT NOTE: the passcode for this turn is {code}. "
            "Keep it in mind for later.")
    half = max((budget - pre_tok - 128 - len(tail) // 4) // (2 * tokenize(port, FILLER)), 1)
    for _ in range(5):
        text = pre + FILLER * half + "\n" + tail + "\n" + FILLER * half
        got = tokenize(port, text)
        if abs(got - budget) <= 120 or got > budget:
            break
        half = max(int(half * (budget - pre_tok) / got), 1)
    return text, got


def chat(port, messages, max_tokens, min_tokens):
    b = json.dumps({"model": "qwen3.8-flash-next-win", "messages": messages,
                    "max_tokens": max_tokens, "min_tokens": min_tokens,
                    "temperature": 0.0}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               data=b, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=900) as f:
            d = json.loads(f.read())
    except urllib.error.HTTPError as e:
        return ("HTTP %s" % e.code, e.read().decode()[:160].strip(), 0, 0, "")
    u = d.get("usage", {})
    return (round(time.time() - t0, 1), u.get("prompt_tokens"),
            u.get("completion_tokens"), d["choices"][0].get("finish_reason"),
            (d["choices"][0]["message"]["content"] or "").strip())


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


def count(path, pat):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return len(re.findall(pat, f.read()))
    except Exception:
        return -1


def grow(port, conv, turns, turn_tokens, reply_tokens):
    messages = []
    for t in range(turns):
        code = CODES[t % len(CODES)]
        if t == turns - 1:
            text = ("Earlier you were given a passcode in each user message, in order. "
                    "List every one of them, in order, one per line.")
            got = tokenize(port, text)
        else:
            text, got = turn_text(port, turn_tokens, conv, t, code)
        messages.append({"role": "user", "content": text})
        lat, prompt, out, finish, ans = chat(port, messages, reply_tokens, reply_tokens)
        print(f"[conv {conv} turn {t}] prompt={prompt} (hist ~{got} new) "
              f"out={out} {finish} {lat}s  code={code} present={code in ans}")
        if isinstance(lat, str):
            return messages
        # The reply becomes part of the history the next turn has to carry.
        messages.append({"role": "assistant", "content": ans})
    return messages


def run(port, convs, turns, turn_tokens, reply_tokens, gpu, log):
    usage0, preempt0 = metrics(port)
    heal0 = count(log, r"alloc-heal fired")
    print(f"[grow] before: kv_usage={usage0} preemptions={preempt0} "
          f"heal={heal0} free={free_mib(gpu)} MiB")
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            u, p = metrics(port)
            samples.append((u, p, free_mib(gpu)))
            time.sleep(0.5)

    th = threading.Thread(target=sampler)
    th.start()
    t_start = time.time()
    threads = [threading.Thread(target=grow, args=(port, c, turns, turn_tokens,
                                                   reply_tokens)) for c in range(convs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(1.0)
    stop.set()
    th.join()

    demand = convs * turns * (turn_tokens + reply_tokens)
    print(f"[grow] {convs} conversations x {turns} turns x {turn_tokens} tok "
          f"~ {demand} tok; wall {time.time()-t_start:.1f}s")
    usage = [s[0] for s in samples if s[0] is not None]
    preempts = [s[1] for s in samples if s[1] is not None]
    frees = [s[2] for s in samples if s[2] > 0]
    if usage:
        print(f"[grow] kv_usage max={max(usage):.3f} "
              f"(>=0.98 in {sum(1 for u in usage if u >= 0.98)} of {len(usage)} samples)")
    if preempts:
        print(f"[grow] preemptions {preempts[0]} -> {preempts[-1]} "
              f"(delta {preempts[-1]-preempts[0]:.0f})")
    if frees:
        print(f"[grow] free MiB min={min(frees)} max={max(frees)} last={frees[-1]}")
    heal1 = count(log, r"alloc-heal fired")
    print(f"[grow] heal {heal0} -> {heal1} (delta {heal1-heal0})")
    usage1, preempt1 = metrics(port)
    print(f"[grow] after : kv_usage={usage1} preemptions={preempt1} free={free_mib(gpu)} MiB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9394)
    ap.add_argument("--conversations", type=int, default=2)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--turn-tokens", type=int, default=60000)
    ap.add_argument("--reply-tokens", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    raise SystemExit(run(a.port, a.conversations, a.turns, a.turn_tokens,
                         a.reply_tokens, a.gpu, a.log))
