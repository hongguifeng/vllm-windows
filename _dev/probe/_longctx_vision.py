"""Acceptance probe for the 2026-09-22 20:12 boot (-MaxLen 70000 / KV 5.8e9).

One single request that is *simultaneously* long-context and multimodal: a
~Nk-token filler document with a needle near the middle PLUS one full-budget
1024x1024 image, asking the model to return both the needle and the image
content.  This is the literal "70000 + vision" case -- it proves the KV budget
covers context+image in one sequence, which the `GPU KV cache size` line alone
does not.

    python _longctx_vision.py --port 8000 --target-tokens 68000
"""

import argparse
import base64
import io
import json
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
)
NEEDLE = "IMPORTANT NOTE: the secret passcode is 7391-QKXM. Keep it in mind."


def make_image(px: int) -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (px, px), (250, 205, 20))
    d = ImageDraw.Draw(img)
    cx = cy = px // 2
    r = px // 3
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 90, 220))
    for i in range(5):
        x = cx - r + (i + 1) * (2 * r) // 6
        d.rectangle([x, cy - r // 2, x + px // 40, cy + r // 2], fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tokenize(port, s):
    """Ask the server, instead of guessing chars/token (the filler is 3.48, and
    a wrong guess turns the run into a 400 'exceeds 70000' with no measurement)."""
    b = json.dumps({"model": "qwen3.8-27b", "prompt": s}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=b,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())["count"]


QUESTION = ("\n\nAnswer with exactly two lines:\n"
            "1) the secret passcode from the text above\n"
            "2) the background colour and the shape in the image")


def size_prompt(port, budget, max_ctx):
    """Fit the filler so the *finished* prompt lands as close under `budget` as
    the /tokenize endpoint can tell.

    Multiplying a per-line count by `half` undershoots by ~2.5k tokens: the chat
    template, the image placeholder and the joins are all extra.  Tokenizing the
    final string and iterating once or twice is exact, and it is the difference
    between testing "71k context" and testing "68.5k context" on a card whose
    whole point is the last 3k.
    """
    one = tokenize(port, FILLER)
    half = max((budget - 1024 - 64) // (2 * one), 1)
    for _ in range(4):
        text = FILLER * half + "\n" + NEEDLE + "\n" + FILLER * half + QUESTION
        got = tokenize(port, text)
        # +1,024 vision tokens + ~10 template/placeholder tokens
        total = got + 1034
        if abs(total - budget) <= 150 or got + 1034 > max_ctx:
            break
        half = max(int(half * (budget - 1034) / got), 1)
    print(f"[longctx+vision] sizing: text {got} tok + image 1,024 -> prompt ~{got + 1034} "
          f"(budget {budget}, ctx {max_ctx}), half={half}")
    return text


def run(port, target_tokens, px):
    # target_tokens is the *total* prompt budget: text + image + chat overhead.
    max_ctx = 71680
    text = size_prompt(port, target_tokens, max_ctx)
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + make_image(px)}},
        ]}],
        "max_tokens": 48,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        print(f"[longctx+vision] FAILED after {time.perf_counter()-t0:.1f}s: {type(e).__name__}: {e}")
        return 1
    dt = time.perf_counter() - t0
    u = d.get("usage", {})
    mm = (u.get("prompt_tokens_details") or {}).get("multimodal_tokens") or {}
    print(f"[longctx+vision] {dt:.1f}s  prompt={u.get('prompt_tokens')} tok "
          f"(image {mm.get('image', 0)})  completion={u.get('completion_tokens')}")
    print("[longctx+vision] answer:", (d["choices"][0]["message"]["content"] or "").strip().replace("\n", " | "))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--target-tokens", type=int, default=68000)
    ap.add_argument("--px", type=int, default=1024)
    a = ap.parse_args()
    raise SystemExit(run(a.port, a.target_tokens, a.px))
