"""Vision-cap and multi-image check for the Windows flash-next service (:9393).

N images with distinct background colours and shapes go into one request at a
size that is well over the max_pixels cap, so prompt_tokens alone tell whether
--mm-processor-kwargs {"max_pixels": 1310720} is actually applied: a 2048x2048
image costs 4096 tokens uncapped and about 1296 tokens capped (one vision token
covers 32x32 px, patch 16 x merge 2).

    python _flashnext_multi_vision.py --port 9393 --images 3 --px 2048
"""

import argparse
import base64
import io
import json
import math
import time
import urllib.request

FACTOR = 32
NEEDLE = "IMPORTANT NOTE: the secret passcode is 7391-QKXM. Keep it in mind."
STYLES = [
    ((250, 205, 20), "circle", "yellow"),
    ((60, 190, 90), "square", "green"),
    ((150, 80, 230), "triangle", "purple"),
    ((70, 180, 210), "circle", "cyan"),
]


def make_image(px, color, shape):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (px, px), color)
    d = ImageDraw.Draw(img)
    cx = cy = px // 2
    r = px // 3
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(245, 245, 245))
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(245, 245, 245))
    else:
        d.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], fill=(245, 245, 245))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def round_to(px, factor):
    return int(px / factor + 0.5) * factor


def tokens_for(px, max_pixels):
    """Mirror smart_resize: rounded multiples of FACTOR, then clamp to max_pixels."""
    h = w = round_to(px, FACTOR)
    if h * w > max_pixels:
        beta = math.sqrt((px * px) / max_pixels)
        h = round_to(px / beta, FACTOR)
        w = round_to(px / beta, FACTOR)
    return (h // FACTOR) * (w // FACTOR)


def post(port, body):
    b = json.dumps(body).encode()
    r = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=b, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=600) as f:
            d = json.loads(f.read())
    except urllib.error.HTTPError as e:
        # A rejection is a result too: it shows whether --limit-mm-per-prompt is
        # actually enforced and how readable the message is.
        print(f"[multi-vision] rejected {e.code}: "
              f"{e.read().decode()[:300].strip()}")
        return time.time() - t0, None
    return time.time() - t0, d


def run(port, model, images, px, max_pixels):
    parts = [{"type": "text", "text": NEEDLE}]
    for i in range(images):
        color, shape, name = STYLES[i % len(STYLES)]
        parts.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + make_image(px, color, shape)},
        })
    parts.append({
        "type": "text",
        "text": ("The images above appear in order, image 1 first. For each image, "
                 "state its background colour and the shape drawn on it, then repeat "
                 "the passcode."),
    })
    uncapped = tokens_for(px, 16777216)
    capped = tokens_for(px, max_pixels) if max_pixels > 0 else uncapped
    print(f"[multi-vision] {images} images at {px}px: "
          f"{uncapped} tok/image uncapped vs {capped} tok/image capped")
    dt, d = post(port, {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": 64,
        "temperature": 0.0,
    })
    if d is None:
        return 1
    u = d.get("usage", {})
    print(f"[multi-vision] {dt:.1f}s  prompt={u.get('prompt_tokens')} tok "
          f"(images predicted {capped}x{images}={capped * images})")
    print("[multi-vision] answer:",
          (d["choices"][0]["message"]["content"] or "").strip().replace("\n", " | "))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9393)
    ap.add_argument("--model", default="qwen3.8-flash-next-win")
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--px", type=int, default=2048)
    ap.add_argument("--max-pixels", type=int, default=1310720,
                    help="cap the server was started with; 0 = checkpoint 4096x4096")
    a = ap.parse_args()
    raise SystemExit(run(a.port, a.model, a.images, a.px, a.max_pixels))
