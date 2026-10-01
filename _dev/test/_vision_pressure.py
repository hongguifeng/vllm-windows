"""Vision pressure test: does a big / multi-image request push the engine back
over the VRAM cliff that cost us the 4x decode slowdown on 2026-09-22?

_vision_smoke.py answers "does the offloaded tower still work" with a 448x448
image, which only costs **196** vision tokens.  The start_server.ps1 default budget is
1,024 tokens per image (--mm-processor-kwargs longest_edge=1048576), so a real
image is ~5x bigger, and --limit-mm-per-prompt allows 32 of them per prompt.
The 18:21 boot (KV 4.2e9 / MaxLen 48000) leaves only ~626 MiB free after the
tower has run once, so this is the case worth probing.

It only *drives* the traffic and reports the vision-token count; the ITL and the
memory buckets are measured afterwards by _dev/probe/_paging_watch.ps1 -- keep
the two steps separate so the request is not slowed by the sampling loop.

  python _vision_pressure.py --port 8000 --px 1024 --count 2
"""

import argparse
import base64
import io
import json
import time
import urllib.request

PROMPT = (
    "In one short line: the background colour, the shape in the middle, and how "
    "many bars are inside it."
)


def make_image(px: int, seed: int = 0) -> bytes:
    """Yellow background, blue circle, vertical white bars inside it.

    Drawn with strokes rather than text so the content survives any resize the
    image processor applies.

    ``seed`` shifts every coordinate a few pixels.  --count copies of the
    *same* image are one encode plus N-1 MM-cache hits, so a repeat-based
    pressure test never reaches the encoder; --unique needs distinct pixels.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (px, px), (250, 205, 20))
    d = ImageDraw.Draw(img)
    off = (seed * 7) % 41
    d.ellipse((px // 5 + off, px // 5, px * 4 // 5 + off, px * 4 // 5), fill=(20, 60, 220))
    bar_w = max(6, px // 32)
    left, right = px // 3, px * 2 // 3
    n = 5
    step = (right - left) // n
    for i in range(n):
        x0 = left + i * step
        d.rectangle((x0 + off, px * 2 // 5, x0 + bar_w + off, px * 3 // 5), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--px", type=int, default=1024,
                    help="square edge; 1024 tops out the longest_edge budget")
    ap.add_argument("--count", type=int, default=2,
                    help="images in ONE request (limit-mm-per-prompt allows 32)")
    ap.add_argument("--unique", action="store_true",
                    help="make every image distinct so each one is really encoded "
                         "(without it the MM cache serves N-1 of them for free)")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    if args.unique:
        imgs = [base64.b64encode(make_image(args.px, seed=i)).decode()
                for i in range(args.count)]
        print(f"images: {args.count} DISTINCT {args.px}x{args.px}, all to be encoded")
    else:
        b64 = base64.b64encode(make_image(args.px)).decode()
        imgs = [b64] * args.count
        print(f"image: {args.px}x{args.px} x{args.count} identical in one request")

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{u}"}}
        for u in imgs
    ]
    content.append({"type": "text", "text": PROMPT})

    t0 = time.time()
    try:
        r = post(base + "/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 64, "temperature": 0,
        })
    except Exception as e:  # noqa: BLE001
        body = getattr(e, "read", lambda: b"")()
        print(f"[pressure] FAILED: {type(e).__name__}: {e}")
        if body:
            print("           " + body.decode(errors="replace")[:600])
        return 1

    el = time.time() - t0
    u = r.get("usage", {})
    mm = u.get("prompt_tokens_details", {}).get("multimodal_tokens", {})
    print(f"[pressure] {el:6.1f}s  content={r['choices'][0]['message'].get('content')!r}")
    print(f"[pressure] usage={u}")
    print(f"[pressure] vision tokens: {mm.get('image')} for {args.count} image(s) "
          f"-> {mm.get('image', 0) / args.count:.0f} tok/image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
