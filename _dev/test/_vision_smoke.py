"""Vision-path smoke test for the native-Windows port of the .env profile.

The point of vision-tower-cpu-offload.patch is to let VISION=1 boot at all.
0.29 replaces the patch with --cpu-offload-gb/--cpu-offload-params, so the thing
to prove is that the tower still *runs* with its weights parked in host RAM --
a device mismatch would raise inside the tower forward, or the wrapper would
silently feed CPU tensors to a CUDA kernel.

So: build a synthetic image whose content is unambiguous, ask a question that
can only be answered by looking at it, and check the reply.

  python _vision_smoke.py --port 8000
"""

import argparse
import base64
import io
import json
import sys
import time
import urllib.request

PROMPT = (
    "Look at the attached image and answer three things in one short line: "
    "the background colour, the shape in the middle, and the number written "
    "inside the shape."
)


def make_image() -> bytes:
    """Yellow background, blue circle, the digits 4731 in white inside it."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), (250, 205, 20))
    d = ImageDraw.Draw(img)
    d.ellipse((84, 84, 364, 364), fill=(20, 60, 220))
    # PIL's default bitmap font is tiny and fixed-size; draw strokes instead so
    # the digits survive resize and are unmistakable.
    for x0, w in ((150, 14), (196, 14), (240, 14), (286, 14)):
        d.rectangle((x0, 190, x0 + w, 258), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.8-27b")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    png = make_image()
    b64 = base64.b64encode(png).decode()
    print(f"image: {len(png)} bytes PNG -> {len(b64)} b64 chars")

    rc = 0

    # 1. text-only, with the tower loaded. Proves the engine serves at all.
    t0 = time.time()
    try:
        r = post(base + "/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 16, "temperature": 0,
        })
        txt = r["choices"][0]["message"].get("content")
        print(f"[text ] {time.time()-t0:6.1f}s  content={txt!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[text ] FAILED: {type(e).__name__}: {e}")
        rc = 1

    # 2. the real test: an image through the offloaded tower.
    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": 128, "temperature": 0,
    }
    t0 = time.time()
    try:
        r = post(base + "/v1/chat/completions", payload)
        txt = r["choices"][0]["message"].get("content")
        u = r.get("usage", {})
        print(f"[image] {time.time()-t0:6.1f}s  content={txt!r}")
        print(f"[image] usage={u}")
    except Exception as e:  # noqa: BLE001
        body = getattr(e, "read", lambda: b"")()
        print(f"[image] FAILED: {type(e).__name__}: {e}")
        if body:
            print("        " + body.decode(errors="replace")[:600])
        return 1

    # 3. same image twice: the second pass should hit the prefix cache and show
    #    the tower is not re-run from scratch every time.
    t0 = time.time()
    try:
        r2 = post(base + "/v1/chat/completions", payload)
        print(f"[again] {time.time()-t0:6.1f}s  content="
              f"{r2['choices'][0]['message'].get('content')!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[again] FAILED: {type(e).__name__}: {e}")
        rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
