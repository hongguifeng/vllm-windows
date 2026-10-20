"""Prove the CPU KV tier refills what it evicted.

Fills a deliberately small GPU KV pool with three long, distinct prompts, then
asks the first one again: its blocks must come back from the host tier
(CPU_to_GPU bytes rise, cached_tokens > 0) and the greedy answer must be the
same as it was before the eviction.

  python _dev/probe/_kv_offload_reload_test.py --port 9402 --model <name>
"""

import argparse
import json
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--chars", type=int, default=120000)
ap.add_argument("--tokens", type=int, default=24)
args = ap.parse_args()

BASE = f"http://127.0.0.1:{args.port}"
FILES = [
    r"D:\code\vllm-windows\vllm\models\qwen4_exp\nvidia\model.py",
    r"D:\code\vllm-windows\vllm\v1\worker\gpu\model_runner.py",
    r"D:\code\vllm-windows\_scratch\w4a16_recipe\docs\results.md",
]


def prompt_for(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        text = fh.read().replace("\r\n", "\n")
    return text[: args.chars]


def ask(prompt: str) -> dict:
    body = json.dumps(
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())


def offload_bytes() -> dict:
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=60) as resp:
        text = resp.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("vllm:kv_offload_total_bytes_total{"):
            kind = "CPU_to_GPU" if 'transfer_type="CPU_to_GPU"' in line else "GPU_to_CPU"
            out[kind] = float(line.rsplit(" ", 1)[1])
    return out


prompts = [prompt_for(path) for path in FILES]
before = offload_bytes()

first = ask(prompts[0])
text0 = first["choices"][0]["text"]
print(
    f"pass 1: prompt {first['usage']['prompt_tokens']} tok, "
    f"{first['usage']['completion_tokens']} out, "
    f"cached {first['usage'].get('prompt_tokens_details', {}).get('cached_tokens')}"
)

for i in (1, 2):
    out = ask(prompts[i])
    print(
        f"fill {i + 1}: prompt {out['usage']['prompt_tokens']} tok, "
        f"cached {out['usage'].get('prompt_tokens_details', {}).get('cached_tokens')}"
    )

after_fill = offload_bytes()
print(
    "tier after filling: "
    f"GPU_to_CPU {after_fill.get('GPU_to_CPU', 0) / 1e6:.1f} MB, "
    f"CPU_to_GPU {after_fill.get('CPU_to_GPU', 0) / 1e6:.1f} MB"
)

again = ask(prompts[0])
text1 = again["choices"][0]["text"]
cached = again["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
after = offload_bytes()
print(
    f"pass 2: prompt {again['usage']['prompt_tokens']} tok, "
    f"{again['usage']['completion_tokens']} out, cached {cached}"
)

refilled = after.get("CPU_to_GPU", 0) - after_fill.get("CPU_to_GPU", 0)
print(f"tier refill during pass 2: {refilled / 1e6:.1f} MB")

fail = []
if refilled <= 0:
    fail.append("no CPU_to_GPU traffic: the second pass did not read the tier")
if not cached:
    fail.append("cached_tokens was 0: the prompt was recomputed, not reloaded")
if text0 != text1:
    fail.append(f"output changed after the reload:\n  {text0!r}\n  {text1!r}")

if fail:
    print(f"RESULT FAIL ({len(fail)})")
    for line in fail:
        print(f"  - {line}")
    sys.exit(1)
print(f"RESULT OK (answer identical: {text1[:60]!r}...)")
