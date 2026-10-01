#!/usr/bin/env python3
"""Run the depth campaign against a fixed request table, one stream at a time.

One pass over the table is one measurement block. A pass is long enough that a
fast configuration cannot pull extra requests into the same window and see a
different prompt mix, which is what made the earlier window-based numbers
uncomparable. Each pass appends one JSON line with the overall rate and a rate per
class, so medians and per-class regressions can be read off the same file. A pass
with any failed request is marked incomplete and must not be counted.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _battle_table import CLASSES, table  # noqa: E402

MODEL = "qwen3.8-flash-next-full"


def ask(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(base, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        ch = data.get("choices", [{}])[0]
        return {
            "finish": ch.get("finish_reason"),
            "completion": data.get("usage", {}).get("completion_tokens", 0),
            "prompt": data.get("usage", {}).get("prompt_tokens", 0),
            "wall": time.time() - t0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc), "completion": 0, "prompt": 0,
                "wall": time.time() - t0}


def one_pass(base: str, model: str) -> dict:
    per_class: dict[str, dict[str, float]] = {
        c: {"tokens": 0.0, "wall": 0.0, "requests": 0, "errors": 0}
        for c in CLASSES
    }
    stops: dict[str, dict[str, int]] = {
        c: {"length": 0, "stop": 0, "error": 0} for c in CLASSES
    }
    t_start = time.time()
    for kind, prompt, max_tokens in table():
        r = ask(base, model, prompt, max_tokens)
        slot = per_class[kind]
        slot["requests"] += 1
        if r.get("error"):
            slot["errors"] += 1
            stops[kind]["error"] += 1
            continue
        slot["tokens"] += r["completion"]
        slot["wall"] += r["wall"]
        finish = r["finish"] or "none"
        if finish in stops[kind]:
            stops[kind][finish] += 1
    span = time.time() - t_start
    tokens = sum(s["tokens"] for s in per_class.values())
    wall = sum(s["wall"] for s in per_class.values())
    errors = sum(s["errors"] for s in per_class.values())
    record = {
        "t": int(t_start),
        "span": span,
        "tokens": tokens,
        "wall": wall,
        "rate": tokens / wall if wall else 0.0,
        "requests": sum(s["requests"] for s in per_class.values()),
        "errors": errors,
        "complete": errors == 0 and tokens > 0,
        "class_rate": {
            c: (s["tokens"] / s["wall"] if s["wall"] else 0.0)
            for c, s in per_class.items()
        },
        "class_tokens": {c: s["tokens"] for c, s in per_class.items()},
        "class_wall": {c: s["wall"] for c, s in per_class.items()},
        "class_stops": stops,
    }
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--out", default="", help="append one JSON line per pass")
    ap.add_argument("--label", default="", help="tag written into every record")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}/v1/chat/completions"

    for i in range(args.passes):
        rec = one_pass(base, args.model)
        rec["pass"] = i
        rec["label"] = args.label
        line = json.dumps(rec)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        parts = ", ".join(
            f"{c}={rec['class_rate'][c]:.1f}" for c in CLASSES)
        print(f"pass {i} ({args.label}): {rec['rate']:.1f} tok/s over "
              f"{int(rec['tokens'])} tokens / {rec['wall']:.1f}s "
              f"[{parts}] requests={rec['requests']} errors={rec['errors']} "
              f"complete={rec['complete']}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
