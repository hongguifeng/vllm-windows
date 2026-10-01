"""Render the two `_ab_bench.py` JSONs into the markdown tables for the report."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))              # ...\_dev\bench
DEV  = os.path.abspath(os.path.join(HERE, os.pardir))          # ...\_dev
REPO = os.path.abspath(os.path.join(DEV, os.pardir))           # D:\code\vllm-windows
OUT = os.path.join(DEV, "out", "_ab_results")


def load(label):
    with open(os.path.join(OUT, f"{label}.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    return {(r["kind"], r.get("len", r.get("conc"))): r for r in data["rows"]}


def f(v, nd=1):
    if v is None or v != v:
        return "n/a"
    return f"{v:.{nd}f}"


def main():
    labels = sys.argv[1:] or ["wsl", "win"]
    a, b = (load(x) for x in labels[:2])
    keys = sorted(set(a) | set(b), key=lambda k: (k[0], k[1]))

    print(f"| 项目 | {labels[0]} | {labels[1]} | {labels[1]}/{labels[0]} |")
    print("|---|---|---|---|")
    for k in keys:
        ra, rb = a.get(k), b.get(k)
        kind = k[0]
        if not ra or not rb:
            continue
        if kind == "prefill":
            va, vb = ra.get("tok_s"), rb.get("tok_s")
            label = f"prefill {k[1]} tok, C=1 (tok/s)"
            if not va or not vb or va != va or vb != vb:
                # one side failed to produce a measurement (e.g. the SSE
                # keep-alive client bug above 32768 tokens) -- do not render a
                # bogus 0 or a fake ratio
                print(f"| {label} | {f(va, 0) if va else '失败'} | "
                      f"{f(vb, 0) if vb else '失败'} | n/a |")
                print("|---|---|---|---|")
                continue
            ratio = vb / va
            print(f"| {label} | {f(va, 0)} | {f(vb, 0)} | {f(ratio, 2)}x |")
            va, vb = ra.get("mean_ttft_ms"), rb.get("mean_ttft_ms")
            print(f"| ↳ mean TTFT (ms) | {f(va)} | {f(vb)} | {f(vb / va, 2) if va else 'n/a'}x |")
        elif kind == "cohort":
            for key, name, nd in (("e2e_tok_s", "e2e (tok/s)", 1),
                                  ("decode_tok_s", "decode (tok/s)", 1),
                                  ("med_tpot_ms", "med TPOT (ms)", 2),
                                  ("mean_ttft_ms", "mean TTFT (ms)", 1),
                                  ("med_itl_ms", "med ITL (ms)", 2),
                                  ("tok_step", "tok/step", 2),
                                  ("dur_s", "duration (s)", 1)):
                va, vb = ra.get(key), rb.get(key)
                ratio = f(vb / va, 2) + "x" if va and va == va and va > 0 else "n/a"
                print(f"| C={k[1]} {name} | {f(va, nd)} | {f(vb, nd)} | {ratio} |")
        print("|---|---|---|---|")


if __name__ == "__main__":
    main()
