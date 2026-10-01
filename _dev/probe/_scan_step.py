"""Derive per-step latency over the lifetime of each serve log.

SpecDecoding metrics lines give  Accepted throughput (tok/s) and
Mean acceptance length (tok/step)  =>  steps/s and ms/step, plus the
running/waiting counts from the neighbouring loggers.py line.
"""
import glob
import os
import re
import time

SPEC = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Mean acceptance length: ([\d.]+), "
    r"Accepted throughput: ([\d.]+)"
)
RUN = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Running: (\d+) reqs, Waiting: (\d+) reqs,"
    r" GPU KV cache usage: ([\d.]+)%"
)


def main():
    files = sorted(
        set(glob.glob(r"D:\code\vllm-windows\logs\serve_*.log")
            + glob.glob(r"D:\code\vllm-windows\_dev\out\logs\serve_*.log")),
        key=os.path.getmtime,
    )
    for f in files:
        txt = open(f, encoding="utf-8", errors="replace").read()
        runs = {m.group(1): (m.group(2), m.group(3), m.group(4)) for m in RUN.finditer(txt)}
        rows = []
        for m in SPEC.finditer(txt):
            ts, acc, tps = m.group(1), float(m.group(2)), float(m.group(3))
            if acc <= 0:
                continue
            steps = tps / acc
            r = runs.get(ts, ("?", "?", "?"))
            rows.append((ts, acc, tps, steps, 1000 / steps if steps > 0 else 0, r))
        if not rows:
            continue
        print(f"\n=== {os.path.basename(f)}")
        print("   time        acc   tok/s  step/s  ms/step  run/wait  KV%")
        for ts, acc, tps, steps, ms, r in rows:
            print(f"   {ts}  {acc:5.2f} {tps:7.2f} {steps:7.2f} {ms:8.1f}   {r[0]}/{r[1]}      {r[2]}")


if __name__ == "__main__":
    main()
