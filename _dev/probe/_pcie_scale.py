"""Is the host->device PCIe traffic per decode step, or a one-off per request?

Fires three requests with the same tiny prompt and different generation
lengths (1 / 64 / 384 tokens) while `nvidia-smi dmon -s t` samples PCIe
throughput.  If the transferred bytes scale with the step count, a weight
(or activation) is being re-uploaded every step.
"""
import json
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime

PORT = 8000
LOG = r"D:\code\vllm-windows\_dev\out\_pcie_scale_dmon.txt"


PROMPT = "Write a detailed story about a robot learning to paint."


def fire(max_tokens, label, marks):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = datetime.now()
    chunks = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            if raw.startswith(b"data: ") and b"[DONE]" not in raw:
                chunks += 1
    t1 = datetime.now()
    marks.append((label, t0, t1, chunks))
    print(f"[{label}] max_tokens={max_tokens} chunks={chunks} "
          f"{(t1 - t0).total_seconds():.1f}s", flush=True)


def main():
    marks = []
    dm = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "t", "-c", "90", "-o", "T"],
        stdout=open(LOG, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    time.sleep(4)
    fire(32, "A_32tok", marks)
    time.sleep(3)
    fire(160, "B_160tok", marks)
    time.sleep(3)
    fire(640, "C_640tok", marks)
    dm.wait(timeout=140)

    rows = []
    for line in open(LOG, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*(\d\d:\d\d:\d\d)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            rows.append((m.group(1), int(m.group(3)), int(m.group(4))))
    print("\n  time      rxpci   txpci   window")
    for ts, rx, tx in rows:
        tag = ""
        hh, mm, ss = (int(x) for x in ts.split(":"))
        sec = hh * 3600 + mm * 60 + ss
        for label, t0, t1, _ in marks:
            a = t0.hour * 3600 + t0.minute * 60 + t0.second
            b = t1.hour * 3600 + t1.minute * 60 + t1.second
            if a <= sec <= b:
                tag = label
        print(f"  {ts}  {rx:6d}  {tx:6d}   {tag}")

    print("\n  per-request totals (MB):")
    for label, t0, t1, ntok in marks:
        a = t0.hour * 3600 + t0.minute * 60 + t0.second
        b = t1.hour * 3600 + t1.minute * 60 + t1.second
        rx = sum(r[1] for r in rows if a <= int(r[0][:2]) * 3600 + int(r[0][3:5]) * 60 + int(r[0][6:]) <= b)
        tx = sum(r[2] for r in rows if a <= int(r[0][:2]) * 3600 + int(r[0][3:5]) * 60 + int(r[0][6:]) <= b)
        print(f"    {label:10} out={ntok:<5} rx={rx/1.0:6.0f} MB  tx={tx/1.0:6.0f} MB")


if __name__ == "__main__":
    main()
