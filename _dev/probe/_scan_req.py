"""Scan every serve log for per-request timing, so decode speed can be compared
across boots (topk impl / probe flags / boot health) without re-running anything.

Usage: python _scan_req.py [--min-gen N]
"""
import argparse
import glob
import os
import re
import time

PAT = re.compile(
    r"Request finished: req (\S+) finish_reason=\S+ prompt_tokens=(\d+) "
    r"generated_tokens=(\d+) elapsed_s=([\d.]+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gen", type=int, default=0)
    ap.add_argument("--repo", default=r"D:\code\vllm-windows")
    a = ap.parse_args()

    files = sorted(
        set(glob.glob(os.path.join(a.repo, "logs", "serve_*.log"))
            + glob.glob(os.path.join(a.repo, "_dev", "out", "logs", "serve_*.log"))),
        key=os.path.getmtime,
    )
    for f in files:
        txt = open(f, encoding="utf-8", errors="replace").read()
        head = txt[:4000]
        inv = re.search(r"invocation\s+:\s*(.*)", head)
        topk = re.search(r"^\s+topk\s+:\s*(.*)", head, re.M)
        probe = re.search(r"^\s+probe\s+:\s*(.*)", head, re.M)
        rows = [(int(m.group(2)), int(m.group(3)), float(m.group(4)))
                for m in PAT.finditer(txt)]
        rows = [r for r in rows if r[1] >= a.min_gen]
        if not rows:
            continue
        tps = [g / e for _, g, e in rows]
        print(f"\n=== {os.path.basename(f)}  last={time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(f)))}"
              f"  n={len(rows)}")
        print(f"    inv  : {inv.group(1).strip() if inv else '?'}")
        print(f"    topk : {topk.group(1).strip() if topk else '?'}")
        print(f"    probe: {probe.group(1).strip() if probe else '?'}")
        print(f"    gen tok/s: med={sorted(tps)[len(tps)//2]:6.2f}  min={min(tps):6.2f}  max={max(tps):6.2f}")
        for pt, gt, el in rows[:60]:
            print(f"      prompt={pt:<7} gen={gt:<5} elapsed={el:>7.2f}s  tok/s={gt/el:6.2f}")


if __name__ == "__main__":
    main()
