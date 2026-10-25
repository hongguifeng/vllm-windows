#!/usr/bin/env python3
"""Summarize failure reasons for the Pro V2 and DeepSWE runs."""

import glob
import json
import os


def trial_dirs(pattern):
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def reward_json(d):
    for name in ("verifier/reward.json", "verifier/reward.txt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            if name.endswith(".json"):
                try:
                    return json.load(open(p))
                except Exception:
                    return None
            return open(p).read().strip()
    return None


def patch_size(d):
    for name in ("agent/model.patch", "artifacts/logs/artifacts/model.patch"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return os.path.getsize(p)
    return None


def exception_text(d):
    p = os.path.join(d, "exception.txt")
    if not os.path.exists(p):
        return ""
    return open(p, errors="replace").read()


print("== DeepSWE (AutoRound) ==")
for job in ("dswe-ar-h1", "dswe-ar-h2", "dswe-ar-e1", "dswe-ar-e2", "dswe-ar-e3", "dswe-ar-e4"):
    for d in trial_dirs(f"/home/hong/deep-swe/jobs/{job}/*"):
        r = reward_json(d)
        if r is None:
            continue
        name = os.path.basename(d).split("__")[0]
        exc = exception_text(d)
        ctx = "maximum context length" in exc
        print(f"  {job:<10} {name:<45} reward={r if not isinstance(r, dict) else r.get('reward')} "
              f"f2p={r.get('f2p_passed') if isinstance(r, dict) else '?'}/"
              f"{r.get('f2p_total') if isinstance(r, dict) else '?'} "
              f"p2p={r.get('p2p_passed') if isinstance(r, dict) else '?'}/"
              f"{r.get('p2p_total') if isinstance(r, dict) else '?'} "
              f"patch={patch_size(d)} ctx_error={ctx}")

print("\n== Pro V2 AutoRound failures ==")
for job in ("pro20-ar1", "pro20-ar1b", "pro20-ar2", "pro20-ar2b"):
    for d in trial_dirs(f"/home/hong/SWE-bench_Pro-os/jobs/{job}/instance_*"):
        p = os.path.join(d, "verifier", "reward.txt")
        if not os.path.exists(p) or open(p).read().strip() == "1":
            continue
        name = os.path.basename(d)
        out = os.path.join(d, "verifier", "output.json")
        failed = []
        if os.path.exists(out):
            try:
                data = json.load(open(out))
                failed = [t["name"] for t in data.get("tests", []) if t.get("status") != "PASSED"]
            except Exception:
                pass
        print(f"  {job:<10} {name:<40} patch={patch_size(d)} failed_tests={len(failed)}")
        for f in failed[:5]:
            print("      ", f[:110])
