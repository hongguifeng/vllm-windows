#!/usr/bin/env python3
"""Render timing tables (markdown) for the results document."""

import glob
import json
import os
import statistics
from datetime import datetime

ROOT = "/home/hong/SWE-bench_Pro-os/jobs"
DSWE = "/home/hong/deep-swe/jobs"


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def secs(block):
    if not block:
        return None
    a, b = parse(block.get("started_at")), parse(block.get("finished_at"))
    return (b - a).total_seconds() if a and b else None


def tid(trial):
    try:
        return json.load(open(os.path.join(trial, "config.json")))["task"]["path"].split("/")[-1]
    except Exception:
        return os.path.basename(trial)


def reward(trial):
    p = os.path.join(trial, "verifier", "reward.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    p = os.path.join(trial, "verifier", "reward.json")
    if os.path.exists(p):
        return "1" if json.load(open(p)).get("reward") == 1 else "0"
    return None


def collect(jobs, base):
    rows = {}
    for job in jobs:
        for trial in glob.glob(f"{base}/{job}/*"):
            rp = os.path.join(trial, "result.json")
            if not os.path.isdir(trial) or not os.path.exists(rp):
                continue
            d = json.load(open(rp))
            rows[tid(trial)] = {
                "agent": secs(d.get("agent_execution")),
                "verifier": secs(d.get("verifier")),
                "env": secs(d.get("environment_setup")),
                "total": secs(d),
                "reward": reward(trial),
            }
    return rows


deepseek = collect(["pro20-ds"], ROOT)
exl3 = collect(["pro20-exl3", "pro20-exl3-rest2"], ROOT)
autoround = collect(["pro20-ar1", "pro20-ar1b", "pro20-ar2", "pro20-ar2b"], ROOT)
dswe = collect(["dswe-ds3"], DSWE)

print("| 任务 | DeepSeek | AutoRound | EXL3 |")
print("|---|---:|---:|---:|")
for task in sorted(deepseek):
    def fmt(row):
        if not row or row.get("agent") is None:
            return "—"
        mark = "✅" if row.get("reward") == "1" else "❌"
        return f"{mark} {row['agent']:.0f}s"
    print(f"| `{task.replace('instance_', '')[:44]}` | {fmt(deepseek[task])} | {fmt(autoround.get(task))} | {fmt(exl3.get(task))} |")

print()
print("| DeepSWE 任务 | agent | verifier | 结果 |")
print("|---|---:|---:|---|")
for task in sorted(dswe):
    row = dswe[task]
    mark = "✅" if row["reward"] == "1" else "❌"
    print(f"| {task} | {row['agent']:.0f}s | {row['verifier']:.0f}s | {mark} |")
