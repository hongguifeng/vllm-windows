#!/usr/bin/env python3
"""Extract per-trial phase timings from Harbor/Pier job directories."""

import glob
import json
import os
import statistics
from datetime import datetime

ROOT = "/home/hong/SWE-bench_Pro-os/jobs"
DSWE = "/home/hong/deep-swe/jobs"
ARMS = {
    "ProV2-DeepSeek": ("swebench-pro", ["pro20-ds"]),
    "ProV2-EXL3": ("swebench-pro", ["pro20-exl3", "pro20-exl3-rest2"]),
    "ProV2-AutoRound": ("swebench-pro", ["pro20-ar1", "pro20-ar1b", "pro20-ar2", "pro20-ar2b"]),
    "DeepSWE-DeepSeek": ("datacurve", ["dswe-ds3"]),
}


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def seconds(block):
    if not block:
        return None
    start, end = parse(block.get("started_at")), parse(block.get("finished_at"))
    if not start or not end:
        return None
    return (end - start).total_seconds()


def task_id(trial_dir):
    cfg = os.path.join(trial_dir, "config.json")
    try:
        return json.load(open(cfg))["task"]["path"].split("/")[-1]
    except Exception:
        return os.path.basename(trial_dir)


def main():
    summary = {}
    per_task = {}
    for arm, (prefix, jobs) in ARMS.items():
        rows = []
        for job in jobs:
            base = DSWE if job.startswith("dswe") else ROOT
            for trial in glob.glob(f"{base}/{job}/*"):
                if not os.path.isdir(trial):
                    continue
                result_path = os.path.join(trial, "result.json")
                if not os.path.exists(result_path):
                    continue
                data = json.load(open(result_path))
                reward_path = os.path.join(trial, "verifier", "reward.txt")
                reward = None
                if os.path.exists(reward_path):
                    reward = open(reward_path).read().strip()
                rows.append(
                    {
                        "task": task_id(trial),
                        "job": job,
                        "reward": reward,
                        "total": seconds(data),
                        "env": seconds(data.get("environment_setup")),
                        "setup": seconds(data.get("agent_setup")),
                        "agent": seconds(data.get("agent_execution")),
                        "verifier": seconds(data.get("verifier")),
                    }
                )
        agents = [r["agent"] for r in rows if r["agent"]]
        vers = [r["verifier"] for r in rows if r["verifier"]]
        envs = [r["env"] for r in rows if r["env"]]
        summary[arm] = {
            "n": len(rows),
            "resolved": sum(1 for r in rows if r["reward"] == "1"),
            "agent_median_s": round(statistics.median(agents), 1) if agents else None,
            "agent_mean_s": round(statistics.mean(agents), 1) if agents else None,
            "agent_min_s": round(min(agents), 1) if agents else None,
            "agent_max_s": round(max(agents), 1) if agents else None,
            "verifier_median_s": round(statistics.median(vers), 1) if vers else None,
            "env_median_s": round(statistics.median(envs), 1) if envs else None,
        }
        per_task[arm] = sorted(rows, key=lambda r: r["task"])
    print(json.dumps(summary, indent=1))
    with open("/home/hong/bench/timings.json", "w") as handle:
        json.dump({"summary": summary, "per_task": per_task}, handle, indent=1)
    for arm, rows in per_task.items():
        print(f"\n== {arm} ==")
        for row in rows:
            print(
                f"{row['task'][:58]:<58} reward={str(row['reward']):<4} "
                f"agent={row['agent'] and round(row['agent'])}s "
                f"verifier={row['verifier'] and round(row['verifier'])}s "
                f"total={row['total'] and round(row['total'])}s"
            )


if __name__ == "__main__":
    main()
