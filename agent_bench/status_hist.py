#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Histogram episode exit statuses, empty final patches, and token use per arm.

Use this before comparing scores: an arm dominated by `limits_exceeded` or format
errors is a harness/protocol story, not a capability story.
"""

import argparse
import glob
import json
import os
from collections import Counter


def describe(root):
    files = glob.glob(os.path.join(root, "*", "*.traj.json"))
    counts = Counter()
    empty = 0
    gen = []
    reason_chars = []
    steps = []
    for f in files:
        try:
            with open(f) as fh:
                t = json.load(fh)
        except Exception:
            continue
        info = t.get("info", {})
        counts[info.get("exit_status", "?")] += 1
        msgs = t.get("messages", [])
        patch = ""
        for m in reversed(msgs):
            if m.get("role") == "exit":
                patch = m.get("content") or ""
                break
        if not patch.strip():
            empty += 1
        stats = info.get("model_stats", {})
        if stats.get("generation_tokens"):
            gen.append(stats["generation_tokens"])
        rc = sum(len(m.get("reasoning_content") or "") for m in msgs)
        if rc:
            reason_chars.append(rc)
        steps.append(sum(1 for m in msgs if m.get("role") == "assistant"))
    print(f"\n=== {root}: {len(files)} trajectories ===")
    if not files:
        print("   no trajectories yet")
        return
    for status, n in counts.most_common():
        print(f"   {status:<22} {n:3d}  ({100.0 * n / len(files):5.1f}%)")
    print(f"   empty final patch: {empty} ({100.0 * empty / len(files):.1f}%)")
    med = lambda xs: sorted(xs)[len(xs) // 2]
    mean_gen = sum(gen) / len(gen) if gen else 0
    print(f"   median steps/episode: {med(steps)} | mean gen_tokens: {mean_gen:.0f}")
    if reason_chars:
        mean_rc = sum(reason_chars) / len(reason_chars)
        print(f"   reasoning chars/ep: {mean_rc:.0f} | median: {med(reason_chars)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DIR",
        help="arm output directory (repeatable)",
    )
    a = p.parse_args()
    for root in a.run:
        describe(root)


if __name__ == "__main__":
    main()
