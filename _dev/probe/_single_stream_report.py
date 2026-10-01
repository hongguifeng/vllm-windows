# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import json
from pathlib import Path

import regex as re

OUT = Path(__file__).resolve().parents[1] / "out"


def metrics(path):
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"(vllm:[^{ ]+)(?:\{.*\})? ([\d.eE+-]+)$", line)
        if match:
            values[match[1]] = float(match[2])
    return values


for size in (2048, 8192, 32768):
    label = f"buffered_single_prefill_{size}"
    data = json.loads((OUT / f"{label}.json").read_text(encoding="utf-8"))
    before = metrics(OUT / f"{label}_before.metrics")
    after = metrics(OUT / f"{label}_after.metrics")
    print(
        f"prefill {size}: "
        f"TTFT seconds={data.get('ttfts')} "
        f"input lens={data.get('input_lens')} "
        f"output lens={data.get('output_lens')}"
    )
    for name in (
        "request_prompt_tokens_sum",
        "request_prefill_kv_computed_tokens_sum",
        "prefix_cache_hits_total",
        "num_preemptions_total",
    ):
        key = "vllm:" + name
        print(f"  delta {name}: {after[key] - before[key]:.0f}")

before = metrics(OUT / "buffered_c1_spy_before.metrics")
after = metrics(OUT / "buffered_c1_spy_after.metrics")
steps = (
    after["vllm:iteration_tokens_total_count"]
    - before["vllm:iteration_tokens_total_count"]
)
print(f"\nprofile window counter delta: {steps:.0f} engine iterations")
counts = collections.Counter()
for line in (OUT / "spy_buffered_c1.raw").read_text(encoding="utf-8").splitlines():
    stack, count = line.rsplit(" ", 1)
    if "execute_model" not in stack and "sample_tokens" not in stack:
        continue
    count = int(count)
    counts["step stacks"] += count
    if "wait (threading.py" in stack:
        counts["blocked wait"] += count
    if "build_attn_metadata" in stack:
        counts["metadata builder"] += count
    if "triton" in stack:
        counts["Triton stack (overlaps metadata)"] += count
print("Sampled WALL stack residency (not CPU service time or critical-path cost):")
for name, count in counts.items():
    print(f"  {name}: {count / 100 * 1000 / steps:.3f} ms per iteration")

snapshots = json.loads(
    (OUT / "buffered_cache_disk_snapshots.json").read_text(encoding="utf-8-sig")
)
print("\nDisk raw-counter deltas; D: includes any shared-device background traffic:")
for record in snapshots:
    a, b = record["before"], record["after"]
    print(
        f"  {record['label']}: "
        f"{(b['disk_read_bytes_raw'] - a['disk_read_bytes_raw']) / 2**20:.3f} MiB, "
        f"{b['disk_read_ops_raw'] - a['disk_read_ops_raw']} ops, "
        f"avail={b['available_gib']:.3f} GiB"
    )
