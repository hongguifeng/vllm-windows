#!/usr/bin/env python3
"""Summarize campaign passes: medians overall and per class, with deltas.

Depth is only worth shipping if no class it matters for regresses repeatably, so
the per-class medians carry the same weight as the overall one. Reads the JSON
lines written by _battle_pass.py and ignores warm-up and discarded passes by
label, which keeps the choice explicit rather than buried in the script.
"""
import argparse
import json
import statistics
import sys

COUNTED = ("m1", "m2", "m3")


def load(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "warm" in rec["label"]:
                continue
            if rec["label"].endswith(COUNTED) and rec["complete"]:
                records.append(rec)
    return records


def medians(records: list[dict]) -> tuple[float, dict[str, float]]:
    overall = statistics.median(r["rate"] for r in records)
    per_class: dict[str, float] = {}
    classes = sorted(records[0]["class_rate"])
    for name in classes:
        per_class[name] = statistics.median(
            r["class_rate"][name] for r in records)
    return overall, per_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--labels", nargs="+", default=[])
    args = ap.parse_args()
    labels = args.labels or [p.rsplit(".", 1)[0] for p in args.paths]

    data = {}
    for label, path in zip(labels, args.paths):
        records = load(path)
        if len(records) < 3:
            print(f"{label}: only {len(records)} counted passes, skipped")
            continue
        data[label] = medians(records)
        print(f"{label}: n={len(records)} median {data[label][0]:.1f} tok/s")

    if "x2" not in data:
        sys.exit("no x2 baseline to compare against")
    base_overall, base_class = data["x2"]
    for label in labels:
        if label == "x2":
            continue
        overall, per_class = data[label]
        print(f"{label} vs x2: overall "
              f"{100 * (overall / base_overall - 1):+.1f}%")
        for name in sorted(per_class):
            delta = 100 * (per_class[name] / base_class[name] - 1)
            flag = " REGRESS" if delta <= -5.0 else ""
            print(f"    {name:8s} {per_class[name]:6.1f} "
                  f"({delta:+.1f}%){flag}")


if __name__ == "__main__":
    main()
