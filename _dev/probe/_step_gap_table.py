#!/usr/bin/env python3
"""Turn the engine's ``Step timing window`` lines into a per-step idle table.

Each window reports how many steps ran in how much wall time, plus p50 of the
forward, the drafter and the whole step span. The wall divided by the step count
is the cadence the engine actually kept; the span is how long one step occupied
the device. Their difference is how long the GPU sat with nothing queued, which is
the quantity a host-side optimization has to be measured against.

Usage:
    python _step_gap_table.py LOG
    python _step_gap_table.py LOG --from 22:52:41 --to 22:53:42
    python _step_gap_table.py LOG --last 12
"""
import argparse
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def decode(path: str) -> str:
    spec = importlib.util.spec_from_file_location(
        "_log_decode", os.path.join(HERE, "_log_decode.py"))
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    with open(path, "rb") as fh:
        return decoder.decode(fh.read())


def clock_of(line: str) -> int | None:
    match = re.search(r"(\d{2}):(\d{2}):(\d{2})", line)
    if not match:
        return None
    hour, minute, second = (int(part) for part in match.groups())
    return hour * 3600 + minute * 60 + second


def to_clock(text: str) -> int:
    hour, minute, second = (int(part) for part in text.split(":"))
    return hour * 3600 + minute * 60 + second


def parse_windows(text: str):
    """Yield one record per reported step window."""
    for line in text.splitlines():
        if "Step timing window" not in line:
            continue
        stamp = clock_of(line)
        count = re.search(r"n=(\d+)", line)
        wall = re.search(r"wall ([\d.]+) ms", line)
        if not (stamp and count and wall):
            continue
        record = {
            "clock": stamp,
            "n": int(count.group(1)),
            "wall": float(wall.group(1)),
        }
        for name, pattern in (
            ("forward", r"forward p50 ([\d.]+)"),
            ("drafter", r"drafter p50 ([\d.]+)"),
            ("span", r"step span p50 ([\d.]+)"),
            ("busy", r"device busy ([\d.]+)%"),
        ):
            match = re.search(pattern, line)
            if match:
                record[name] = float(match.group(1))
        yield record


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--from", dest="t_from", default="")
    ap.add_argument("--to", dest="t_to", default="")
    ap.add_argument("--last", type=int, default=0)
    args = ap.parse_args()

    records = list(parse_windows(decode(args.log)))
    if args.t_from and args.t_to:
        start, end = to_clock(args.t_from), to_clock(args.t_to)
        records = [r for r in records if start <= r["clock"] <= end]
    if args.last:
        records = records[-args.last:]
    if not records:
        print("no step windows in range")
        return

    print(f"{'clock':9s} {'n':>4s} {'period':>7s} {'forward':>8s} "
          f"{'drafter':>8s} {'span':>7s} {'idle':>6s} {'busy':>5s}")
    totals = {"wall": 0.0, "n": 0, "idle": 0.0, "busy": 0.0}
    busy_count = 0
    for r in records:
        period = r["wall"] / r["n"]
        span = r.get("span")
        idle = period - span if span is not None else float("nan")
        clock = f"{r['clock'] // 3600:02d}:{r['clock'] % 3600 // 60:02d}:" \
                f"{r['clock'] % 60:02d}"
        print(f"{clock:9s} {r['n']:4d} {period:7.2f} "
              f"{r.get('forward', float('nan')):8.3f} "
              f"{r.get('drafter', float('nan')):8.3f} "
              f"{float('nan') if span is None else span:7.3f} "
              f"{idle:6.3f} {r.get('busy', float('nan')):5.1f}")
        totals["wall"] += r["wall"]
        totals["n"] += r["n"]
        if span is not None:
            totals["idle"] += (period - span) * r["n"]
        if "busy" in r:
            totals["busy"] += r["busy"]
            busy_count += 1

    period = totals["wall"] / totals["n"]
    print(f"\naggregate over {len(records)} windows: {totals['n']} steps, "
          f"mean period {period:.2f} ms, "
          f"mean idle {totals['idle'] / totals['n']:.2f} ms/step, "
          f"mean busy {totals['busy'] / max(busy_count, 1):.1f}%")
    print("idle here means cadence minus the device span of a step.")


if __name__ == "__main__":
    main()
