#!/usr/bin/env python3
"""Summarize attention metadata build timing lines against the step idle gap.

The engine logs a window every N builds with the p50 wall time of the whole
build and a split by batch size. This turns those into a per step cost by
counting decode steps in the same time range from the step timing windows, so
the build cost can be compared directly with the idle gap of the same range.

Usage:
    python _attn_build_report.py LOG --from 23:13:19 --to 23:14:19
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


def in_range(stamp: int, start: int | None, end: int | None) -> bool:
    if start is not None and stamp < start:
        return False
    if end is not None and stamp > end:
        return False
    return True


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--from", dest="t_from", default="")
    ap.add_argument("--to", dest="t_to", default="")
    args = ap.parse_args()

    start = to_clock(args.t_from) if args.t_from else None
    end = to_clock(args.t_to) if args.t_to else None
    text = decode(args.log)

    # Step counts and the idle gap come from the engine's own step windows.
    spec = importlib.util.spec_from_file_location(
        "_step_gap_table", os.path.join(HERE, "_step_gap_table.py"))
    gap_table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gap_table)
    steps = 0
    wall = 0.0
    idle_ms = 0.0
    spans: list[float] = []
    for record in gap_table.parse_windows(text):
        if not in_range(record["clock"], start, end):
            continue
        steps += record["n"]
        wall += record["wall"]
        if "span" in record:
            spans.append(record["span"])
            idle_ms += (record["wall"] / record["n"] - record["span"]) * record["n"]

    calls: dict[int, tuple[int, float]] = {}
    builders: dict[str, tuple[int, float]] = {}
    origins: dict[str, tuple[int, float]] = {}
    totals: list[float] = []
    p95s: list[float] = []
    windows = 0
    first = last = None
    for line in text.splitlines():
        if "Attn build" not in line:
            continue
        stamp = clock_of(line)
        if stamp is None or not in_range(stamp, start, end):
            continue
        if "Attn build origin window" in line:
            for name, count, _p50, total_ms in re.findall(
                    r"([A-Za-z_0-9]+\.py:\d+) n=(\d+) p50 ([\d.]+) ms "
                    r"total ([\d.]+) ms",
                    line,
            ):
                prev_count, prev_total = origins.get(name, (0, 0.0))
                origins[name] = (
                    prev_count + int(count),
                    prev_total + float(total_ms),
                )
            continue
        overall = re.search(
            r"p50 total ([\d.]+) ms p95 ([\d.]+) ms", line)
        if not overall:
            continue
        windows += 1
        totals.append(float(overall.group(1)))
        p95s.append(float(overall.group(2)))
        first = stamp if first is None else first
        last = stamp
        for size, count, value in re.findall(
                r"tokens=(\d+) n=(\d+) p50 ([\d.]+) ms", line):
            key = int(size)
            prev_count, prev_sum = calls.get(key, (0, 0.0))
            calls[key] = (
                prev_count + int(count),
                prev_sum + int(count) * float(value),
            )
        for name, count, _p50, total_ms in re.findall(
                r"([A-Za-z_][A-Za-z_0-9]*) n=(\d+) p50 ([\d.]+) ms "
                r"total ([\d.]+) ms",
                line,
        ):
            prev_count, prev_total = builders.get(name, (0, 0.0))
            builders[name] = (
                prev_count + int(count),
                prev_total + float(total_ms),
            )

    if not windows:
        print("no attn build timing lines in that range")
        return

    span_seconds = (last - first) if last is not None and first is not None else 0
    calls_per_window = sum(count for count, _ in calls.values())
    print(f"attn build windows: {windows} over {span_seconds:.0f} s, "
          f"{calls_per_window} calls")
    print(f"mean p50 total {sum(totals) / len(totals):.3f} ms, "
          f"mean p95 {sum(p95s) / len(p95s):.3f} ms")

    if steps:
        seconds = wall / 1000.0
        step_rate = steps / max(seconds, 1e-9)
        calls_per_step = calls_per_window / max(steps, 1)
        print(f"same range in step windows: {steps} steps over "
              f"{seconds:.1f} s ({step_rate:.1f} steps/s, "
              f"{calls_per_step:.2f} calls per step)")
        print(f"{'tokens':>7s} {'calls':>7s} {'share':>6s} "
              f"{'p50 ms/call':>12s} {'ms/step':>8s}")
        for key, (count, total_ms) in sorted(
                calls.items(), key=lambda item: -item[1][0]):
            p50 = total_ms / max(count, 1)
            share = 100.0 * count / max(calls_per_window, 1)
            per_step = calls_per_step * (count / max(calls_per_window, 1)) * p50
            print(f"{key:7d} {count:7d} {share:6.1f} {p50:12.3f} "
                  f"{per_step:8.3f}")
        builders_rows = sorted(
            builders.items(), key=lambda item: -item[1][1])[:8]
        if builders_rows:
            print(f"\n{'builder':44s} {'calls':>6s} "
                  f"{'total ms':>9s} {'ms/step':>8s} {'share':>6s}")
            builder_total = sum(total for _, (_, total) in builders.items())
            for name, (count, total_ms) in builders_rows:
                print(f"{name:44s} {count:6d} {total_ms:9.1f} "
                      f"{total_ms / max(steps, 1):8.4f} "
                      f"{100.0 * total_ms / max(builder_total, 1e-9):6.1f}")

        origin_rows = sorted(
            origins.items(), key=lambda item: -item[1][1])[:6]
        if origin_rows:
            print(f"\n{'call site':28s} {'calls':>6s} "
                  f"{'total ms':>9s} {'ms/step':>8s} {'p50 ms/call':>12s}")
            for name, (count, total_ms) in origin_rows:
                print(f"{name:28s} {count:6d} {total_ms:9.1f} "
                      f"{total_ms / max(steps, 1):8.4f} "
                      f"{total_ms / max(count, 1):12.3f}")

        weighted = sum(
            (count / max(calls_per_window, 1)) * (total_ms / max(count, 1))
            for count, total_ms in calls.values())
        print(f"\nbuild time per step (call rate weighted): "
              f"{calls_per_step * weighted:.3f} ms")
        print(f"idle gap per step in the same range: "
              f"{idle_ms / max(steps, 1):.3f} ms")
        print("the idle gap is the ceiling on what removing host work can buy.")
    else:
        print("no step windows in that range, so no per step conversion")


if __name__ == "__main__":
    main()
