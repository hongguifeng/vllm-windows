#!/usr/bin/env python3
"""Attribute step-thread CPU time from a py-spy raw capture, per subsystem.

Leaf-frame tallies are too flat to read here: the host cost of a step is spread
over many small dispatch operations, so the useful cut is by subsystem. A sample
counts toward a subsystem if that subsystem appears anywhere in the stack, and
toward the step only if a step marker appears at all.

Converts samples to CPU milliseconds per step given the sampling rate and the
number of steps in the capture window, which is what makes the result comparable
to a step length measured in milliseconds.

Usage:
    python _pyspy_step_shares.py RAW --steps 3069 --seconds 60
    python _pyspy_step_shares.py RAW --log LOG --from 22:47:12 --to 22:48:11

With a log and a time range, the step count comes from the engine's own
``Step timing window`` lines. Each window reports ``n`` steps over ``wall``
milliseconds a few seconds after those steps ran, so the dependable figure is the
mean period of windows overlapping the capture, not a raw sum of ``n``.
"""
import argparse
import collections
import importlib.util
import os
import re
import sys

SUBSYSTEMS = {
    "triton dispatch": ("triton",),
    "h2d / uva copies": ("async_tensor_h2d", "uva", "copy_to_uva"),
    "moe / humming": ("humming", "moe"),
    "cudagraph replay": ("cuda.graphs", "replay", "cudagraph"),
    "ple_ssd": ("ple_ssd",),
    "gdn / linear attn": ("gdn_attn", "linear_attention"),
    "qsa": ("qsa",),
    "torch op dispatch": ("torch\\_ops.py", "__torch_function__"),
    "attention backends": ("backends\\",),
    "worker plumbing": ("vllm\\v1\\worker",),
}
STEP_MARKERS = ("execute_model", "sample_tokens")
HERE = os.path.dirname(os.path.abspath(__file__))


def clock_of(line: str) -> int | None:
    """Pull a time of day out of a log line, ignoring any date prefix."""
    match = re.search(r"(\d{2}):(\d{2}):(\d{2})", line)
    if not match:
        return None
    hour, minute, second = (int(part) for part in match.groups())
    return hour * 3600 + minute * 60 + second


def to_clock(text: str) -> int:
    hour, minute, second = (int(part) for part in text.split(":"))
    return hour * 3600 + minute * 60 + second


def steps_from_log(path: str, t_from: str, t_to: str) -> tuple[int, float]:
    """Estimate the steps and seconds of a capture window from engine reports."""
    spec = importlib.util.spec_from_file_location(
        "_log_decode", os.path.join(HERE, "_log_decode.py"))
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    with open(path, "rb") as fh:
        text = decoder.decode(fh.read())

    start, end = to_clock(t_from), to_clock(t_to)
    # Windows are reported after the steps they describe finished, so widen the
    # start by the reporting cadence to catch windows whose work ran earlier.
    loose = start - 5
    total_wall = 0.0
    total_steps = 0
    for line in text.splitlines():
        if "Step timing window" not in line:
            continue
        stamp = clock_of(line)
        if stamp is None or not loose <= stamp <= end:
            continue
        count = re.search(r"n=(\d+)", line)
        wall = re.search(r"wall ([\d.]+) ms", line)
        if not (count and wall):
            continue
        total_steps += int(count.group(1))
        total_wall += float(wall.group(1))
    if not total_steps or not total_wall:
        return 0, 0.0
    period = total_wall / total_steps
    duration = float(end - start)  # clocks are already whole seconds
    return int(round(duration / period * 1000)), duration


def read_raw(path: str):
    """Yield (stack, count, leaf, parent) for every sample in a py-spy raw file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            frames = line.split(";")
            count = 1
            tail = frames[-1].rsplit(" ", 1)
            if len(tail) == 2 and tail[1].isdigit():
                frames[-1] = tail[0]
                count = int(tail[1])
            parent = frames[-2] if len(frames) >= 2 else ""
            yield "\n".join(frames), count, frames[-1], parent


def report(title: str, counter: collections.Counter, hits_total: int,
           hz: float, steps: int) -> None:
    print(title)
    for name, hits in counter.most_common(12):
        line = f"  {hits:6d}  {100 * hits / hits_total:5.1f}%  {name}"
        if steps:
            line += f"   {hits / hz * 1000 / steps:.2f} ms/step"
        print(line)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?",
                    default=r"D:\code\vllm-windows\_dev\out\spy_omp_arm.raw")
    ap.add_argument("--steps", type=int, default=0,
                    help="steps that ran during the capture window")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=100.0,
                    help="py-spy sampling rate")
    ap.add_argument("--subsystem", default="",
                    help="break one subsystem down by call site")
    ap.add_argument("--log", default="", help="host log to read step windows from")
    ap.add_argument("--from", dest="t_from", default="", help="capture start HH:MM:SS")
    ap.add_argument("--to", dest="t_to", default="", help="capture end HH:MM:SS")
    args = ap.parse_args()

    if args.steps == 0 and args.log and args.t_from and args.t_to:
        args.steps, args.seconds = steps_from_log(args.log, args.t_from, args.t_to)
        if args.steps == 0:
            print("no step windows found in that range")
            return
        print(f"derived from log: mean period {args.seconds / args.steps * 1000:.2f} ms, "
              f"{args.steps} steps over {args.seconds:.0f} s")

    total = 0
    step_samples = 0
    per_subsys = collections.Counter()
    leaves = collections.Counter()
    wait_parents = collections.Counter()
    subsys_leaves = collections.Counter()
    subsys_parents = collections.Counter()
    needles = SUBSYSTEMS.get(args.subsystem, ())
    in_subsystem = 0

    for stack, count, leaf, parent in read_raw(args.raw):
        total += count
        if not any(marker in stack for marker in STEP_MARKERS):
            continue
        step_samples += count
        leaves[leaf] += count
        if leaf.startswith("wait (") and parent:
            wait_parents[parent] += count
        for name, tokens in SUBSYSTEMS.items():
            if any(token in stack for token in tokens):
                per_subsys[name] += count
        if args.subsystem and any(token in stack for token in needles):
            in_subsystem += count
            subsys_leaves[leaf] += count
            if parent:
                subsys_parents[parent] += count

    if step_samples == 0:
        print("no step-thread samples found")
        return
    steps, hz = args.steps, args.hz
    ms_per_step = step_samples / hz * 1000 / steps if steps else 0.0
    print(f"total samples {total}, step-thread samples {step_samples} "
          f"({100 * step_samples / total:.1f}% of all)")
    print(f"step-thread CPU inside the step: {step_samples / hz:.2f} s over "
          f"{args.seconds:.0f} s wall")
    if steps:
        print(f"per step: {ms_per_step:.2f} ms CPU ({steps} steps in the window)")
    report("\nsubsystem shares of step-thread samples (a sample may count twice):",
           per_subsys, step_samples, hz, steps)
    waiting = sum(wait_parents.values())
    report("\nwho is waiting (parent of the ``wait`` leaf):",
           wait_parents, step_samples, hz, steps)
    if steps and waiting:
        print(f"  blocked in wait: {waiting / hz * 1000 / steps:.2f} ms/step of the "
              f"{ms_per_step:.2f} ms the thread is inside the step; "
              f"the rest is real CPU work")
    if args.subsystem:
        report(f"\ncall sites inside {args.subsystem!r} ({in_subsystem} samples):",
               subsys_parents, in_subsystem, hz, steps)
        report("  leaves:", subsys_leaves, in_subsystem, hz, steps)
        return
    report("\ntop leaf frames within the step:", leaves, step_samples, hz, steps)


if __name__ == "__main__":
    main()
