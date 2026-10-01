#!/usr/bin/env python3
"""Where does the CPU time inside ``step_with_batch_queue`` actually go?

A plain (no --idle) py-spy capture only records threads that are running, so the
sample count in the submit window is real CPU time: at the default 100 Hz, one
sample is ten milliseconds of on-CPU time for that thread. This script keeps the
stacks that pass through ``step_with_batch_queue`` and reports:

  * how much CPU time the submit window carries in total;
  * the leaf frames, i.e. what the CPU is executing at each sample;
  * for samples that sit in Triton's dispatch path, which call site launched it.
"""
import sys
from collections import Counter
from pathlib import Path

SUBMIT = "step_with_batch_queue"
TRITON_MARKS = (
    "triton\\runtime\\jit.py",
    "triton/runtime/jit.py",
    "triton\\backends\\nvidia\\driver.py",
    "triton/backends/nvidia/driver.py",
)


def main() -> None:
    path = Path(sys.argv[1])
    total = 0
    submit_samples = 0
    leaf = Counter()
    triton_total = 0
    launcher = Counter()
    non_submit_leaf = Counter()
    non_submit_total = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        stack, _, count = line.rpartition(" ")
        try:
            n = int(count)
        except ValueError:
            continue
        total += n
        frames = stack.split(";")
        if not frames or frames == [""]:
            continue
        leaf_frame = frames[-1]
        if any(SUBMIT in f for f in frames):
            submit_samples += n
            leaf[leaf_frame] += n
            if any(any(m in f for m in TRITON_MARKS) for f in frames):
                triton_total += n
                for f in reversed(frames):
                    if not any(any(m in f for m in TRITON_MARKS) for f in [f]):
                        launcher[f] += n
                        break
        else:
            non_submit_total += n
            non_submit_leaf[leaf_frame] += n

    print(f"total samples: {total}   inside submit: {submit_samples} "
          f"({100 * submit_samples / max(total, 1):.1f}%)   "
          f"elsewhere: {non_submit_total}")
    print(f"CPU ms represented by submit samples: "
          f"{submit_samples * 10 / 1000:.1f} ms over the capture window")

    print(f"\nleaf frames inside submit (top 20):")
    for name, n in leaf.most_common(20):
        print(f"  {n:6d}  {100 * n / max(submit_samples, 1):5.1f}%  {name}")

    print(f"\nsamples sitting in Triton dispatch: {triton_total} "
          f"({100 * triton_total / max(submit_samples, 1):5.1f}% of submit)")
    print("launched from:")
    for name, n in launcher.most_common(15):
        print(f"  {n:6d}  {name}")

    print("\nleaf frames outside submit (top 10):")
    for name, n in non_submit_leaf.most_common(10):
        print(f"  {n:6d}  {100 * n / max(non_submit_total, 1):5.1f}%  {name}")


if __name__ == "__main__":
    main()
