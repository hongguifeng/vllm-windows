#!/usr/bin/env python3
"""Summarise a ``py-spy record --idle`` raw capture.

With --idle, blocked threads are sampled too, so the total sample count is the
wall-clock duty of every thread rather than only on-CPU time. The interesting
question is what the engine step thread is *waiting* on, since that is where
the unexplained part of a decode step lives.

Raw lines are ``frame;frame;...;frame count``.
"""
import sys
from collections import Counter
from pathlib import Path

WAIT_MARKERS = (
    "synchronize",
    "poll (zmq",
    "recv (zmq",
    "send (zmq",
    "get (queue",
    "wait (threading",
    "acquire (threading",
    "select (selectors",
    "sleep (time",
    "read (ssl",
    "connect (zmq",
)


def main() -> None:
    path = Path(sys.argv[1])
    total = 0
    step_leaf = Counter()
    wait_leaf = Counter()
    wait_marker = Counter()
    other_leaf = Counter()
    step_samples = 0

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
        leaf = frames[-1] if frames else "?"
        in_step = any("run_busy_loop" in f or "execute_model" in f
                      for f in frames)
        if in_step:
            step_samples += n
            step_leaf[leaf] += n
        else:
            other_leaf[leaf] += n
        if any(leaf.startswith(m.split(" ")[0]) or m in leaf for m in
               WAIT_MARKERS):
            wait_leaf[leaf] += n
            for m in WAIT_MARKERS:
                if m in leaf:
                    wait_marker[m] += n
                    break

    print(f"total samples: {total}")
    print(f"samples inside engine step/execute_model stack: {step_samples} "
          f"({100 * step_samples / max(total, 1):.1f}%)")

    def show(title, counter, limit=12):
        print(f"\n{title}")
        sub = sum(counter.values())
        for leaf, n in counter.most_common(limit):
            print(f"  {n:7d}  {100 * n / max(sub, 1):5.1f}%  {leaf}")
        print(f"  (group total {sub})")

    show("leaf frames while inside the step stack:", step_leaf)
    show("leaf frames everywhere else (other threads / idle):", other_leaf)
    show("leaf frames that look like a wait:", wait_leaf, 15)
    show("which wait marker:", wait_marker, 15)


if __name__ == "__main__":
    main()
