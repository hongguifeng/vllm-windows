#!/usr/bin/env python3
"""Split campaign wall time into engine-busy and engine-idle intervals.

A sequential table spends part of every pass between requests, and that part is
paid by the host while the GPU waits for the next prompt. The logger prints a
line every ten seconds with the number of running requests, so idle share can be
read from a log that already exists instead of being re-measured.
"""
import re
import subprocess
import sys

DECODE = r"D:\code\vllm-windows\_dev\probe\_log_decode.py"

FIELD = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs")


def main() -> None:
    for path in sys.argv[1:]:
        out = subprocess.run(
            [sys.executable, DECODE, path],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8", "replace")
        rows = [FIELD.search(line) for line in out.splitlines()]
        rows = [m for m in rows if m]
        if not rows:
            print(f"{path}: no logger lines")
            continue
        n = len(rows)
        idle = sum(
            1 for m in rows
            if m.group(3) == "0" and m.group(2) == "0.0")
        prefill_only = sum(
            1 for m in rows
            if m.group(3) == "0" and float(m.group(2)) == 0.0
            and float(m.group(1)) > 0.0)
        gen = [float(m.group(2)) for m in rows if m.group(3) != "0"]
        print(f"== {path.split('/')[-1]} intervals={n} "
              f"idle={idle} ({100 * idle / n:.1f}%) "
              f"prefill-only={prefill_only}")
        if gen:
            busy = sorted(gen)
            print(f"   generation tok/s while running: "
                  f"p10={busy[len(busy) // 10]:.1f} "
                  f"p50={busy[len(busy) // 2]:.1f} "
                  f"p90={busy[-len(busy) // 10]:.1f} max={busy[-1]:.1f}")


if __name__ == "__main__":
    main()
