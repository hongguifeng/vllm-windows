#!/usr/bin/env python3
"""Turn the ``Step stats over N steps`` log lines into a step rate.

Each line carries a timestamp and a cumulative step counter, so the difference
between two lines gives the step rate over that interval. Multiplying by tokens
per step and comparing against the measured token rate tells you whether the
step cadence and the throughput agree.

Usage: _step_rate.py <log> [--window 60]
"""
import argparse
import datetime as dt
import re
import sys

LINE = re.compile(r"INFO \d\d-\d\d (\d\d:\d\d:\d\d).*Step stats over (\d+) steps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--window", type=float, default=0.0,
                    help="only report intervals up to this many seconds apart")
    args = ap.parse_args()

    text = open(args.log, "rb").read()
    try:
        text = text.decode("utf-16-le")
    except UnicodeDecodeError:
        text = text.decode("utf-8", errors="replace")

    points = []
    for match in LINE.finditer(text):
        stamp = dt.datetime.strptime(f"2000-01-01 {match.group(1)}",
                                     "%Y-%m-%d %H:%M:%S")
        points.append((stamp, int(match.group(2))))

    if len(points) < 2:
        print("not enough step stats lines")
        return

    first, last = points[0], points[-1]
    span = (last[0] - first[0]).total_seconds()
    print(f"span {span:.1f}s: {last[1] - first[1]} steps -> "
          f"{(last[1] - first[1]) / span:.2f} steps/s (whole log)")

    best = None
    for index in range(1, len(points)):
        gap = (points[index][0] - points[index - 1][0]).total_seconds()
        if gap <= 0:
            continue
        if args.window and gap > args.window:
            continue
        rate = (points[index][1] - points[index - 1][1]) / gap
        if best is None or rate > best[1]:
            best = (points[index][0], rate)
    if best:
        print(f"fastest interval: {best[0].strftime('%H:%M:%S')} at "
              f"{best[1]:.2f} steps/s -> step cadence "
              f"{1000 / best[1]:.2f} ms")


if __name__ == "__main__":
    sys.exit(main())
