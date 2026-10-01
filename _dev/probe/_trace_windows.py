#!/usr/bin/env python3
"""Tabulate PLE trace decode windows and correlate row cost with step length.

Windows within one run differ in content, so the row cost swings by an order of
magnitude while the ids wait stays put. If the step length does not follow the
row cost, the row path is not on the critical path, and that can be settled from
a log that was already produced rather than by another restart.
"""
import re
import subprocess
import sys

DECODE = r"D:\code\vllm-windows\_dev\probe\_log_decode.py"
TOKENS = 5


def as_path(text: str) -> str:
    """Accept a shell path or a Windows path for the same file."""
    hit = re.match(r"^/([a-zA-Z])/(.*)$", text)
    return f"{hit.group(1).upper()}:/{hit.group(2)}" if hit else text


def parse(path: str) -> dict[str, dict[str, float]]:
    out = subprocess.run(
        [sys.executable, DECODE, as_path(path), "PLE trace"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", "replace")
    windows: dict[str, dict[str, float]] = {}
    for line in out.splitlines():
        stamp = re.search(r"(\d\d:\d\d:\d\d) \[ple_ssd", line)
        if not stamp or f"tokens={TOKENS}" not in line:
            continue
        key = stamp.group(1)
        if "chain" in line:
            win = windows.setdefault(key, {})
            win["chain"] = float(
                re.search(r"chain \(tokens=\d+\): p50 ([\d.]+)", line).group(1))
            win["rows"] = float(re.search(r"rows ([\d.]+)/", line).group(1))
            win["ids"] = float(re.search(r" ids ([\d.]+)/", line).group(1))
            win["pending"] = float(
                re.search(r"pending ([\d.]+)/", line).group(1))
        elif "step gap" in line:
            win = windows.setdefault(key, {})
            win["gap"] = float(
                re.search(r"landed p50 ([\d.]+)", line).group(1))
    return {k: v for k, v in windows.items() if "gap" in v and "rows" in v}


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n
    vx = sum((a - mx) ** 2 for a in xs) / n
    vy = sum((b - my) ** 2 for b in ys) / n
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / (vx**0.5 * vy**0.5)


def main() -> None:
    for path in sys.argv[1:]:
        windows = parse(path)
        keys = sorted(windows)
        if len(keys) < 6:
            print(f"{path}: {len(keys)} usable windows")
            continue
        print(f"== {path.split('/')[-1]} windows={len(keys)}")
        print(f"{'t':9}{'rows':>8}{'ids':>8}{'chain':>8}{'gap':>8}{'pend':>8}")
        for key in keys:
            win = windows[key]
            print(f"{key:9}{win['rows']:8.0f}{win['ids']:8.0f}"
                  f"{win['chain']:8.0f}{win['gap']:8.0f}{win['pending']:8.0f}")
        rows = [windows[k]["rows"] for k in keys]
        gaps = [windows[k]["gap"] for k in keys]
        ids = [windows[k]["ids"] for k in keys]
        print(f"rows {min(rows):.0f}-{max(rows):.0f} us, r(rows,gap)="
              f"{pearson(rows, gaps):+.3f}")
        print(f"ids {min(ids):.0f}-{max(ids):.0f} us, r(ids,gap)="
              f"{pearson(ids, gaps):+.3f}")


if __name__ == "__main__":
    main()
