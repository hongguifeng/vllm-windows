#!/usr/bin/env python3
"""Put the in-graph phase table back into step units.

Phase spans are read from event pairs, so their absolute scale depends on how
far the device queue runs ahead of the reader: every pair is inflated by the
same factor. The step timing collector measures the same forward with host
driven marks, so the ratio between the two medians gives that factor and the
table can be divided back out.

Spans may come back negative when the elapsed time arguments were flipped; the
magnitudes are still the same, so this takes absolute values and says so.
"""
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_decode import decode  # noqa: E402

PHASE_LINE = re.compile(r"PHASES harvest=(\d+) (.*)")
PHASE_VALUE = re.compile(r"(\S+)=(-?[\d.]+)/(-?[\d.]+)/(-?[\d.]+) n=(\d+)")
STEP_WINDOW = re.compile(
    r"Step timing window \(full, tokens=\d+, n=\d+, wall [\d.]+ ms\): "
    r"forward p50 ([\d.]+) p95 [\d.]+; drafter p50 ([\d.]+)"
)


def main() -> None:
    path = sys.argv[1]
    with open(path, "rb") as fh:
        text = decode(fh.read())

    series: dict[str, list[float]] = {}
    counts: dict[str, list[int]] = {}
    lines = 0
    for m in PHASE_LINE.finditer(text):
        lines += 1
        for v in PHASE_VALUE.finditer(m.group(2)):
            name, p50, _p90, _mx, n = v.groups()
            series.setdefault(name, []).append(abs(float(p50)))
            counts.setdefault(name, []).append(int(n))

    forward: list[float] = []
    drafter: list[float] = []
    for m in STEP_WINDOW.finditer(text):
        forward.append(float(m.group(1)))
        drafter.append(float(m.group(2)))

    if not lines or not series.get("model") or not forward:
        print("not enough data: phases=%d model series=%d forward series=%d" % (
            lines, len(series.get("model", [])), len(forward)))
        return

    model = statistics.median(series["model"])
    fwd = statistics.median(forward)
    factor = model / fwd
    print("phase table lines=%d  model p50=%.3f  forward p50=%.3f  "
          "factor=%.2f  drafter p50=%.3f" % (
              lines, model, fwd, factor, statistics.median(drafter)))

    print("\nrescaled to step units (ms), by magnitude of median:")
    rows = []
    for name, values in series.items():
        med = statistics.median(values)
        rows.append((med / factor, name, med, statistics.median(counts[name])))
    for scaled, name, raw, n in sorted(rows, reverse=True):
        print("  %-14s raw %8.3f  step %7.4f  samples/line %7d" % (
            name, raw, scaled, n))

    print("\nratios that do not depend on the factor:")
    def ratio(a, b):
        return statistics.median(series[a]) / statistics.median(series[b])
    print("  layer1 inner shares of layer1: ple %.3f attn %.3f mlp %.3f" % (
        ratio("layer1.ple", "layer1"), ratio("layer1.attn", "layer1"),
        ratio("layer1.mlp", "layer1")))
    print("  layer32 inner shares of layer32: attn %.3f mlp %.3f" % (
        ratio("layer32.attn", "layer32"), ratio("layer32.mlp", "layer32")))
    for name in ("layer0", "layer1", "layer3", "layer16", "layer32", "layer47"):
        print("  %s / model = %.4f" % (name, ratio(name, "model")))
    print("  logits / model = %.4f" % ratio("logits", "model"))


if __name__ == "__main__":
    main()
