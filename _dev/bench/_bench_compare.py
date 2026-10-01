"""Compare the SUMMARY blocks of several _bench_27b.py runs side by side.

    python _bench_compare.py baseline patched

Reads _bench_<tag>_r1.log / _r2.log, prints the second (warm) run of each and
the delta against the first tag given.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))              # ...\_dev\bench
DEV  = os.path.abspath(os.path.join(HERE, os.pardir))          # ...\_dev
REPO = os.path.abspath(os.path.join(DEV, os.pardir))           # D:\code\vllm-windows

# order matters: first tag is the reference
ROWS = [
    ("engine_init_s", "engine init", "s", False),
    ("prefill_s", "  prefill 8.8k", "s", False),
    ("prefill_tok_s", "  prefill", "tok/s", True),
    ("decode_s", "  decode 256 tok", "s", False),
    ("decode_tok_s", "  decode bs=1", "tok/s", True),
    ("batch_s", "  decode 8x256", "s", False),
    ("batch_tok_s", "  decode bs=8", "tok/s", True),
    ("batch_prefill_s", "  batch prefill", "s", False),
    ("batch_prefill_tok_s", "  batch prefill", "tok/s", True),
    ("peak_alloc_gib", "peak alloc", "GiB", False),
    ("peak_reserved_gib", "peak reserved", "GiB", False),
]


def load(tag, run=2):
    path = os.path.join(DEV, "out", f"_bench_{tag}_r{run}.log")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    match = re.search(r"\n\{\n(?:.*\n)*?\}\n", text)
    if not match:
        return None
    return json.loads(match.group(0))


def main():
    tags = sys.argv[1:] or ["baseline"]
    runs = {}
    for tag in tags:
        data = load(tag, 2) or load(tag, 1)
        if data is None:
            print(f"  !! no usable log for tag {tag}")
            continue
        runs[tag] = data

    if not runs:
        return 1

    name_width = max(len(label) for _k, label, _u, _h in ROWS) + 1
    col_width = max(14, max(len(t) for t in tags) + 4)

    header = "metric".ljust(name_width) + "".join(t.rjust(col_width) for t in tags)
    if len(tags) > 1:
        header += f"{'delta vs ' + tags[0]:>26}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for key, label, unit, higher_is_better in ROWS:
        line = label.ljust(name_width)
        base = None
        for tag in tags:
            value = runs[tag].get(key)
            if value is None:
                line += "n/a".rjust(col_width)
                continue
            line += f"{value:>13,.1f} {unit}".rjust(col_width)
            if tag == tags[0]:
                base = value
        if len(tags) > 1 and base not in (None, 0):
            last = runs[tags[-1]].get(key)
            if last is not None:
                pct = (last - base) / abs(base) * 100.0
                arrow = "+" if pct >= 0 else ""
                mark = ""
                # a metric moving the "wrong" way is worth calling out
                if (pct > 0) != higher_is_better and abs(pct) > 1.0:
                    mark = "  <-- slower" if not higher_is_better else "  <-- lower"
                line += f"{arrow}{pct:6.1f}%{mark}".rjust(26)
        print(line)

    print("=" * len(header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
