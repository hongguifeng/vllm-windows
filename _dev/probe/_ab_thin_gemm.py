"""A/B driver: run the standard decode bench N times and summarise.

Runs `_bench_port.py` against a live endpoint, parses its decode line, and prints
the median across repeats so two arms are compared the same way.

  .venv\\Scripts\\python.exe _dev\\probe\\_ab_thin_gemm.py --port 9402 --arm off --reps 3
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv" / "Scripts" / "python.exe"
BENCH = REPO / "_dev" / "probe" / "_bench_port.py"

LINE = re.compile(
    r"decode: (\d+) tok in ([\d.]+)s -> ([\d,\.]+) tok/s aggregate, "
    r"median ITL ([\d.]+) ms"
)


def run_once(port: int, model: str, conc: int, decode_tokens: int,
             prefill_len: int, label: str) -> dict:
    cmd = [
        str(PY), str(BENCH),
        "--port", str(port),
        "--model", model,
        "--prefill-len", str(prefill_len),
        "--decode-tokens", str(decode_tokens),
        "--conc", str(conc),
        "--label", label,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    text = out.stdout + out.stderr
    m = LINE.search(text)
    if not m:
        print(text[-1500:])
        raise SystemExit(f"no decode line from bench (exit {out.returncode})")
    tokens, wall, rate, itl = m.groups()
    return {
        "tokens": int(tokens),
        "wall_s": float(wall),
        "tok_s": float(rate.replace(",", "")),
        "itl_ms": float(itl),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--prefill-len", type=int, default=4096)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    for i in range(args.reps):
        r = run_once(args.port, args.model, args.conc, args.decode_tokens,
                     args.prefill_len, f"{args.arm}#{i + 1}")
        rows.append(r)
        print(f"  rep {i + 1}: {r['tok_s']:,.1f} tok/s  ITL {r['itl_ms']:.2f} ms  "
              f"({r['tokens']} tok / {r['wall_s']:.2f}s)", flush=True)

    summary = {
        "arm": args.arm,
        "port": args.port,
        "conc": args.conc,
        "decode_tokens": args.decode_tokens,
        "itl_ms_median": statistics.median(r["itl_ms"] for r in rows),
        "tok_s_median": statistics.median(r["tok_s"] for r in rows),
        "itl_ms_all": [r["itl_ms"] for r in rows],
        "tok_s_all": [r["tok_s"] for r in rows],
    }
    print(f"[{args.arm}] conc={args.conc}: median ITL "
          f"{summary['itl_ms_median']:.2f} ms, median {summary['tok_s_median']:,.1f} tok/s")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
