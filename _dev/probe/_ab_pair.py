"""Paired A/B of two live endpoints, alternating so drift cancels.

Runs the same bench request against arm A and arm B alternately, `pairs` times,
and reports the median per-arm ITL and the median of the per-pair differences
(ON - OFF), which is robust to a slow thermal/clock drift that would bias a
"all of A, then all of B" comparison.

  .venv\\Scripts\\python.exe _dev\\probe\\_ab_pair.py --a 9393 --b 9402 --pairs 6
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv" / "Scripts" / "python.exe"
BENCH = REPO / "_dev" / "probe" / "_bench_port.py"


def once(port: int, model: str, conc: int, decode_tokens: int, prefill: int,
         label: str) -> tuple[float, float]:
    cmd = [str(PY), str(BENCH), "--port", str(port), "--model", model,
           "--prefill-len", str(prefill), "--decode-tokens", str(decode_tokens),
           "--conc", str(conc), "--label", label]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    text = out.stdout + out.stderr
    m = re.search(r"decode: \d+ tok in [\d.]+s -> ([\d,\.]+) tok/s aggregate, "
                  r"median ITL ([\d.]+) ms", text)
    if not m:
        raise SystemExit(f"no decode line for {label}:\n{text[-800:]}")
    tok = float(m.group(1).replace(",", ""))
    itl = float(m.group(2))
    return itl, tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, required=True, help="baseline port")
    ap.add_argument("--b", type=int, required=True, help="treatment port")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--prefill-len", type=int, default=4096)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    a_itl, a_tok, b_itl, b_tok = [], [], [], []
    for i in range(args.pairs):
        order = [(args.a, "a"), (args.b, "b")]
        if i % 2:
            order.reverse()
        for port, arm in order:
            itl, tok = once(port, args.model, args.conc, args.decode_tokens,
                            args.prefill_len, f"{arm}#{i + 1}")
            (a_itl if arm == "a" else b_itl).append(itl)
            (a_tok if arm == "a" else b_tok).append(tok)
            print(f"  pair {i + 1} {arm.upper()} ({port}): ITL {itl:.2f} ms, "
                  f"{tok:,.1f} tok/s", flush=True)

    d_itl = [b - a for a, b in zip(a_itl, b_itl)]
    d_tok = [(b / a - 1) * 100 for a, b in zip(a_tok, b_tok)]
    summary = {
        "conc": args.conc,
        "a_port": args.a,
        "b_port": args.b,
        "a_itl_ms": a_itl,
        "b_itl_ms": b_itl,
        "a_tok_s": a_tok,
        "b_tok_s": b_tok,
        "itl_a_median": statistics.median(a_itl),
        "itl_b_median": statistics.median(b_itl),
        "itl_change_median_pct": statistics.median(d_itl) / statistics.median(a_itl) * 100,
        "tok_s_a_median": statistics.median(a_tok),
        "tok_s_b_median": statistics.median(b_tok),
        "tok_s_change_median_pct": statistics.median(d_tok),
    }
    print(f"\nA (baseline, :{args.a}): ITL {summary['itl_a_median']:.2f} ms, "
          f"{summary['tok_s_a_median']:,.1f} tok/s")
    print(f"B (treatment, :{args.b}): ITL {summary['itl_b_median']:.2f} ms, "
          f"{summary['tok_s_b_median']:,.1f} tok/s")
    print(f"paired change: ITL {summary['itl_change_median_pct']:+.1f}%, "
          f"throughput {summary['tok_s_change_median_pct']:+.1f}% "
          f"(median of {args.pairs} pairs)")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
