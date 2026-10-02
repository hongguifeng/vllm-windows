"""One protocol, two servers -- the WSL <-> native-Windows A/B driver.

The point of this file is that the *client* is held fixed. `vllm bench serve`
here is the Windows venv's 0.29 client and it drives both the WSL container on
18020 and the native Windows server on 8000, so neither side's bench client can
flatter its own server. Protocol is `bench/run_benchmarks.sh single` from the
WSL repo: the same warmup, the same real-prompt cohorts, the same prefill
matrix, the same per-call distinct `--seed`.

Usage (cwd must be outside the repo, see project convention 5):

    python _ab_bench.py --port 18020 --label wsl
    python _ab_bench.py --port 8000  --label win

Writes <outdir>/<label>.json plus one log per call.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))              # ...\_dev\bench
DEV  = os.path.abspath(os.path.join(HERE, os.pardir))          # ...\_dev
REPO = os.path.abspath(os.path.join(DEV, os.pardir))           # D:\code\vllm-windows
# Convention 5: the unbuilt vllm/ source tree shadows the installed package.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (REPO, HERE)]

VLLM = os.path.join(REPO, ".venv", "Scripts", "vllm.exe")
PROMPTS = os.path.join(HERE, "_prompts_real.jsonl")
SERVED = os.environ.get("AB_SERVED", "qwen3.8-27b")
# `--model` is what bench serve loads a *tokenizer* from; `--served-model-name`
# is what goes in the request body. Passing the served name as `--model` sends
# the client to HuggingFace hunting for a repo called `qwen3.8-27b`, which does
# not exist, and every call dies in retry backoff. Same checkpoint on both
# sides, so the tokenizer is identical either way.
MODEL_PATH = os.environ.get(
    "AB_MODEL_PATH", r"D:\models\Qwen3.8-27B-W4A16-AutoRound-fast")
SEED = int(os.environ.get("AB_SEED", "1000"))


def num(text, field):
    """`Mean TTFT (ms):  12.3` and `Drafts:  42` both parse."""
    m = re.search(rf"^{re.escape(field)}\s*(?:\([^)]*\))?:\s*([\d.eE+-]+)", text, re.M)
    return float(m.group(1)) if m else float("nan")


def fmt(v, nd=1):
    return "n/a" if v != v else f"{v:.{nd}f}"


def spec_counters(port):
    """(drafts, accepted) cumulative. WSL's tok/step source, same field names."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as fh:
            body = fh.read().decode("utf-8", "replace")
    except Exception:
        return None
    out = {}
    for name, suffix in (("drafts", "spec_decode_num_drafts_total"),
                         ("accepted", "spec_decode_num_accepted_tokens_total")):
        tot = 0.0
        found = False
        for line in body.splitlines():
            if line.startswith("#") or not line.startswith("vllm:"):
                continue
            head = line.split("{")[0].split()[0]
            if head.endswith(suffix):
                found = True
                try:
                    tot += float(line.split()[-1])
                except ValueError:
                    pass
        out[name] = tot if found else None
    return out


def run(port, label, args, outdir, tag):
    log_path = os.path.join(outdir, f"{tag}.log")
    cmd = [
        VLLM, "bench", "serve",
        "--host", "127.0.0.1", "--port", str(port),
        "--model", MODEL_PATH, "--served-model-name", SERVED,
        "--disable-tqdm",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
    ] + args
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("# " + " ".join(cmd) + "\n")
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    text = open(log_path, encoding="utf-8", errors="replace").read()
    print(f"    {tag}: {time.time() - t0:.0f}s", flush=True)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag-prefix", default="")
    ap.add_argument("--only", default="warmup,prefill,cohort",
                    help="subset of warmup,prefill,cohort")
    ap.add_argument("--lens", default="1024,4096,16384,32768",
                    help="prefill input lengths; keep TTFT under the server's "
                         "--sse-keep-alive-interval or bench serve drops the "
                         "response as 'never received a valid chunk'")
    args = ap.parse_args()
    want = set(args.only.split(","))
    lens = [int(x) for x in args.lens.split(",") if x]
    os.makedirs(args.outdir, exist_ok=True)
    seed = SEED
    rows = []

    def seed_inc():
        nonlocal seed
        seed += 1
        return seed

    print(f"\n########## {args.label}  (port {args.port}) ##########", flush=True)

    if "warmup" in want:
        # warmup -- first call after boot includes JIT; never report it.
        run(args.port, args.label,
            ["--dataset-name", "random", "--seed", str(seed_inc()),
             "--random-input-len", "256", "--random-output-len", "256",
             "--num-prompts", "16", "--max-concurrency", "8"],
            args.outdir, f"{args.tag_prefix}warmup")

    # prefill matrix, concurrency 1 -- the compute-bound row, no queueing in it
    if "prefill" in want:
        for ln in lens:
            n = {1024: 16, 4096: 8, 16384: 4, 32768: 2, 65536: 2}.get(ln, 2)
            text = run(args.port, args.label,
                       ["--dataset-name", "random", "--seed", str(seed_inc()),
                        "--random-input-len", str(ln), "--random-output-len", "1",
                        "--num-prompts", str(n), "--max-concurrency", "1"],
                       args.outdir, f"{args.tag_prefix}prefill_{ln}_c1")
            tot_in = num(text, "Total input tokens")
            dur = num(text, "Benchmark duration")
            ok = num(text, "Successful requests") == n
            rows.append({
                "kind": "prefill", "len": ln, "conc": 1, "prompts": n,
                "ok": ok,
                "tok_s": tot_in / dur if dur and dur == dur and dur > 0 else float("nan"),
                "mean_ttft_ms": num(text, "Mean TTFT"),
                "p99_ttft_ms": num(text, "P99 TTFT"),
                "dur_s": dur,
            })
            r = rows[-1]
            print(f"  prefill len={ln} c=1 | {fmt(r['tok_s'], 0)} tok/s | "
                  f"TTFT {fmt(r['mean_ttft_ms'])} ms"
                  f"{'' if ok else '  <-- FAILED'}", flush=True)

    # real-prompt cohorts, greedy -- decode-side
    if "cohort" in want:
        for c in (1, 2, 4, 8):
            before = spec_counters(args.port)
            text = run(args.port, args.label,
                       ["--dataset-name", "custom", "--dataset-path", PROMPTS,
                        "--custom-output-len", "1024", "--num-prompts", "8",
                        "--max-concurrency", str(c), "--temperature", "0"],
                       args.outdir, f"{args.tag_prefix}cohort_c{c}")
            after = spec_counters(args.port)
            tokstep = float("nan")
            if before and after and before["drafts"] is not None and after["drafts"] is not None:
                dd = after["drafts"] - before["drafts"]
                da = after["accepted"] - before["accepted"]
                if dd > 0:
                    tokstep = 1 + da / dd
            if tokstep != tokstep:
                tokstep = num(text, "Acceptance length")
            med_tpot = num(text, "Median TPOT")
            dec = c * 1000 / med_tpot if med_tpot == med_tpot and med_tpot > 0 else float("nan")
            rows.append({
                "kind": "cohort", "conc": c,
                "ok": num(text, "Successful requests") == 8,
                "e2e_tok_s": num(text, "Output token throughput"),
                "decode_tok_s": dec,
                "med_tpot_ms": med_tpot,
                "mean_ttft_ms": num(text, "Mean TTFT"),
                "p99_ttft_ms": num(text, "P99 TTFT"),
                "med_itl_ms": num(text, "Median ITL"),
                "p99_itl_ms": num(text, "P99 ITL"),
                "tok_step": tokstep,
                "accept_len": num(text, "Acceptance length"),
                "gen_tokens": num(text, "Total generated tokens"),
                "dur_s": num(text, "Benchmark duration"),
            })
            r = rows[-1]
            print(f"  cohort c={c} | e2e {fmt(r['e2e_tok_s'])} | decode {fmt(r['decode_tok_s'])} | "
                  f"tok/step {fmt(r['tok_step'], 2)} | TTFT {fmt(r['mean_ttft_ms'])} ms", flush=True)

    out_path = os.path.join(args.outdir, f"{args.label}.json")
    prev = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            prev = json.load(fh)
    merged = {r["kind"] + ("_%d" % r.get("len", r.get("conc", 0))): r for r in prev.get("rows", [])}
    for r in rows:
        merged[r["kind"] + ("_%d" % r.get("len", r.get("conc", 0)))] = r
    out = {"label": args.label, "port": args.port, "rows": list(merged.values()),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
