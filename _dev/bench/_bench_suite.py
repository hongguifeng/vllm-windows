#!/usr/bin/env python
"""Baseline benchmark suite for the Windows/CMP-170HX stack.

Ports the protocol from the WSL project's ``bench/run_benchmarks.sh`` so the two
machines produce comparable numbers:

* ``vllm bench serve`` for every measurement.
* A distinct ``--seed`` per call.  With prefix caching on, reusing a seed makes
  later calls silent partial prefix-cache hits, which reads prefill 15-20% low
  on one config and 3x high on another.
* Spec-decode health comes from Prometheus, not from the bench log: the ratio
  ``1 + d(accepted)/d(drafted)`` is the real tokens-per-step.

Usage::

    python _bench_suite.py            # full suite (~10 min)
    python _bench_suite.py --only warmup,cohort
    python _bench_suite.py --list
"""

from __future__ import annotations

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
VLLM = os.path.join(REPO, ".venv", "Scripts", "vllm.exe")
HOST = "127.0.0.1"
PORT = 8000
MODEL = r"D:\models\Qwen3.8-27B-W4A16-AutoRound-fast"
# The name the client puts in the request body.  If it does not match one of the
# server's --served-model-name values vLLM answers **404 NotFoundError**, and the
# access-log line then reads exactly like a missing route -- so a mismatch looks
# like "the endpoint disappeared".  Measured 2026-09-21 20:24: this constant had
# drifted to the full model path while start_server.ps1 passes `--served-model-name
# qwen3.8-27b` -> 8/8 requests rejected, every cohort row a zero, and no CUDA
# error to be found anywhere.  main() overrides it with whatever /v1/models
# reports, which is authoritative.
SERVED = "qwen3.8-27b"
PROMPTS = os.path.join(HERE, "_prompts_real.jsonl")
# VBENCH_OUT lets a run against a different GPU keep its own log set instead of
# overwriting the baseline it is being compared against.
OUT = os.environ.get("VBENCH_OUT", os.path.join(DEV, "out", "_bench_base"))
SUMMARY = os.path.join(OUT, "SUMMARY.md")

SEED_BASE = 2000
_seed = {"n": SEED_BASE}
# Set when any bench call comes back with 0 successful requests; main() turns it
# into a non-zero exit so a totally rejected run is never mistaken for "no data".
_had_failures = False


def detect_served() -> str:
    """Ask the server which model name it wants in the request body."""
    with urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=10) as r:
        ids = [m["id"] for m in json.load(r).get("data", []) if m.get("id")]
    if not ids:
        raise RuntimeError("/v1/models reported no model ids")
    return ids[0]


def next_seed() -> int:
    _seed["n"] += 1
    return _seed["n"]


def metric(name: str) -> float:
    """Read one Prometheus counter/gauge, 0.0 if absent."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/metrics", timeout=20) as r:
            for raw in r:
                line = raw.decode()
                if line.startswith("#") or "{" not in line:
                    continue
                if line.split("{")[0].split()[0] == name:
                    return float(line.split()[1])
    except Exception:
        pass
    return 0.0


def spec_pair() -> tuple[float, float]:
    return (
        metric("vllm:spec_decode_num_drafts_total"),
        metric("vllm:spec_decode_num_accepted_tokens_total"),
    )


def last_float(line: str) -> float | None:
    hits = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    return float(hits[-1]) if hits else None


def num(text: str, label: str) -> float:
    for line in text.splitlines():
        if label in line:
            v = last_float(line)
            if v is not None:
                return v
    return float("nan")


def fmt(v: float, nd: int = 1) -> str:
    return "n/a" if v != v else f"{v:.{nd}f}"


def bench(label: str, args: list[str]) -> str:
    """One `vllm bench serve` invocation -> log file, return its text."""
    global _had_failures
    log_path = os.path.join(OUT, f"{label}.log")
    cmd = [
        VLLM, "bench", "serve",
        "--host", HOST, "--port", str(PORT),
        "--model", MODEL, "--served-model-name", SERVED,
        "--disable-tqdm",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
    ] + args
    print(f"\n>>> {label}\n    {' '.join(args)}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("# " + " ".join(cmd) + "\n")
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    text = open(log_path, encoding="utf-8", errors="replace").read()
    print(f"    done in {time.time() - t0:.1f}s -> {os.path.basename(log_path)}", flush=True)
    if num(text, "Successful requests") == 0:
        # Never let a rejected run flow into a 0.0 table unnoticed.
        errs = re.findall(r"^Error \d+: (.*)$", text, re.M)
        _had_failures = True
        print(f"    !! {label}: 0 successful requests -- the server rejected every "
              f"request ({errs[0] if errs else 'see ' + log_path})", flush=True)
    return text


def row(label: str, text: str, conc: int, d0: float, a0: float,
        from_log: bool = False) -> str:
    if from_log:
        # Re-parsing a saved log: the Prometheus deltas are gone, so trust the
        # bench's own acceptance-length field.
        tokstep = num(text, "Acceptance length")
    else:
        d1, a1 = spec_pair()
        dd, da = d1 - d0, a1 - a0
        # Prometheus deltas are the live truth; the log's own acceptance length
        # is the fallback.
        tokstep = 1 + da / dd if dd > 0 else num(text, "Acceptance length")
    e2e = num(text, "Output token throughput")
    med_tpot = num(text, "Median TPOT")
    med_itl = num(text, "Median ITL")
    mean_ttft = num(text, "Mean TTFT")
    p99_ttft = num(text, "P99 TTFT")
    p99_itl = num(text, "P99 ITL")
    dur = num(text, "Benchmark duration")
    dec = conc * 1000 / med_tpot if med_tpot and med_tpot == med_tpot and med_tpot > 0 else float("nan")
    return (
        f"| {label} | {conc} | {fmt(e2e)} | {fmt(dec)} | {fmt(med_tpot, 2)} | "
        f"{fmt(mean_ttft, 1)} | {fmt(p99_ttft, 1)} | {fmt(tokstep, 3)} | "
        f"{fmt(med_itl, 2)} | {fmt(p99_itl, 2)} | {fmt(dur, 1)} |"
    )


COHORT_HEADER = (
    "| 项目 | C | e2e tok/s | decode tok/s | medTPOT ms | meanTTFT ms | p99TTFT ms | tok/step | medITL ms | p99ITL ms | dur s |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)

# Concurrency above the server's --max-num-seqs is split into batches, so TTFT
# measures queueing rather than parallel serving, and `decode` (C x 1000/TPOT)
# overstates the batch that actually ran.
# 2026-09-21: this note used to hardcode "本次服务为 --max-num-seqs 4, 故 C=8 实为
# 4+4" -- wrong for every run whose serve script passes --max-num-seqs 8, where
# C=8 really is one batch.  The client cannot read the flag over HTTP, so state
# the relation and point at the source of truth instead of guessing.
COHORT_NOTE = [
    "",
    "> ⚠️ C 超过服务端 `--max-num-seqs` 的档位会被拆成多批：`meanTTFT` 反映**排队时间**"
    "而非并发延迟，`decode` 列（C × 1000 / medTPOT）会高估实际在跑的 batch。",
    "> 本次服务实际的 `--max-num-seqs` 见 serve 日志头部的 `--- argv ---`（`_repro_async_tower.ps1` "
    "与 `start_server.ps1` 默认传 8，故 C=8 是一批；若该档与更低档总时长相同则说明被拆批）。",
]


def do_warmup() -> list[str]:
    text = bench(
        "warmup",
        ["--dataset-name", "random", "--seed", str(next_seed()),
         "--random-input-len", "256", "--random-output-len", "256",
         "--num-prompts", "16", "--max-concurrency", "8"],
    )
    return ["## warmup", "", COHORT_HEADER,
            row("random 256/256", text, 8, 0.0, 0.0), ""]


def do_cohort() -> list[str]:
    out = ["## 真实 prompt 并发阶梯（prompts_real.jsonl, out=1024, T=0）", "", COHORT_HEADER]
    for c in (1, 2, 4, 8):
        d0, a0 = spec_pair()
        text = bench(
            f"cohort_c{c}",
            ["--dataset-name", "custom", "--dataset-path", PROMPTS,
             "--custom-output-len", "1024", "--num-prompts", "8",
             "--max-concurrency", str(c), "--temperature", "0"],
        )
        out.append(row("real prompts", text, c, d0, a0))
    # Concurrency above the server's --max-num-seqs is split into batches, so
    # TTFT measures queueing, not parallel serving, and `decode` (C x 1000/TPOT)
    # overstates the running batch.  Flag it rather than let it be misread.
    out += COHORT_NOTE
    out.append("")
    return out


# Default prefill ladder for the full suite.  `--lens 1024,16384` replaces it,
# which is what the 09-20 A/B protocol used (`_ab_bench.py --lens 1024,16384`),
# and that protocol ran the prefill stage BEFORE the cohort -- a long-context
# warm state the 2026-09-21 repro runs did not reproduce (they ran `--only
# cohort`), which is the leading suspect for why they came back clean.
PREFILL_MATRIX = ((4096, 1, 8), (16384, 1, 4), (65536, 1, 2), (102400, 1, 1))
PREFILL_COUNTS = {1024: 16, 4096: 8, 16384: 4, 32768: 2, 65536: 2}
LENS: list[int] = []


def prefill_matrix() -> tuple:
    if not LENS:
        return PREFILL_MATRIX
    return tuple((ln, 1, PREFILL_COUNTS.get(ln, 2)) for ln in LENS)


def do_prefill() -> list[str]:
    out = ["## prefill 阶梯（random dataset, output=1）", "",
           "| 输入长度 | C | n | prefill tok/s | meanTTFT ms | p99TTFT ms | dur s |",
           "|---|---|---|---|---|---|---|"]
    for length, conc, n in prefill_matrix():
        text = bench(
            f"prefill_{length}",
            ["--dataset-name", "random", "--seed", str(next_seed()),
             "--random-input-len", str(length), "--random-output-len", "1",
             "--num-prompts", str(n), "--max-concurrency", str(conc)],
        )
        tin = num(text, "Total input tokens")
        dur = num(text, "Benchmark duration")
        rate = tin / dur if dur and dur == dur and dur > 0 else float("nan")
        out.append(
            f"| {length} | {conc} | {n} | {fmt(rate, 0)} | "
            f"{fmt(num(text, 'Mean TTFT'), 1)} | {fmt(num(text, 'P99 TTFT'), 1)} | "
            f"{fmt(dur, 1)} |"
        )
    out.append("")
    return out


def do_long() -> list[str]:
    out = ["## 长上下文", "",
           "| 项目 | meanTTFT ms | p99TTFT ms | medTPOT ms | medITL ms | dur s |",
           "|---|---|---|---|---|---|"]
    d0, a0 = spec_pair()
    text = bench(
        "long_100k_256",
        ["--dataset-name", "random", "--seed", str(next_seed()),
         "--random-input-len", "100000", "--random-output-len", "256",
         "--num-prompts", "1", "--max-concurrency", "1", "--ignore-eos"],
    )
    d1, a1 = spec_pair()
    dd, da = d1 - d0, a1 - a0
    tokstep = 1 + da / dd if dd > 0 else float("nan")
    out.append(
        f"| 1x100k/256 | {fmt(num(text, 'Mean TTFT'), 1)} | {fmt(num(text, 'P99 TTFT'), 1)} | "
        f"{fmt(num(text, 'Median TPOT'), 2)} | {fmt(num(text, 'Median ITL'), 2)} | "
        f"{fmt(num(text, 'Benchmark duration'), 1)} |"
    )
    out.append(f"\ntok/step = {fmt(tokstep, 3)}；prefill = "
               f"{fmt(num(text, 'Total input tokens') / max(num(text, 'Benchmark duration'), 1e-9), 0)} tok/s")
    out.append("")
    return out


def reparse() -> int:
    """Rebuild every table from the saved logs. No GPU work, no server needed."""

    def read(name: str) -> str:
        path = os.path.join(OUT, f"{name}.log")
        if not os.path.exists(path):
            return ""
        return open(path, encoding="utf-8", errors="replace").read()

    def conc_of(text: str) -> int:
        m = re.search(r"--max-concurrency (\d+)", text)
        return int(m.group(1)) if m else 1

    blocks: list[str] = []

    text = read("warmup")
    if text:
        blocks += ["## warmup（256 in / 256 out, 16 prompts）", "", COHORT_HEADER,
                   row("random 256/256", text, conc_of(text), 0.0, 0.0, from_log=True), ""]

    cohort = [row("real prompts", read(f"cohort_c{c}"), c, 0.0, 0.0, from_log=True)
              for c in (1, 2, 4, 8) if read(f"cohort_c{c}")]
    if cohort:
        blocks += ["## 真实 prompt 并发阶梯（prompts_real.jsonl, out=1024, T=0）", "",
                   COHORT_HEADER] + cohort + COHORT_NOTE + [""]

    pf = []
    for length, conc, n in ((4096, 1, 8), (16384, 1, 4), (65536, 1, 2), (102400, 1, 1)):
        t = read(f"prefill_{length}")
        if not t:
            continue
        tin, dur = num(t, "Total input tokens"), num(t, "Benchmark duration")
        rate = tin / dur if dur and dur == dur and dur > 0 else float("nan")
        pf.append(f"| {length} | {conc} | {n} | {fmt(rate, 0)} | "
                  f"{fmt(num(t, 'Mean TTFT'), 1)} | {fmt(num(t, 'P99 TTFT'), 1)} | "
                  f"{fmt(dur, 1)} |")
    if pf:
        blocks += ["## prefill 阶梯（random dataset, output=1）", "",
                   "| 输入长度 | C | n | prefill tok/s | meanTTFT ms | p99TTFT ms | dur s |",
                   "|---|---|---|---|---|---|---|"] + pf + [""]

    t = read("long_100k_256")
    if t:
        tin, dur = num(t, "Total input tokens"), num(t, "Benchmark duration")
        blocks += ["## 长上下文", "",
                   "| 项目 | meanTTFT ms | p99TTFT ms | medTPOT ms | medITL ms | p99ITL ms | dur s |",
                   "|---|---|---|---|---|---|---|",
                   f"| 1x100k/256 | {fmt(num(t, 'Mean TTFT'), 1)} | {fmt(num(t, 'P99 TTFT'), 1)} | "
                   f"{fmt(num(t, 'Median TPOT'), 2)} | {fmt(num(t, 'Median ITL'), 2)} | "
                   f"{fmt(num(t, 'P99 ITL'), 2)} | {fmt(dur, 1)} |", "",
                   f"tok/step = {fmt(num(t, 'Acceptance length'), 3)}，"
                   f"prefill = {fmt(tin / dur if dur and dur > 0 else float('nan'), 0)} tok/s", ""]

    old = open(SUMMARY, encoding="utf-8").read() if os.path.exists(SUMMARY) else ""
    head = old.split("\n## ", 1)[0]
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(head + "\n" + "\n".join(blocks) +
                 "\n---\n由 `_bench_suite.py --reparse` 从 `_bench_base/*.log` 重建。\n")
    print(f"re-parsed -> {SUMMARY}")
    return 0


STAGES = {
    "warmup": do_warmup,
    "cohort": do_cohort,
    "prefill": do_prefill,
    "long": do_long,
}
ORDER = ["warmup", "cohort", "prefill", "long"]


def config_snapshot() -> list[str]:
    cfg = {
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip(),
        "max_model_len": json.load(urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models"))
        ["data"][0]["max_model_len"],
        "spec_decode_drafts_total": metric("vllm:spec_decode_num_drafts_total"),
        "spec_decode_accepted_total": metric("vllm:spec_decode_num_accepted_tokens_total"),
    }
    cfg_blob = metric("vllm:cache_config_info")  # touch, real values below
    lines = ["## 机器与服务配置快照", "",
             "| 项 | 值 |", "|---|---|",
             f"| GPU | {cfg['gpu']} |",
             f"| max_model_len | {cfg['max_model_len']} |",
             f"| spec drafts/accepted | {cfg['spec_decode_drafts_total']:.0f} / {cfg['spec_decode_accepted_total']:.0f} |",
             ""]
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/metrics", timeout=20) as r:
            for raw in r:
                line = raw.decode()
                if line.startswith("vllm:cache_config_info"):
                    body = line[line.index("{") + 1:line.rindex("}")]
                    for kv in body.split(","):
                        k, _, v = kv.partition("=")
                        if k.strip() in ("block_size", "num_gpu_blocks", "gpu_memory_utilization"):
                            lines.append(f"| cache_config_info.{k.strip()} | {v.strip()} |")
                    break
    except Exception:
        pass
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list: warmup,cohort,prefill,long")
    ap.add_argument("--reparse", action="store_true",
                    help="rebuild SUMMARY.md from saved logs instead of running")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--served", default="",
                    help="override the model name sent in requests "
                         "(default: whatever /v1/models reports)")
    ap.add_argument("--lens", default="",
                    help="comma list of prefill input lengths, e.g. 1024,16384 "
                         "(default: the full ladder 4096/16384/65536/102400)")
    a = ap.parse_args()
    if a.list:
        print(" ".join(ORDER))
        return 0
    if a.reparse:
        return reparse()

    stages = [s.strip() for s in a.only.split(",") if s.strip()] or ORDER
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"unknown stage(s): {bad}", file=sys.stderr)
        return 2

    os.makedirs(OUT, exist_ok=True)
    try:
        urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=10)
    except Exception as e:
        print(f"no server on {HOST}:{PORT}: {e}", file=sys.stderr)
        return 1

    global SERVED, LENS
    LENS = [int(x) for x in a.lens.split(",") if x.strip()]
    if LENS:
        print(f"prefill lens override: {LENS} -> {prefill_matrix()}", flush=True)
    try:
        SERVED = a.served or detect_served()
    except Exception as e:
        print(f"cannot read {HOST}:{PORT}/v1/models: {e}", file=sys.stderr)
        return 1
    print(f"served model name: {SERVED}  (from /v1/models)", flush=True)

    t0 = time.time()
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines()[0]
    head = [f"# {gpu_name} 基线基准（{time.strftime('%Y-%m-%d %H:%M:%S')}）", "",
            f"协议复刻自 WSL 项目 `bench/run_benchmarks.sh`；seed 起点 {SEED_BASE}，每次调用递增。", ""]
    head += config_snapshot()
    blocks: list[str] = []
    # Write progressively so a killed run still leaves usable results.
    for name in stages:
        print(f"\n===== stage: {name} =====", flush=True)
        blocks += STAGES[name]()
        with open(SUMMARY, "w", encoding="utf-8") as fh:
            fh.write("\n".join(head + blocks) + "\n")

    total = time.time() - t0
    with open(SUMMARY, "a", encoding="utf-8") as fh:
        fh.write(f"\n---\n总耗时 {total / 60:.1f} min，原始日志见 "
                 f"`{os.path.basename(OUT)}/*.log`。\n")
    if _had_failures:
        print(f"\n!! SOME CALLS WERE REJECTED -- {SUMMARY} is not a usable baseline "
              f"(see the '!!' lines above)", file=sys.stderr, flush=True)
        return 1
    print(f"\n===== ALL DONE in {total / 60:.1f} min -> {SUMMARY} =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
