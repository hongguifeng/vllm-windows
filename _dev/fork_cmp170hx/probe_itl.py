# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inter-token latency probe for fair chunked prefill.

Streams one long-output request (the "victim") and, while it is decoding,
submits a long-prompt prefill (the "disturber"). Reports the victim's
inter-token latency distribution, which is what fair chunked prefill is meant
to protect.

Usage:
    python _dev/fork_cmp170hx/probe_itl.py --base http://127.0.0.1:9394/v1
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

LONG_OUTPUT_PROMPT = "List every integer from 1 to 200, one per line."
DISTURBER_WORDS = 4000


def _payload(prompt: str, max_tokens: int) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Force the stream length so the decode phase reliably overlaps the
        # disturber's prefill instead of ending early on EOS.
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _open(base: str, payload: dict):
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=600)


def disturb(base: str, delay: float, trial: int) -> tuple[float, float]:
    """Wait `delay` seconds, then start a long-prefill request.

    The prompt is unique per trial so prefix caching cannot make a repeat arm
    cheap; only the first trial of a run would otherwise do real prefill work.
    """
    time.sleep(delay)
    prompt = "Repeat the following list verbatim, nothing else:\n\n" + " ".join(
        f"t{trial}w{i}" for i in range(DISTURBER_WORDS)
    )
    t0 = time.perf_counter()
    first = None
    prompt_tokens = 0
    with _open(base, _payload(prompt, 8)) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            if delta.get("content") and first is None:
                first = time.perf_counter() - t0
            usage = chunk.get("usage") or {}
            if usage.get("prompt_tokens"):
                prompt_tokens = usage["prompt_tokens"]
    return t0, first if first is not None else time.perf_counter() - t0, prompt_tokens


def victim(base: str, max_tokens: int) -> list[float]:
    """Return the inter-token gaps (ms) observed while streaming."""
    stamps: list[float] = []
    with _open(base, _payload(LONG_OUTPUT_PROMPT, max_tokens)) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            if delta.get("content"):
                stamps.append(time.perf_counter())
    return stamps


def summarize(stamps: list[float]) -> dict:
    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    if not gaps:
        return {}
    gaps_sorted = sorted(gaps)
    pct = lambda q: gaps_sorted[min(len(gaps_sorted) - 1, int(q * len(gaps_sorted)))]
    return {
        "tokens": len(stamps),
        "gaps": len(gaps),
        "mean_ms": statistics.fmean(gaps),
        "median_ms": statistics.median(gaps),
        "p90_ms": pct(0.90),
        "p99_ms": pct(0.99),
        "max_ms": max(gaps),
        "over_100ms": sum(1 for g in gaps if g > 100),
        "over_300ms": sum(1 for g in gaps if g > 300),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default="http://127.0.0.1:9394/v1", help="OpenAI-compatible root"
    )
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument(
        "--disturb-delay",
        type=float,
        default=1.5,
        help="seconds after the victim starts before the disturber is submitted",
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="number of trials to average"
    )
    args = parser.parse_args()
    base = args.base.rstrip("/")

    runs = []
    for trial in range(args.repeat):
        box: list[tuple[float, float, int]] = []

        def runner(
            box: list[tuple[float, float, int]] = box,
            trial: int = trial,
        ) -> None:
            box.append(disturb(base, args.disturb_delay, trial))

        thread = threading.Thread(target=runner)
        thread.start()
        stamps = victim(base, args.max_tokens)
        thread.join()
        stats = summarize(stamps)
        runs.append(stats)
        started, ttft, ptok = box[0] if box else (0.0, 0.0, 0)
        print(
            f"[trial {trial}] victim tokens={stats.get('tokens')} "
            f"mean={stats.get('mean_ms', 0):.1f}ms "
            f"median={stats.get('median_ms', 0):.1f}ms "
            f"p90={stats.get('p90_ms', 0):.1f}ms p99={stats.get('p99_ms', 0):.1f}ms "
            f"max={stats.get('max_ms', 0):.1f}ms >100ms={stats.get('over_100ms')} "
            f">300ms={stats.get('over_300ms')} | "
            f"disturber prompt={ptok} tok ttft={ttft:.3f}s"
        )

    keys = [k for k in runs[0] if k != "tokens"]
    print("[mean over trials]")
    for key in keys:
        vals = [r[key] for r in runs]
        print(f"  {key}: {statistics.fmean(vals):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
