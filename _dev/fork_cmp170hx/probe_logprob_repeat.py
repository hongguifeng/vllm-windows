# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check run-to-run reproducibility of logprobs on a running service.

Sends the same short prompt N times with logprobs enabled and compares the
returned per-token logprob vectors. Any difference means the MoE / attention
numerics are not reproducible for this serving configuration, which is what
VLLM_MOE_MASK_PADDING and VLLM_DETERMINISTIC_MOE_ALIGN are meant to fix.

Usage:
    python probe_logprob_repeat.py --base http://127.0.0.1:9394/v1 [--reps 8]
"""

import argparse
import json
import urllib.request


def _logprobs_vector(resp: dict) -> list[tuple[str, float]]:
    """Return (token_text, logprob) pairs for the first choice.

    The completions endpoint returns the legacy ``token_logprobs``/``tokens``
    arrays; the chat endpoint returns ``content``. Both are accepted.
    """
    ch = resp["choices"][0]
    lp = ch.get("logprobs") or {}
    content = lp.get("content")
    if content:
        return [
            (str(item.get("token")), round(float(item.get("logprob", 0.0)), 6))
            for item in content
        ]
    tokens = lp.get("tokens") or []
    values = lp.get("token_logprobs") or []
    return [(str(tok), round(float(lp_i), 6)) for tok, lp_i in zip(tokens, values)]


def _ask(base: str, prompt: str, logprobs: int, max_tokens: int) -> dict:
    payload = {
        "model": "Qwen3.8-Flash-Next",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": logprobs,
        "stream": False,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.loads(fh.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:9393/v1")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument(
        "--prompt",
        default="The inventory list contains 900 items. Continue the sentence.",
    )
    args = ap.parse_args()

    runs = []
    for i in range(args.reps):
        resp = _ask(args.base, args.prompt, 1, args.max_tokens)
        runs.append(_logprobs_vector(resp))
        print(f"[run {i}] tokens={len(runs[-1])} text={resp['choices'][0]['text']!r}")

    ref = runs[0]
    diffs = 0
    worst = 0.0
    for i, run in enumerate(runs[1:], start=1):
        if len(run) != len(ref):
            print(f"[diff ] run {i} has {len(run)} tokens vs reference {len(ref)}")
            diffs += 1
            continue
        for (tok_a, lp_a), (tok_b, lp_b) in zip(ref, run):
            if tok_a != tok_b or abs(lp_a - lp_b) > 1e-6:
                diffs += 1
                worst = max(worst, abs(lp_a - lp_b))
                print(f"[diff ] run {i} pos {tok_a!r}: {lp_a} vs {lp_b}")

    total = (args.reps - 1) * len(ref)
    print(
        f"[result] {diffs} differing tokens out of {total} compared "
        f"(worst |delta| = {worst:.6f})"
    )


if __name__ == "__main__":
    main()
