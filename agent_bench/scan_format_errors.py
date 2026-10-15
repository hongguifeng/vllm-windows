#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scan format errors in one arm: what came back vs what the agent had been sending.

Distinguishes the two protocol failure shapes:
  * truncated thinking (finish_reason=length, reasoning == completion == cap)
  * tool calls whose arguments parse to an empty object

Both are recorded by mini-swe-agent as format errors, and two in a row end the
episode, so an arm can lose most of its episodes without any capability signal.
"""

import argparse
import glob
import json
import os
from collections import Counter


def scan(root, per_instance_limit):
    files = sorted(glob.glob(os.path.join(root, "*", "*.traj.json")))
    shapes = Counter()
    total_fmt = 0
    for f in files:
        with open(f) as fh:
            t = json.load(fh)
        msgs = t.get("messages", [])
        inst = f.split("/")[-2]
        n = 0
        for m in msgs:
            if (m.get("extra") or {}).get("interrupt_type") != "FormatError":
                continue
            total_fmt += 1
            n += 1
            resp = (m.get("extra") or {}).get("response")
            if isinstance(resp, str):
                resp = json.loads(resp)
            if not isinstance(resp, dict):
                shape = "no_response_recorded"
            else:
                ch = (resp.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                usage = resp.get("usage") or {}
                reasoning = (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ) or 0
                completion = usage.get("completion_tokens") or 0
                args = [
                    (tc.get("function") or {}).get("arguments")
                    for tc in (msg.get("tool_calls") or [])
                ]
                if ch.get("finish_reason") == "length":
                    shape = (
                        "thinking_filled_cap"
                        if reasoning == completion
                        else "truncated_other"
                    )
                elif args and all((a or "").strip() in ("{}", "") for a in args):
                    shape = "tool_call_args_empty"
                else:
                    shape = "no_tool_call_emitted"
            shapes[shape] += 1
            if n > per_instance_limit:
                continue
            print(f"\n{inst} format_error #{n}: {shape}")
            if isinstance(resp, dict):
                ch = (resp.get("choices") or [{}])[0]
                usage = resp.get("usage") or {}
                snippet = (m.get("content") or "").split("<error>")[-1][:140]
                print(f"   error: {snippet}")
                r_tok = (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ) or 0
                c_tok = usage.get("completion_tokens") or 0
                prior_args = [
                    repr((tc.get("function") or {}).get("arguments"))[:60]
                    for tc in ((ch.get("message") or {}).get("tool_calls") or [])
                ]
                print(
                    f"   finish={ch.get('finish_reason')} reasoning={r_tok} "
                    f"completion={c_tok} args={prior_args}"
                )
        prior = [
            ((tc.get("function") or {}).get("arguments"))
            for m in msgs
            if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        ]
        bad = [x for x in prior if x in ("{}", "", None)]
        if bad:
            print(f"{inst}: prior tool_calls={len(prior)} of which empty: {len(bad)}")
    print(f"\nformat errors total: {total_fmt}")
    for shape, n in shapes.most_common():
        print(f"   {shape:<24} {n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="arm output directory")
    p.add_argument(
        "--limit", type=int, default=2, help="format errors to print per instance"
    )
    a = p.parse_args()
    scan(a.out, a.limit)


if __name__ == "__main__":
    main()
