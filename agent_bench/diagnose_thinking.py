#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Decide whether an arm's failure is capability or protocol.

Run this before accepting any low score as a capability result. It replays the
real message state from a finished trajectory against the endpoint and varies one
thing at a time.

Reads:
  * Does thinking length track the token cap exactly at every cap? If yes, the
    model never terminates thinking in that state, and raising max_tokens cannot
    raise the success rate -- it only buys more of the same non-productive text.
  * Does dropping echoed reasoning turn a failed step into a working tool call?
    If yes, history thinking, not the model, is what broke the step.
  * Does a second endpoint terminate the same state naturally? If yes, the
    problem is the first endpoint's template/decoding, not the task.

A step whose finish_reason is `length` with reasoning == completion == cap, or
whose parsed tool arguments come back as an empty object, is a protocol failure:
mini-swe-agent records it as a format error and, after two in a row, kills the
episode. That shows up as a low resolution rate with no capability signal.
"""

import argparse
import json
import urllib.request


def normalize(api_base):
    """Accept a base with or without the /v1 suffix and return one that has it."""
    base = api_base.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def post(api_base, path, body, timeout=900):
    url = normalize(api_base) + path
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def state_from_traj(traj_path):
    with open(traj_path) as f:
        t = json.load(f)
    msgs = [
        m
        for m in t.get("messages", [])
        if m.get("role") in ("system", "user", "assistant", "tool")
        and not (m.get("extra") or {}).get("interrupt_type")
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a bash command in the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "context": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        }
    ]
    cfg = ((t.get("info") or {}).get("config") or {}).get("model") or {}
    return msgs, tools, cfg.get("model_kwargs") or {}


def chat(api_base, model, msgs, tools, cap, effort, temperature):
    body = {
        "model": model,
        "messages": msgs,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": cap,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort},
    }
    return post(api_base, "/chat/completions", body)


def report(label, r):
    ch = (r.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    u = r.get("usage") or {}
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    ct = u.get("completion_tokens") or 0
    args = [
        (tc.get("function") or {}).get("arguments")
        for tc in (m.get("tool_calls") or [])
    ]
    print(
        f"{label:<28} prompt={u.get('prompt_tokens') or 0:6d} "
        f"finish={ch.get('finish_reason'):9} reasoning={rt:6d} completion={ct:6d} "
        f"content={len(m.get('content') or ''):5d} "
        f"args={[repr(x)[:60] for x in args]}"
    )
    return rt, ct, bool(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="trajectory of a failing episode")
    p.add_argument("--api-base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--caps", default="3072,6144,20000", help="comma separated max_tokens ladder"
    )
    p.add_argument("--effort", default="medium")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--second-api-base", default="", help="cross-check another endpoint")
    p.add_argument("--second-model", default="")
    p.add_argument("--no-strip", action="store_true", help="skip the strip A/B")
    a = p.parse_args()

    msgs, tools, kwargs = state_from_traj(a.traj)
    n_rc = sum(
        1
        for m in msgs
        if m.get("role") == "assistant" and (m.get("reasoning_content") or "")
    )
    print(f"messages={len(msgs)} assistant turns carrying reasoning_content={n_rc}")

    caps = [int(x) for x in a.caps.split(",") if x.strip()]
    fills = 0
    for cap in caps:
        r = chat(a.api_base, a.model, msgs, tools, cap, a.effort, a.temperature)
        rt, ct, has_args = report(f"cap={cap} effort={a.effort}", r)
        if rt == cap and ct == cap:
            fills += 1
    print(
        f"caps where reasoning and completion both equal the cap: {fills}/{len(caps)}"
    )
    if fills == len(caps):
        print("=> thinking never terminates in this state; max_tokens is not the fix.")

    if not a.no_strip and n_rc:
        stripped = [
            {k: v for k, v in m.items() if k != "reasoning_content"}
            if m.get("role") == "assistant"
            else m
            for m in msgs
        ]
        cap = caps[0]
        r = chat(a.api_base, a.model, stripped, tools, cap, a.effort, a.temperature)
        report(f"strip={not a.no_strip} cap={cap}", r)

    if a.second_api_base:
        cap = caps[-1]
        r = chat(
            a.second_api_base,
            a.second_model or a.model,
            msgs,
            tools,
            cap,
            a.effort,
            a.temperature,
        )
        report(f"second endpoint cap={cap}", r)


if __name__ == "__main__":
    main()
