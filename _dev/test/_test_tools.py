# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Probe how the running server handles tool calls and reasoning.

Sends a request that should trigger a tool call, then reports exactly where the
model's output landed: the structured `tool_calls` field, or raw XML text in
`content`.  Also shows whether thinking was split into `reasoning_content`.

Runs two variants because they fail differently on a misconfigured server:
  A) tool_choice="auto"     -- rejected with HTTP 400 unless BOTH
                               --enable-auto-tool-choice and
                               --tool-call-parser are set
  B) no tool_choice         -- defaults to "auto" when `tools` are present, so
                               it hits the same 400 on a server without them

    python _test_tools.py [endpoint] [served-model-name]
"""

import json
import os
import sys
import urllib.error
import urllib.request

EP = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1/chat/completions"
MODEL = (
    sys.argv[2]
    if len(sys.argv) > 2
    else os.environ.get("VLLM_TEST_MODEL", "Qwen3.8-27B-W4A16-AutoRound-fast")
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return its stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

BASE = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "List the files in the current directory."}
    ],
    "tools": TOOLS,
    "max_tokens": 300,
    "temperature": 0.0,
}


def call(payload, label):
    print("=" * 72)
    print(f"[{label}]  tool_choice={payload.get('tool_choice', '(absent)')}")
    print("=" * 72)
    req = urllib.request.Request(
        EP,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        print(f"HTTP {exc.code}  {body[:500]}")
        print()
        return

    choice = out["choices"][0]
    msg = choice["message"]
    print("finish_reason      :", choice.get("finish_reason"))
    print("tool_calls present :", bool(msg.get("tool_calls")))
    if msg.get("tool_calls"):
        print("tool_calls         :", json.dumps(msg["tool_calls"], ensure_ascii=False))
    print("message keys       :", sorted(msg.keys()))
    print("reasoning_content  :", repr(msg.get("reasoning_content"))[:160])
    print("-" * 72)
    print("content            :")
    print(msg.get("content"))
    print("=" * 72)
    content = msg.get("content") or ""
    if msg.get("tool_calls"):
        print("VERDICT: parsed into the structured `tool_calls` field.  OK")
    elif "<tool_call>" in content:
        print("VERDICT: tool call leaked into `content` as raw XML -- no tool parser.")
    else:
        print("VERDICT: no tool call emitted.")
    print()


call({**BASE, "tool_choice": "auto"}, "A")
call(dict(BASE), "B")
