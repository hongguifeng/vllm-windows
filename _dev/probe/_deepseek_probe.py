"""Probe the DeepSeek API for the fields mini-swe-agent needs.

Checks: plain completion, tool calling, reasoning_content, and whether the
OpenAI-compatible knobs (temperature=0, max_tokens, reasoning_effort,
extra_body) are accepted.

    python _dev/probe/_deepseek_probe.py
"""

import json
import os
import urllib.error
import urllib.request

BASE = "https://api.deepseek.com"
KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = "deepseek-flash"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the container.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def call(label, body):
    request = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read())
        choice = data["choices"][0]
        message = choice["message"]
        print(f"\n== {label}: ok finish={choice.get('finish_reason')}")
        print(f"   message keys: {sorted(message)}")
        print(f"   content: {str(message.get('content'))[:100]!r}")
        reasoning = message.get("reasoning_content")
        print(f"   reasoning_content: {str(reasoning)[:80]!r}")
        calls = message.get("tool_calls") or []
        print(f"   tool_calls: {len(calls)}")
        for entry in calls[:2]:
            print(f"     {entry['function']['name']} {entry['function']['arguments'][:120]}")
        print(f"   usage: {data.get('usage')}")
    except urllib.error.HTTPError as error:
        print(f"\n== {label}: HTTP {error.code}")
        print(error.read().decode()[:500])
    except Exception as error:  # noqa: BLE001
        print(f"\n== {label}: {type(error).__name__}: {error}")


def main():
    if not KEY:
        raise SystemExit("set DEEPSEEK_API_KEY")
    call(
        "plain",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 17*23? Answer with the number."}],
            "temperature": 0,
            "max_tokens": 64,
        },
    )
    call(
        "tools",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Run `pwd` in the shell."}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
        },
    )
    call(
        "tools+reasoning_effort",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Run `ls /tmp` in the shell."}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
            "reasoning_effort": "medium",
        },
    )
    call(
        "tools+extra_body",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Run `ls /tmp` in the shell."}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        },
    )


if __name__ == "__main__":
    main()
