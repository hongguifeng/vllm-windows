"""Probe /v1/responses SSE tool-call streaming against a live server.

Prints every event type and reconstructs the tool call arguments, so a
truncated `response.function_call_arguments.done` is obvious.

Run from anywhere:  python _probe_responses_stream.py [port]
"""

import json
import sys

import requests

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
URL = f"http://127.0.0.1:{PORT}/v1/responses"

TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a text file from disk and return its content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "offset": {"type": "integer", "description": "Start line"},
            },
            "required": ["path"],
        },
    }
]

BODY = {
    "model": "qwen3.8-27b",
    "input": [
        {
            "role": "system",
            "content": "You are a coding agent. Use the provided tools when asked.",
        },
        {
            "role": "user",
            "content": (
                "Read the file D:/code/vllm-windows/AGENTS.md starting at line 12. "
                "Use the read_file tool. Do not explain, just call the tool."
            ),
        },
    ],
    "tools": TOOLS,
    "tool_choice": "auto",
    "stream": True,
    "max_output_tokens": 200,
}

EVENT_OF_INTEREST = (
    "response.output_item.added",
    "response.output_item.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.completed",
    "response.failed",
)


def main() -> int:
    args = []
    acc = {}
    types = {}
    with requests.post(URL, json=BODY, stream=True, timeout=180) as r:
        print(f"HTTP {r.status_code}  content-type={r.headers.get('content-type')}")
        if r.status_code != 200:
            print(r.text[:2000])
            return 1
        for raw in r.iter_lines(decode_unicode=False):
            if not raw or not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            et = ev.get("type", "?")
            types[et] = types.get(et, 0) + 1
            if et == "response.function_call_arguments.delta":
                idx = ev.get("output_index")
                acc[idx] = acc.get(idx, "") + ev.get("delta", "")
            if et in EVENT_OF_INTEREST:
                args.append((et, ev))

    print("\n=== event counts ===")
    for k, v in sorted(types.items()):
        print(f"  {v:3d}  {k}")

    print("\n=== tool-call related events ===")
    for et, ev in args:
        if et == "response.function_call_arguments.delta":
            print(f"  {et}  out={ev.get('output_index')}  delta={ev.get('delta')!r}")
        elif et == "response.function_call_arguments.done":
            print(f"  {et}  out={ev.get('output_index')}  name={ev.get('name')!r}")
            print(f"      arguments = {ev.get('arguments')!r}")
            try:
                json.loads(ev.get("arguments") or "")
                print("      >>> arguments parse OK")
            except Exception as exc:
                print(f"      >>> arguments PARSE FAILED: {exc}")
        elif et == "response.output_item.added":
            print(f"  {et}  out={ev.get('output_index')}  {json.dumps(ev.get('item'), ensure_ascii=False)[:200]}")
        elif et == "response.output_item.done":
            print(f"  {et}  out={ev.get('output_index')}  {json.dumps(ev.get('item'), ensure_ascii=False)[:200]}")
        else:
            print(f"  {et}  {json.dumps(ev, ensure_ascii=False)[:300]}")

    print("\n=== reconstructed from deltas ===")
    for idx, text in acc.items():
        print(f"  out={idx}  {text!r}")
        try:
            json.loads(text)
            print("      >>> parse OK")
        except Exception as exc:
            print(f"      >>> parse FAILED: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
