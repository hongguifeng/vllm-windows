"""Ask a running vLLM server which chat prompt it actually rendered.

Sends the same conversation with and without `reasoning_effort` in
chat_template_kwargs, asks for the prompt token ids (`return_token_ids`), and
decodes them with the local tokenizer. The Qwen template injects a long
"Reasoning effort is set to xhigh..." system sentence when the variable is
missing and nothing when it is "medium", so the prompt length tells which one
the server passed through.

    python _chat_prompt_probe.py --base http://127.0.0.1:9393/v1 --model Qwen3.8-Flash-Next
"""

import argparse
import json
import urllib.request

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
]


def ask(base: str, model: str, kwargs: dict | None) -> dict:
    body = {
        "model": model,
        "messages": MESSAGES,
        "max_tokens": 1,
        "temperature": 0,
        "return_token_ids": True,
    }
    if kwargs is not None:
        body["chat_template_kwargs"] = kwargs
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(request, timeout=300).read())


def prompt_ids(response: dict) -> list[int] | None:
    choice = response["choices"][0]
    for container in (choice, response):
        if container.get("prompt_token_ids"):
            return container["prompt_token_ids"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    cases = {
        "medium": {"enable_thinking": True, "reasoning_effort": "medium"},
        "xhigh": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "omitted": {"enable_thinking": True},
    }
    for label, kwargs in cases.items():
        response = ask(args.base, args.model, kwargs)
        ids = prompt_ids(response)
        if ids is None:
            print(label, "no prompt_token_ids in response; keys:", sorted(response))
            print(json.dumps(response)[:400])
            continue
        text = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"{label:8s} prompt_tokens={len(ids):3d} chars={len(text):4d} xhigh={('xhigh' in text)}")
        print(f"         {text[:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
