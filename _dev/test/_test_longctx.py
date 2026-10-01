"""Long-context sanity check against a running vLLM OpenAI server.

Builds a ~30k-token prompt with a secret buried in the middle and asks the
model to retrieve it.  A correct answer proves the long-context path really
works (chunked prefill + correct position handling) rather than just proving
the server accepts a large `max_model_len`.

    python _test_longctx.py [target_chars] [endpoint]
"""

import json
import os
import sys
import time
import urllib.request

TARGET_CHARS = int(sys.argv[1]) if len(sys.argv) > 1 else 120_000
ENDPOINT = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/v1/chat/completions"
MODEL = os.environ.get("VMODEL_NAME", "Qwen3.8-27B-W4A16-AutoRound-fast")

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
)
NEEDLE = "IMPORTANT NOTE: the secret passcode is 7391-QKXM. Keep it in mind."
HALF = max(TARGET_CHARS // 2 // len(FILLER), 1)

context = FILLER * HALF + "\n" + NEEDLE + "\n" + FILLER * HALF
prompt = (
    context
    + "\n\nBased only on the text above, what is the secret passcode?"
    + " Answer with just the passcode, nothing else.\n"
)

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 64,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}

print(f"prompt chars   : {len(prompt):,}  (~{len(prompt) // 4:,} tokens expected)")
req = urllib.request.Request(
    ENDPOINT,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as resp:
    out = json.loads(resp.read())
wall = time.time() - t0

usage = out["usage"]
answer = out["choices"][0]["message"]["content"].strip()
print(f"prompt_tokens  : {usage['prompt_tokens']:,}")
print(f"completion_tok : {usage['completion_tokens']}")
print(f"wall time      : {wall:.1f} s")
if usage["completion_tokens"]:
    print(f"generation     : {usage['completion_tokens'] / wall:.1f} tok/s (includes prefill)")
print(f"answer         : {answer!r}")
print("NEEDLE FOUND   :", "7391-QKXM" in answer)
