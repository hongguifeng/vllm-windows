"""Verify the ported `no-output-token-reservation` behaviour.

Upstream reserves the requested output budget in the prompt-length pre-check:
TokenizeParams computes max_input_tokens = max_total_tokens - max_output_tokens
and rejects a longer prompt with a 400, even though the prompt alone fits the
context window. The ported patch flips `reserve_output_tokens` to False for the
OpenAI protocol builders, so the input check fires only when the prompt alone
exceeds max_model_len.

Expected, with max_model_len=16384 and max_tokens=8000:
  stock    -> rejected (prompt of 13000 > 16384 - 8000 = 8384)
  patched  -> accepted, generation clamped at the context boundary

Run:  python _test_token_reserve.py
"""

import os
import sys
import time

MODEL = os.environ["VBENCH_MODEL"]
MAX_LEN = int(os.environ.get("VMAXLEN", "16384"))
PROMPT_TOKENS = int(os.environ.get("VPROMPT_TOKENS", "13000"))

_HERE = os.path.dirname(os.path.abspath(__file__))
_CWD = os.path.abspath(os.getcwd())
sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in (_HERE, _CWD)]


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        max_model_len=MAX_LEN,
        gpu_memory_utilization=float(os.environ.get("VMEM", "0.90")),
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        language_model_only=True,
        enforce_eager=True,  # this test is about admission, not speed
        trust_remote_code=True,
    )

    tok = llm.get_tokenizer()
    # Build a prompt of roughly PROMPT_TOKENS tokens.
    body = "The quick brown fox jumps over the lazy dog. " * (PROMPT_TOKENS // 9)
    n_prompt = len(tok.encode(body))
    max_tokens = 8000
    reserved_limit = MAX_LEN - max_tokens

    print(f"max_model_len      = {MAX_LEN}")
    print(f"max_tokens asked   = {max_tokens}")
    print(f"stock would allow  = {reserved_limit} prompt tokens")
    print(f"actual prompt      = {n_prompt} tokens")
    print(f"=> prompt is {n_prompt - reserved_limit:+d} tokens past the stock limit")
    print()

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.time()
    try:
        outs = llm.generate([body], sp)
    except Exception as exc:  # noqa: BLE001
        print("RESULT: REJECTED")
        print(f"  {type(exc).__name__}: {str(exc)[:300]}")
        print("  -> the reservation pre-check is still in force (stock behaviour)")
        return 1

    produced = len(outs[0].outputs[0].token_ids)
    reason = outs[0].outputs[0].finish_reason
    print("RESULT: ACCEPTED")
    print(f"  generated {produced} tokens in {time.time() - t0:.1f}s"
          f" (asked {max_tokens}, context clamps it), finish_reason={reason!r}")
    print("  -> prompt passed the input check on its own length alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
