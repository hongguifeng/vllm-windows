"""Benchmark Qwen3.8-27B-W4A16 on the CMP 170HX.

Separates the three numbers that the ported patches actually move:

  engine init  -- Marlin repack window dominates this (marlin-repack-staged-sm80)
  prefill      -- int8 tensor cores on GA100 (VLLM_MARLIN_INPUT_DTYPE=int8)
  decode       -- should be unchanged; used as a regression guard

Every knob comes from the environment so the same script can be run against the
stock tree and the patched tree without edits.

  VBENCH_MODEL   model path / repo id          (required)
  VBENCH_TAG     label printed in the summary  (default "run")
  VMAXLEN        max_model_len                 (default 16384)
  VMEM           gpu_memory_utilization        (default 0.90)
  VMAXSEQS       max_num_seqs                  (default 8)
  VBATCHED       max_num_batched_tokens        (default 4096)
  VONEPROMPT     1 = single-stream tests only
  VPREFILL_TOKENS  tokens per prefill prompt   (default 8000)
  VDECODE_TOKENS   tokens generated per decode (default 256)
  VSPEC          JSON for `speculative_config` (default: off). e.g.
                 {"method":"mtp","num_speculative_tokens":1} or
                 {"method":"dflash","model":"D:/models/<drafter dir>",
                  "num_speculative_tokens":7}
"""

import json
import os
import sys
import time

MODEL = os.environ["VBENCH_MODEL"]
TAG = os.environ.get("VBENCH_TAG", "run")
MAX_LEN = int(os.environ.get("VMAXLEN", "16384"))
UTIL = float(os.environ.get("VMEM", "0.90"))
MAX_SEQS = int(os.environ.get("VMAXSEQS", "8"))
BATCHED = int(os.environ.get("VBATCHED", "4096"))
ONE_PROMPT = os.environ.get("VONEPROMPT", "0") == "1"

PREFILL_TOKENS = int(os.environ.get("VPREFILL_TOKENS", "8000"))
DECODE_TOKENS = int(os.environ.get("VDECODE_TOKENS", "256"))
SPEC = os.environ.get("VSPEC", "").strip()
BATCH_SIZE = int(os.environ.get("VBATCH_SIZE", "8"))

# The script lives in the repo root, so its own directory lands in sys.path[0]
# and `import vllm` would hit the unbuilt source tree instead of the installed
# wheel. Drop both that directory and the cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CWD = os.path.abspath(os.getcwd())
sys.path = [
    p for p in sys.path if os.path.abspath(p or ".") not in (_HERE, _CWD)
]

results = {}


def hr(label):
    print("\n" + "=" * 68)
    print(label)
    print("=" * 68)


def report(key, value, unit=""):
    results[key] = value
    print(f"  >> {key} = {value}{unit}")


def main():
    import torch
    from vllm import LLM, SamplingParams

    hr(f"device ({TAG})")
    print("  gpu          :", torch.cuda.get_device_name(0))
    print("  capability   :", torch.cuda.get_device_capability(0))
    print("  total memory : %.2f GiB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print("  free at start: %.2f GiB" % ((torch.cuda.mem_get_info()[0]) / 1024**3))
    print("  speculative  :", SPEC or "(none)")

    hr("engine init")
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        max_model_len=MAX_LEN,
        gpu_memory_utilization=UTIL,
        max_num_seqs=MAX_SEQS,
        max_num_batched_tokens=BATCHED,
        language_model_only=True,
        enforce_eager=False,
        trust_remote_code=True,
        **({"speculative_config": json.loads(SPEC)} if SPEC else {}),
        # Prefix caching is left ON because this model's Mamba cache mode is
        # derived from it (`align` when on), so turning it off would benchmark a
        # different engine. The prefill prompts are salted with a per-call nonce
        # instead, so a timed run never reads its own warm-up out of the cache.
    )
    init_s = time.time() - t0
    report("engine_init_s", round(init_s, 1), " s")

    # vLLM logs the split we care about; grab it from the engine's own config.
    # (In V1 the block counts are only populated on the engine core process, so
    # read the tokens-per-request the log line reports instead of guessing here.)
    try:
        cfg = llm.llm_engine.vllm_config.cache_config
        n_blocks = getattr(cfg, "num_gpu_blocks", 0) or 0
        report("kv_cache_gib", round(n_blocks * cfg.block_size * 1 / 1024**3, 2), " GiB")
    except Exception as exc:  # noqa: BLE001
        print("  (kv cache introspection unavailable:", type(exc).__name__, ")")

    torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------------------- prefill
    hr(f"prefill: 1 prompt of ~{PREFILL_TOKENS} tokens, 1 output token")
    repeats = max(1, PREFILL_TOKENS // 9)
    prefill_prompt = "The quick brown fox jumps over the lazy dog. " * repeats
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    tok = llm.get_tokenizer()
    n_prompt = len(tok.encode(prefill_prompt))
    # One repeat is ~11 tokens, not the 9 the divider suggests, so asking for N
    # repeats overshoots N -- and the renderer rejects the whole call with a 400
    # instead of truncating. Scale down once, so an over-large VPREFILL_TOKENS
    # degrades into a shorter prompt rather than a dead run.
    if n_prompt > MAX_LEN - 8:
        repeats = max(1, repeats * (MAX_LEN - 8) // n_prompt)
        prefill_prompt = "The quick brown fox jumps over the lazy dog. " * repeats
        n_prompt = len(tok.encode(prefill_prompt))
    print(f"  actual prompt tokens: {n_prompt}")
    t0 = time.time()
    llm.generate([prefill_prompt], sp1)
    prefill_s = time.time() - t0
    report("prefill_s", round(prefill_s, 3), " s")
    report("prefill_tok_s", round(n_prompt / prefill_s, 1), " tok/s")

    # ----------------------------------------------------------------- decode
    hr(f"decode: 1 stream, {DECODE_TOKENS} tokens")
    short = "Explain what a graphics processing unit does."
    spd = SamplingParams(temperature=0.0, max_tokens=DECODE_TOKENS, ignore_eos=True)
    llm.generate([short], spd)  # warm
    t0 = time.time()
    outs = llm.generate([short], spd)
    decode_s = time.time() - t0
    n_out = len(outs[0].outputs[0].token_ids)
    report("decode_s", round(decode_s, 3), " s")
    report("decode_tok_s", round(n_out / decode_s, 1), " tok/s")

    # --------------------------------------------------------- batched decode
    if not ONE_PROMPT:
        hr(f"batched decode: {BATCH_SIZE} streams x {DECODE_TOKENS} tokens")
        prompts = [short] * BATCH_SIZE
        llm.generate(prompts, spd)  # warm
        t0 = time.time()
        outs = llm.generate(prompts, spd)
        batch_s = time.time() - t0
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        report("batch_s", round(batch_s, 3), " s")
        report("batch_tok_s", round(total / batch_s, 1), " tok/s")

        hr(f"long-batch prefill: {BATCH_SIZE} distinct x ~{PREFILL_TOKENS // 2} tokens")
        # Salting each call keeps the timed run from reading the warm-up out of
        # the prefix cache -- with a shared prefix the aggregate measured cache
        # lookups, not prefill, and came out ~8x too fast.
        body = "The quick brown fox jumps over the lazy dog. " * (PREFILL_TOKENS // 18)
        long_prompts = [f"[{time.time_ns()} stream {i}] {body}" for i in range(BATCH_SIZE)]
        llm.generate(long_prompts, sp1)  # warm kernels / graph buckets
        long_prompts = [f"[{time.time_ns()} stream {i}] {body}" for i in range(BATCH_SIZE)]
        t0 = time.time()
        llm.generate(long_prompts, sp1)
        lpre_s = time.time() - t0
        n_long = len(llm.get_tokenizer().encode(long_prompts[0]))
        report("batch_prefill_s", round(lpre_s, 3), " s")
        report("batch_prefill_tok_s", round(n_long * BATCH_SIZE / lpre_s, 1), " tok/s")

    # ------------------------------------------------------------- memory cost
    hr("memory")
    report("peak_alloc_gib", round(torch.cuda.max_memory_allocated() / 1024**3, 2), " GiB")
    report("peak_reserved_gib", round(torch.cuda.max_memory_reserved() / 1024**3, 2), " GiB")

    # ------------------------------------------------------ correctness probe
    hr("correctness probe")
    probe = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
    outs = llm.generate(["The capital of France is", "2 + 2 =", "Water boils at"], probe)
    for o in outs:
        print("  ", repr(o.prompt), "->", repr(o.outputs[0].text.strip()[:90]))

    hr("SUMMARY")
    print("  tag =", TAG)
    print(json.dumps(results, indent=2))
    print("BENCH DONE")


if __name__ == "__main__":
    main()
