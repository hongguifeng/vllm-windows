"""Load the local Qwen3.8-27B W4A16 AutoRound checkpoint with the built vLLM and generate."""

import os
import sys
import time
import traceback

# Python prepends this script's directory (the repo root) to sys.path, and the
# repo root contains an unbuilt copy of the `vllm` source tree.  That copy would
# shadow the installed package and fail on the missing extension modules, so
# drop it from the search path before importing anything.
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _REPO]

MODEL = os.environ.get("VMODEL27B", r"D:\models\Qwen3.8-27B-W4A16-AutoRound-fast")
MAX_LEN = int(os.environ.get("VMAXLEN", "8192"))
UTIL = float(os.environ.get("VMEM", "0.90"))
MAX_SEQS = int(os.environ.get("VMAXSEQS", "4"))
BATCHED = int(os.environ.get("VBATCHED", "2048"))
TEXT_ONLY = os.environ.get("VTEXTONLY", "1") == "1"
EAGER = os.environ.get("VEAGER", "1") == "1"
# "float16" is what the RTX 3090 reference launcher uses, but it disqualifies the
# fused CUDA GDN kernel (it wants a BF16 convolution cache) and vLLM then falls
# back to the slower Triton path.  Leave it unset to keep vLLM's own default.
MAMBA_DTYPE = os.environ.get("VMAMBADTYPE", "").strip()


def main() -> int:
    print("python:", sys.version.split()[0])

    import torch

    print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
    prop = torch.cuda.get_device_properties(0)
    print("device:", prop.name, "| cc:", (prop.major, prop.minor))
    free, total = torch.cuda.mem_get_info()
    print("vram free/total GB:", round(free / 1024**3, 2), "/", round(total / 1024**3, 2))

    import vllm

    print("vllm:", vllm.__version__)

    if not os.path.isdir(MODEL):
        print("ERROR: model dir not found:", MODEL)
        return 1

    from vllm import LLM, SamplingParams

    print(
        f"loading {MODEL}\n"
        f"  max_model_len={MAX_LEN} gpu_memory_utilization={UTIL} "
        f"max_num_seqs={MAX_SEQS} max_num_batched_tokens={BATCHED}\n"
        f"  language_model_only={TEXT_ONLY} enforce_eager={EAGER}"
    )

    t0 = time.time()
    kwargs = dict(
        model=MODEL,
        max_model_len=MAX_LEN,
        gpu_memory_utilization=UTIL,
        max_num_seqs=MAX_SEQS,
        max_num_batched_tokens=BATCHED,
        language_model_only=TEXT_ONLY,
        enforce_eager=EAGER,
        trust_remote_code=True,
    )
    if MAMBA_DTYPE:
        kwargs["mamba_ssm_cache_dtype"] = MAMBA_DTYPE
    print("  mamba_ssm_cache_dtype:", MAMBA_DTYPE or "(vLLM default)")
    llm = LLM(**kwargs)
    print(f"engine init took {time.time() - t0:.1f}s")

    sp = SamplingParams(
        temperature=0.7, top_p=0.8, max_tokens=int(os.environ.get("VMAXTOK", "96"))
    )
    prompts = [
        "Give me a one-sentence summary of what a GPU is.",
        "Write a Python function that returns the n-th Fibonacci number.",
    ]
    if os.environ.get("VONEPROMPT", "0") == "1":
        prompts = prompts[1:2]
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t1
    gen_tokens = 0
    for o in outs:
        print("-" * 60)
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text.strip()[:400])
        print("tokens:", len(o.outputs[0].token_ids))
        gen_tokens += len(o.outputs[0].token_ids)
    print(f"generate wall time: {dt:.1f}s | output tokens: {gen_tokens} | "
          f"{gen_tokens / dt:.1f} tok/s (aggregate)")
    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
