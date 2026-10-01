"""End-to-end smoke test for vLLM on the CMP 170HX (GA100 / sm_80)."""

import os
import sys
import traceback

# Python prepends this script's directory (the repo root) to sys.path, and the
# repo root contains an unbuilt copy of the `vllm` source tree.  That copy would
# shadow the installed package and fail on the missing extension modules, so
# drop it from the search path before importing anything.
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _REPO]


def main() -> int:
    print("cwd:", os.getcwd())
    print("python:", sys.version.split()[0])

    try:
        import torch

        print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available())
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("mem GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    except Exception:
        traceback.print_exc()

    try:
        import vllm

        print("vllm:", vllm.__version__)
        print("vllm file:", vllm.__file__)
    except Exception:
        traceback.print_exc()
        return 1

    try:
        from vllm import LLM, SamplingParams

        model = os.environ.get("VMODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        eager = os.environ.get("VEAGER", "0") == "1"
        util = float(os.environ.get("VMEM", "0.45"))
        print("model:", model, "| enforce_eager:", eager, "| gpu_mem_util:", util)

        llm = LLM(
            model=model,
            max_model_len=2048,
            gpu_memory_utilization=util,
            enforce_eager=eager,
            trust_remote_code=True,
        )
        sp = SamplingParams(temperature=0.7, max_tokens=48)
        prompts = ["Hello, my name is", "The capital of France is"]
        outs = llm.generate(prompts, sp)
        for o in outs:
            print("PROMPT:", o.prompt)
            print("OUTPUT:", o.outputs[0].text.replace("\n", "\\n"))
        print("SMOKE TEST OK")
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
