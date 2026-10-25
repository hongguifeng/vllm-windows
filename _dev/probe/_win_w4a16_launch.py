"""Launch the Windows vLLM server for the W4A16 checkpoint on two 170HX cards.

Arguments are built as a Python list, so JSON-valued flags need no shell
quoting. The body is guarded by ``if __name__ == "__main__"``: vLLM sets the
multiprocessing start method to forkserver, and its children re-import this
module. Without the guard every child starts a second API server, and both
bind the same handshake port (zmq EADDRINUSE).
"""
import json
import os
import runpy
import sys

ADHOC = {
    "ple_ssd_offload": True,
    "ple_ssd_native_library":
        "D:/code/vllm-windows/_dev/out/ple_ssd_io/ple_ssd_io_win.dll",
    "ple_ssd_io_depth": 512,
    "ple_ssd_cache_mb": 2048,
    "ple_ssd_workers": 16,
    "ple_ssd_prefetch_tokens": 2048,
}
MODEL = r"D:\models\Qwen3.8-Flash-Next-W4A16-AutoRound"


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["VLLM_PP_LAYER_PARTITION"] = "26,22"
    os.environ["VLLM_PLE_ALLOW_PP"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONFAULTHANDLER"] = "1"

    args = [
        "api_server",
        "--model", MODEL,
        "--served-model-name", "Qwen3.8-Flash-Next-W4A16",
        "--host", "127.0.0.1",
        "--port", "9394",
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-seqs", "2",
        "--max-num-batched-tokens", "1024",
        "--gpu-memory-utilization", "0.95",
        "--mamba-cache-mode", "align",
        "--moe-backend", "triton",
        "--block-size", "1680",
        "--pipeline-parallel-size", "2",
        "--data-parallel-rpc-port", "29994",
        "--enforce-eager",
        "--load-format", "dummy",
        "--additional-config", json.dumps(ADHOC, separators=(",", ":")),
    ]
    if "--with-spec" in sys.argv:
        args += [
            "--speculative-config",
            json.dumps({
                "method": "dflash",
                "model": "D:/models/Qwen3.8-Flash-Next-DFlash2",
                "num_speculative_tokens": 7,
                "disable_eagle_block_drop": True,
            }, separators=(",", ":")),
        ]
    if "--with-graphs" in sys.argv:
        args = [a for a in args if a != "--enforce-eager"]

    sys.argv = args
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()




