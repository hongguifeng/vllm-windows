"""Print the runtime knobs _run.ps1 is responsible for. Used to verify wiring."""

import os
import sys

# This file lives in the repo root, so its own directory becomes sys.path[0] and
# `import vllm` would hit the unbuilt source tree instead of the installed wheel.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CWD = os.path.abspath(os.getcwd())
sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in (_HERE, _CWD)]

KEYS = [
    "VLLM_MARLIN_INPUT_DTYPE",
    "VLLM_MARLIN_INT8_INCLUDE_RE",
    "VLLM_MARLIN_INT8_EXCLUDE_RE",
    "VLLM_MARLIN_REPACK_STAGED",
    "VLLM_ENGINE_STALL_SENTINEL_S",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VMODEL",
    "VMAXLEN",
    "VMEM",
]

for key in KEYS:
    print(f"  {key:32s} = {os.environ.get(key)!r}")

import vllm.envs as envs  # noqa: E402

print()
print("  (via vllm.envs, which applies defaults)")
print(f"  VLLM_MARLIN_REPACK_STAGED default -> {envs.VLLM_MARLIN_REPACK_STAGED!r}"
      "  (None = auto: on for cc 8.0)")
print(f"  VLLM_ENGINE_STALL_SENTINEL_S     -> {envs.VLLM_ENGINE_STALL_SENTINEL_S}")
