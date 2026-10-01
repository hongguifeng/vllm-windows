"""Smoke the batch-2 patches without spinning up an engine.

Imports the two new Triton modules, exercises the flash_attn dispatch helpers,
and prints the knob defaults. Fast enough to catch a bad import or a missing
symbol before a five-minute benchmark run.
"""

import os
import sys

# This script lives in the repo root, so its own directory lands in sys.path[0]
# and `import vllm` would hit the unbuilt source tree, not the installed wheel.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CWD = os.path.abspath(os.getcwd())
sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in (_HERE, _CWD)]

import vllm.envs as envs  # noqa: E402

print("knob defaults:")
for name in (
    "VLLM_PREFILL_ATTN",
    "VLLM_SPEC_DECODE_ATTN",
    "VLLM_SPEC_DECODE_ATTN_QMAX",
    "VLLM_SPEC_ATTN_BLOCK_M",
    "VLLM_DRAFT_TOPK_TOPP",
    "VLLM_DRAFT_TEMP_SCALE",
):
    print(f"  {name} = {getattr(envs, name)!r}")

import vllm  # noqa: F401,E402
from vllm.v1.attention.backends import prefill_attn_hd256  # noqa: E402

print("prefill_attn_hd256 ok:", callable(prefill_attn_hd256.prefill_attn))

import vllm  # noqa: F401,E402
from vllm.v1.attention.ops import spec_decode_attn  # noqa: E402

print("spec_decode_attn ok:", hasattr(spec_decode_attn, "spec_decode_attn"))

import vllm  # noqa: F401,E402
from vllm.v1.attention.backends import flash_attn  # noqa: E402

for helper in ("_spec_attn_enabled", "_spec_attn_qmax", "_spec_attn_run"):
    print(f"  flash_attn.{helper}: {hasattr(flash_attn, helper)}")
