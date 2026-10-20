"""Does the EXL3-ported tree still resolve an AutoRound checkpoint?

Load-free: import the tree, run plugin registration, then resolve the
quantization config for D:/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP and
check the EXL3-only hooks are absent, so the AutoRound path is untouched.
"""

import os
import sys

MODEL = r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
FAIL = []


def note(label, value):
    print(f"           {label}: {value}")


print(f"python      {sys.version.split()[0]}")
print(f"PYTHONPATH  {os.environ.get('PYTHONPATH')}")

import vllm  # noqa: E402

print(f"vllm        {vllm.__file__}")
tree = os.path.dirname(os.path.abspath(vllm.__file__)).lower()
want = os.path.join(os.environ["REPO_TREE"], "vllm").lower()
print(f"tree ok     {tree == want}")
if tree != want:
    FAIL.append(f"wrong tree {tree} != {want}")

from vllm.plugins import load_general_plugins  # noqa: E402

load_general_plugins()
print("plugins     ok")

from vllm.config import ModelConfig  # noqa: E402

mc = ModelConfig(
    model=MODEL,
    tokenizer=MODEL,
    dtype="bfloat16",
    max_model_len=262144,
    skip_tokenizer_init=True,
)
print(f"arch        {mc.architectures}")
print(f"hf class    {type(mc.hf_config).__name__}")
print(f"quant name  {mc.quantization}")

from vllm.config import LoadConfig  # noqa: E402
from vllm.model_executor.model_loader.weight_utils import get_quant_config  # noqa: E402

qc = get_quant_config(mc, LoadConfig())
print(f"quant class {type(qc).__name__}")

HOOKS = (
    "mtp_shares_target_embeddings",
    "set_mtp_start_layer_idx",
    "get_ple_embedding_method",
    "ple_embedding_cls",
    "shared_expert_shard_group_size",
    "quantize_lm_head",
)
for hook in HOOKS:
    present = getattr(qc, hook, None) is not None
    print(f"hook {hook:34s} {present}")
    if present:
        FAIL.append(f"EXL3 hook {hook} present on the AutoRound config")

# The MTP draft path dereferences these only when the config declares them.
draft_ok = getattr(qc, "set_mtp_start_layer_idx", None) is None
print(f"draft path safe                 {draft_ok}")

print()
if FAIL:
    print(f"RESULT FAIL ({len(FAIL)})")
    for line in FAIL:
        print(f"  - {line}")
    sys.exit(1)
print("RESULT OK")
