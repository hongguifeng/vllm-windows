"""Route the Qwen3.5 token embedding table through vLLM's quantized-embedding path.

Checkpoints such as ``Qwen3.8-27B-W4A16-AutoRound-*`` pack-quantize
``embed_tokens`` (int8, ``group_size=128``) in addition to the usual int4 linears.
vLLM's ``CompressedTensorsEmbeddingWNA16Int`` kernel already handles that, but the
qwen3_5 model code never passes ``quant_config`` to ``VocabParallelEmbedding``, so
weight loading dies with:

    ValueError: There is no module or parameter named 'embed_tokens.weight_packed'
    in Qwen3_5Model. The available parameters belonging to embed_tokens
    (VocabParallelEmbedding) are: {'embed_tokens.weight'}

Two call sites have to be fixed -- the main model and the MTP draft module; fixing
only the first one makes single-user (MTP) mode fail on
``Qwen3_5MultiTokenPredictor``.

Python-only, so no rebuild is needed: patch the installed package and restart.
Both the installed copy and the repo source should be patched, otherwise a
reinstall silently reintroduces the bug.

Usage:
    fix_qwen3_5_embed_quant.py <vllm_package_dir> [<vllm_package_dir> ...]
e.g.
    fix_qwen3_5_embed_quant.py .venv/Lib/site-packages/vllm vllm
"""

import os
import sys

NEEDLE = "        self.embed_tokens = VocabParallelEmbedding(\n            self.vocab_size,\n            config.hidden_size,\n        )\n"

# Both spellings below are equivalent; `vllm_config.quant_config` is used in the
# MTP module to stay identical to the patch tracked by the upstream project.
REPLACEMENTS = {
    "qwen3_5.py": (
        "        self.embed_tokens = VocabParallelEmbedding(\n"
        "            self.vocab_size,\n"
        "            config.hidden_size,\n"
        "            quant_config=self.quant_config,\n"
        '            prefix=maybe_prefix(prefix, "embed_tokens"),\n'
        "        )\n"
    ),
    "qwen3_5_mtp.py": (
        "        self.embed_tokens = VocabParallelEmbedding(\n"
        "            self.vocab_size,\n"
        "            config.hidden_size,\n"
        "            quant_config=vllm_config.quant_config,\n"
        '            prefix=maybe_prefix(prefix, "embed_tokens"),\n'
        "        )\n"
    ),
}

patched = 0
for pkg_dir in sys.argv[1:]:
    models_dir = os.path.join(pkg_dir, "model_executor", "models")
    if not os.path.isdir(models_dir):
        print(f"skip (not a vllm package dir): {pkg_dir}")
        continue
    for filename, replacement in REPLACEMENTS.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            print(f"skip (missing): {path}")
            continue
        with open(path, mode="r", encoding="utf-8") as file:
            content = file.read()
        if replacement in content:
            print(f"already patched: {path}")
            continue
        if NEEDLE not in content:
            print(f"WARN: anchor not found, left untouched: {path}")
            continue
        content = content.replace(NEEDLE, replacement, 1)
        with open(path, mode="w", encoding="utf-8", newline="") as file:
            file.write(content)
        print(f"patched: {path}")
        patched += 1

print(f"done, {patched} file(s) patched")
