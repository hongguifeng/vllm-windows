"""Enable the vocab-truncated MTP draft head for Qwen3.5/3.8 speculative decoding.

``Qwen3.8-27B-W4A16-AutoRound-*`` ships a *dedicated* MTP head whose LM head is
pruned to a 40 960-row draft vocabulary::

    mtp.draft_lm_head.weight_{packed,scale,shape}   (model_extra_tensors.safetensors)
    mtp_draft_vocab_ids.pt                          (40960 int64 ids -> full vocab)

Upstream vLLM's ``Qwen3_5MTP`` knows nothing about that layout.  It remaps
``mtp.*`` -> ``model.*`` and then tries to load ``model.draft_lm_head`` into
``Qwen3_5MultiTokenPredictor``, which has no such submodule::

    ValueError: There is no module or parameter named 'draft_lm_head' in
    Qwen3_5MultiTokenPredictor.

Port of the HyperQwen (WSL) patch ``qwen3_5-mtp-draft-vocab``, re-anchored onto
vLLM 0.29 and made idempotent.  Three edits, all inside ``qwen3_5_mtp.py``:

1. ``Qwen3_5MultiTokenPredictor.__init__`` -- build the pruned ``draft_lm_head``
   when the checkpoint directory has ``mtp_draft_vocab_ids.pt``.
2. ``Qwen3_5MTP.__init__``                -- a matching pruned ``LogitsProcessor``.
3. ``Qwen3_5MTP.load_weights``            -- drop ``draft_lm_head`` tensors when
   the pruned head is disabled, so the shared full head still loads.

Speculative decoding stays exact -- rejection sampling is done by the target
model -- only the acceptance rate is affected.  Set ``MTP_DRAFT_VOCAB=0`` to go
back to the shared full ``lm_head``.

Python-only, so no rebuild is needed: patch the installed package and restart.
Both the installed copy and the repo source should be patched, otherwise a
reinstall silently reintroduces the bug.

Usage:
    fix_qwen3_5_mtp_draft_vocab.py <vllm_package_dir> [<vllm_package_dir> ...]
e.g.
    fix_qwen3_5_mtp_draft_vocab.py .venv/Lib/site-packages/vllm vllm
"""

import os
import sys

FILENAME = "qwen3_5_mtp.py"

MARKER = "# syv patch: vocab-truncated draft head"

# 1. Draft head construction, inserted just before the mtp.fc workaround block
#    so it does not depend on whether fix_qwen3_5_embed_quant.py has run yet.
ANCHOR_INIT = "        # Workaround: mtp.fc is stored as BF16"

BLOCK_INIT = '''        # syv patch: vocab-truncated draft head. If the checkpoint ships
        # mtp_draft_vocab_ids.pt (built by build_draft_vocab.py) the drafter
        # scores only those rows (mtp.draft_lm_head.*) instead of the full
        # 248k-row lm_head; logits for all other ids are -inf. Speculative
        # decoding stays exact, only the acceptance rate can change.
        import os as _os
        self.draft_lm_head = None
        self.draft_vocab_ids = None
        _ids_path = _os.path.join(model_config.model, "mtp_draft_vocab_ids.pt")
        if _os.path.exists(_ids_path) and _os.environ.get("MTP_DRAFT_VOCAB", "1") != "0":
            _ids = torch.load(_ids_path, map_location="cpu")
            self.draft_vocab_ids = _ids
            self.draft_lm_head = ParallelLMHead(
                int(_ids.numel()),
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "draft_lm_head"),
            )
            logger.info("MTP drafter uses a %d-token draft head", int(_ids.numel()))

'''

# 2. Pruned logits processor, right after the full-vocab one.
ANCHOR_LOGITS = "        self.logits_processor = LogitsProcessor(config.vocab_size)\n"

BLOCK_LOGITS = '''        # syv patch: vocab-truncated draft head
        self.draft_logits_processor = (
            LogitsProcessor(int(self.model.draft_vocab_ids.numel()))
            if getattr(self.model, "draft_lm_head", None) is not None
            else None
        )
'''

# 3. Skip the pruned head's tensors when the pruned head is disabled.
ANCHOR_REMAP = """        def remap_weight_names(weights):
            for name, weight in weights:
"""

BLOCK_REMAP = """                # syv patch: skip the truncated draft head when it is disabled
                if "draft_lm_head" in name and self.model.draft_lm_head is None:
                    continue
"""

EDITS = (
    ("__init__ (draft head)", ANCHOR_INIT, BLOCK_INIT + "\n" + ANCHOR_INIT),
    ("compute_logits (pruned processor)", ANCHOR_LOGITS, ANCHOR_LOGITS + BLOCK_LOGITS),
    ("load_weights (skip when disabled)", ANCHOR_REMAP, ANCHOR_REMAP + BLOCK_REMAP),
)


def patch_file(path: str) -> int:
    with open(path, mode="r", encoding="utf-8") as file:
        content = file.read()

    if MARKER in content:
        print(f"already patched: {path}")
        return 0

    for label, anchor, replacement in EDITS:
        if anchor not in content:
            print(f"WARN: anchor not found ({label}), left untouched: {path}")
            return 0
        content = content.replace(anchor, replacement, 1)

    with open(path, mode="w", encoding="utf-8", newline="") as file:
        file.write(content)
    print(f"patched: {path}")
    return 1


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    patched = 0
    for pkg_dir in sys.argv[1:]:
        models_dir = os.path.join(pkg_dir, "model_executor", "models")
        if not os.path.isdir(models_dir):
            print(f"skip (not a vllm package dir): {pkg_dir}")
            continue
        path = os.path.join(models_dir, FILENAME)
        if not os.path.exists(path):
            print(f"skip (missing): {path}")
            continue
        patched += patch_file(path)

    print(f"done, {patched} file(s) patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
