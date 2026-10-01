"""Backport Qwen sparse-attention layer names into the deployed vLLM tree.

The Windows venv contains Python code paired with the existing Marlin native
extensions.  The checkpoint uses the newer explicit ``qwen_sparse_attention``
layer name, while this deployed Python revision only accepts ``full_attention``.
Keep this narrow compatibility change reproducible and preserve an .orig file.
"""
from pathlib import Path
import shutil


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    path = (
        repo
        / ".venv"
        / "Lib"
        / "site-packages"
        / "vllm"
        / "models"
        / "qwen4_exp"
        / "nvidia"
        / "model.py"
    )
    text = path.read_text(encoding="utf-8")
    if "QSA_LAYER_TYPE = \"qwen_sparse_attention\"" not in text:
        orig = path.with_name(path.name + ".orig")
        if not orig.exists():
            shutil.copy2(path, orig)
        text = text.replace(
            "from .qsa import Qwen4ExpQSAAttention\n",
            "from .qsa import Qwen4ExpQSAAttention\n\n"
            'QSA_LAYER_TYPE = "qwen_sparse_attention"\n'
            'ATTENTION_LAYER_TYPES = ("full_attention", QSA_LAYER_TYPE)\n',
            1,
        )
        text = text.replace(
            '        elif layer_type == "full_attention":\n'
            '            use_qsa = getattr(config, "indexer_n_heads", None) is not None\n',
            '        elif layer_type in ATTENTION_LAYER_TYPES:\n'
            '            use_qsa = (\n'
            '                layer_type == QSA_LAYER_TYPE\n'
            '                or getattr(config, "indexer_n_heads", None) is not None\n'
            '            )\n',
            1,
        )
        text = text.replace(
            '        elif self.layer_type == "full_attention":\n',
            '        elif self.layer_type in ATTENTION_LAYER_TYPES:\n',
            1,
        )
        text = text.replace(
            '            if layer_type == "full_attention"\n'
            '            and getattr(config, "indexer_n_heads", None) is not None\n',
            '            if layer_type == QSA_LAYER_TYPE\n'
            '            or (\n'
            '                layer_type == "full_attention"\n'
            '                and getattr(config, "indexer_n_heads", None) is not None\n'
            '            )\n',
            1,
        )
        path.write_text(text, encoding="utf-8", newline="")
    print(f"QSA compatibility present: {path}")


if __name__ == "__main__":
    main()
