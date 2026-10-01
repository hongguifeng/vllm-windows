"""Build a PLE-free view of the Qwen3.8-Flash-Next checkpoint.

Why: the current Windows build (v0.29.0) can only keep the PLE n-gram table on
the GPU -- there is no Engram / host-offload path in this tree.  The table is
95.37 GiB, so the real checkpoint cannot be loaded here at all.  Dropping the
PLE layer leaves ~47.3 GiB of everything else (GDN, QSA, hyper-connections, the
INC 2/3-bit `RoutedExperts` MoE, the vision tower), which does fit on one 64 GiB
CMP 170HX.  That turns "can this architecture run on Windows at all" into a
question that can be answered without first upgrading the whole tree.

The view is a directory of hardlinks to the copied shards plus two edited
metadata files:

  * `model.safetensors.index.json` -- keys containing `.ngram_embedding.shard_`
    are removed, so the loader never opens the 95.37 GiB shard.
  * `config.json`                  -- `text_config.ple_layer_ids` becomes `[]`,
    so `Qwen4ExpDecoderLayer` never constructs a `Qwen4ExpPLELayer`.

Nothing in the source directory is modified.

Run:  D:\\code\\vllm-windows\\.venv\\Scripts\\python.exe D:\\code\\vllm-windows\\_dev\\probe\\_flashnext_trim.py
Then serve it with the current build, on the free card:
  $env:CUDA_VISIBLE_DEVICES='1'
  .venv\\Scripts\\python.exe -m vllm.entrypoints.openai.api_server --model D:\\models\\Qwen3.8-Flash-Next-struct ...
"""

import argparse
import json
import os
import sys
from pathlib import Path

DEV = Path(__file__).resolve().parent.parent
REPO = DEV.parent

PLE_KEY_MARKER = ".ngram_embedding.shard_"
PLE_SHARD = "model-00001-of-00011.safetensors"

DEFAULT_SRC = Path(r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP")
DEFAULT_DST = Path(r"D:\models\Qwen3.8-Flash-Next-struct")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        print(f"source not present yet: {src}")
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"index not copied yet: {index_path}")
        return 1
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    dropped = {k for k in weight_map if PLE_KEY_MARKER in k}
    kept = {k: v for k, v in weight_map.items() if k not in dropped}
    ple_bytes = sum(
        (src / f).stat().st_size
        for f in {weight_map[k] for k in dropped}
        if (src / f).exists()
    )
    print(f"index        : {len(weight_map)} tensors -> keeping {len(kept)}, "
          f"dropping {len(dropped)} PLE rows shards")
    print(f"dropped bytes: {ple_bytes/2**30:.2f} GiB (the shard stays untouched)")

    # Metadata is left exactly as it was: `total_size` only feeds progress bars,
    # and recomputing it per tensor is not worth the extra pass here.
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({**index, "weight_map": kept}), encoding="utf-8")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    removed_ids = cfg.get("text_config", {}).get("ple_layer_ids")
    cfg["text_config"]["ple_layer_ids"] = []
    (dst / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    print(f"config       : ple_layer_ids {removed_ids} -> [] "
          "(no Qwen4ExpPLELayer is built)")

    linked, skipped, missing = 0, 0, []
    for entry in sorted(src.iterdir()):
        if not entry.is_file():
            continue
        if entry.name == PLE_SHARD:
            skipped += 1
            continue
        if entry.name in ("config.json", "model.safetensors.index.json"):
            continue                      # the edited copies already live here
        target = dst / entry.name
        if target.exists():
            target.unlink()
        try:
            os.link(entry, target)
            linked += 1
        except OSError as e:
            print(f"  hardlink failed on {entry.name}: {e}")
            missing.append(entry.name)
    print(f"hardlinks    : {linked} linked, {skipped} skipped "
          f"(the PLE shard), {len(missing)} failed")
    resident = sum((dst / f).stat().st_size for f in os.listdir(dst))
    print(f"view size    : {resident/2**30:.2f} GiB of files reachable here "
          f"(shards + metadata; weights vLLM will map ~= 47.3 GiB)")
    print(f"\nserve it with the CURRENT build, on the free card:\n"
          f"  $env:CUDA_VISIBLE_DEVICES='1'\n"
          f"  {REPO}\\.venv\\Scripts\\python.exe -m vllm.entrypoints.openai.api_server "
          f"--model {dst} --language-model-only --dtype bfloat16 "
          f"--max-model-len 4096 --gpu-memory-utilization 0.90 --port 8111")
    return 0


if __name__ == "__main__":
    sys.exit(main())
