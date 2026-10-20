"""Dry run of the EXL3 checkpoint manifest on the Windows copy (no weights).

Reads only the safetensors headers of `model`, so it proves the checkpoint is
complete and that vllm/sm80/kernels/exl3/utils.py accepts it on this platform
(NTFS paths, Windows file handles) before any 115 GiB load.

    python _dev/probe/_exl3_manifest_win_test.py "D:\\models\\Qwen3.8-Flash-Next-exl3"
"""

import argparse
import time
from pathlib import Path

from vllm.sm80.kernels.exl3.utils import (
    Exl3Manifest,
    aligned_shard,
    ple_layout,
    read_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = Exl3Manifest(args.model, require_qwen_aux=True)
    elapsed = time.perf_counter() - started
    print(
        f"manifest ok: tensors={len(manifest.tensors)} "
        f"matrices={len(manifest.matrices)} ple_tables={len(manifest.ple_tables)} "
        f"unused={len(manifest.unused)} in {elapsed:.2f}s"
    )

    for prefix, header in manifest.ple_tables.items():
        layout = ple_layout(manifest.tensors, read_metadata(header.path), prefix)
        print(
            f"  PLE {prefix}: rows={layout.rows} dim={layout.dim} "
            f"words={layout.words} row_bytes={layout.row_bytes} heads={layout.heads}"
        )

    payload = sum(header.nbytes for header in manifest.tensors.values())
    print(f"payload {payload / 2**30:.2f} GiB over {len(manifest.tensors)} tensors")

    for prefix, matrix in list(manifest.matrices.items())[: args.show]:
        shard = aligned_shard(matrix.padded_in, matrix.padded_in, 0, args.tp)
        print(
            f"  {prefix}: padded {matrix.padded_in}x{matrix.padded_out} "
            f"bits_x2={matrix.bits_x2} codebook={matrix.codebook} "
            f"tp{args.tp} rank0 in={shard.start}:{shard.stop}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
