"""Measure EXL3's 6-bit PLE table against AutoRound's BF16 PLE table.

Both checkpoints store the same Qwen3.8-Flash-Next n-gram table with identical
head offsets (verified), so row i is the same embedding: AutoRound keeps it in
BF16 (320 B/row), EXL3 in a 6-bit trellis (122 B/row). This reads N random rows
from both, decodes the EXL3 rows with ple_decode_out, and reports the error.

    python _dev/probe/_ple_precision_compare.py --rows 256
"""

import argparse
import json
import struct

import numpy as np
import torch

EXL3_DIR = r"D:\models\Qwen3.8-Flash-Next-exl3"
AR_DIR = r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.trellis"
AR_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
SHARD_ROWS = 2_500_012


def header(path):
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length)), 8 + length


def read_tensor(path, name, dtype):
    meta, base = header(path)
    entry = meta[name]
    start, end = (int(x) for x in entry["data_offsets"])
    with open(path, "rb") as handle:
        handle.seek(base + start)
        raw = handle.read(end - start)
    return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


def ar_shard_map():
    import glob

    mapping = {}
    for path in sorted(glob.glob(f"{AR_DIR}\\*.safetensors")):
        meta, base = header(path)
        for name, entry in meta.items():
            if name.startswith(AR_PREFIX) and name.endswith(".weight"):
                shard = int(name[len(AR_PREFIX) : -len(".weight")])
                mapping[shard] = (path, base + int(entry["data_offsets"][0]))
    return mapping


def read_ar_rows(ids, mapping):
    out = np.empty((len(ids), 160), dtype=np.float16)
    handles = {}
    try:
        for index, row in enumerate(ids):
            shard, local = divmod(int(row), SHARD_ROWS)
            path, offset = mapping[shard]
            handle = handles.get(path)
            if handle is None:
                handle = open(path, "rb")
                handles[path] = handle
            handle.seek(offset + local * 320)
            out[index] = np.frombuffer(handle.read(320), dtype=np.float16)
    finally:
        for handle in handles.values():
            handle.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()

    from vllm.models.qwen4_exp.nvidia.exl3_ple import Exl3PLERowReader
    from vllm.sm80.kernels.exl3.ops import ensure_extension

    ensure_extension()
    table = f"{EXL3_DIR}\\ngram_embedding.safetensors"
    meta, _ = header(table)
    bias = torch.tensor(
        read_tensor(table, KEY.removesuffix(".trellis") + ".head_bias", np.float16)
    )
    offsets = torch.tensor(
        read_tensor(table, KEY.removesuffix(".trellis") + ".head_offsets", np.int64)
    )
    vocab = torch.tensor(
        read_tensor(
            table, KEY.removesuffix(".trellis") + ".head_vocab_sizes", np.int64
        )
    )

    reader = Exl3PLERowReader(table, KEY, 0, 1 << 20)
    try:
        rng = np.random.default_rng(args.seed)
        ids = rng.integers(0, int(offsets[1]), size=args.rows)
        dest = bytearray(len(ids) * reader.row_bytes)
        failed = reader.read_rows(ids.tolist(), memoryview(dest))
        assert not failed, failed
        packed = torch.frombuffer(dest, dtype=torch.int16).reshape(
            len(ids), reader.layout.words
        )
        device = torch.device("cuda", args.device)
        out = torch.empty((len(ids), reader.layout.dim), dtype=torch.bfloat16, device=device)
        torch.ops._exl3_C.ple_decode_out(
            packed.to(device),
            torch.tensor(ids, dtype=torch.int64, device=device),
            bias.to(device),
            offsets.to(device),
            vocab.to(device),
            0,
            int(offsets[1]),
            reader.layout.bits,
            out,
        )
        torch.cuda.synchronize(device)
        decoded = out.float().cpu().numpy()
    finally:
        reader.close()

    reference = read_ar_rows(ids, ar_shard_map()).astype(np.float32)
    diff = decoded - reference
    ref_norm = np.linalg.norm(reference, axis=1)
    rel = np.linalg.norm(diff, axis=1) / np.maximum(ref_norm, 1e-9)
    cos = (decoded * reference).sum(1) / np.maximum(
        np.linalg.norm(decoded, axis=1) * ref_norm, 1e-9
    )
    print(f"rows {len(ids)}  dim {reference.shape[1]}")
    print(f"row norm: ref mean {ref_norm.mean():.4f} max {ref_norm.max():.4f}")
    print(f"relative L2 error: mean {rel.mean():.4%} median {np.median(rel):.4%} p95 {np.percentile(rel, 95):.4%}")
    print(f"cosine: mean {cos.mean():.6f} min {cos.min():.6f}")
    print(f"max |diff|: {np.abs(diff).max():.5f}  mean |diff| {np.abs(diff).mean():.5f}")
    # What a pure bf16 round-trip of the reference would cost, as a yardstick.
    bf16 = reference.astype(np.float16).astype(np.float32)
    rel_bf16 = np.linalg.norm(bf16 - reference, axis=1) / np.maximum(ref_norm, 1e-9)
    print(f"bf16 round-trip relative L2: mean {rel_bf16.mean():.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
