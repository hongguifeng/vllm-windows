"""Bit-exact test of ple_decode_out on Windows against a CPU reference.

Reads real rows of ngram_embedding.safetensors through Exl3PLERowReader, runs
the EXL3 PLE decode kernel on a 170HX, and compares with a numpy port of
csrc/exl3/ple.cu (mul1 codebook + ring bit walk + FP16 head bias).

    python _dev/probe/_exl3_ple_decode_win_test.py "D:\\models\\Qwen3.8-Flash-Next-exl3"
"""

import argparse
from pathlib import Path

import numpy as np
import torch

KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.trellis"


def _half(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def mul1_value(state: int) -> float:
    product = (state * 0x83DCD12D) & 0xFFFFFFFF
    byte_sum = (
        (product & 255)
        + ((product >> 8) & 255)
        + ((product >> 16) & 255)
        + (product >> 24)
    )
    k_inv = np.float32(_half(0x1EEE))
    k_bias = np.float32(_half(0xC931))
    value = np.float32(np.float32(1024 + byte_sum) * k_inv + k_bias)
    return float(np.float16(value))


def reference(
    packed, ids, bias, offsets, vocab, dim: int, bits: int, owner_lo: int, owner_hi: int
) -> np.ndarray:
    rows = packed.shape[0]
    out = np.zeros((rows, dim), dtype=np.float32)
    heads = len(offsets)
    for r in range(rows):
        gid = int(ids[r])
        if not owner_lo <= gid < owner_hi:
            continue
        head = -1
        for h in range(heads):
            if offsets[h] <= gid < offsets[h] + vocab[h]:
                head = h
                break
        if head < 0:
            continue
        scale = np.float32(_half(int(packed[r, 0])))
        for c in range(dim):
            state = 0
            for m in range(16):
                back = m // bits
                src = c - back if c >= back else dim - (back - c)
                ring_bit = src * bits + (m % bits)
                word = int(packed[r, 1 + ring_bit // 16]) & 0xFFFF
                state |= ((word >> (ring_bit % 16)) & 1) << m
            scaled = np.float32(mul1_value(state)) * scale
            out[r, c] = scaled + np.float32(bias[head, c])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()

    from vllm.models.qwen4_exp.nvidia.exl3_ple import Exl3PLERowReader
    from vllm.sm80.kernels.exl3.ops import ensure_extension

    ensure_extension()
    reader = Exl3PLERowReader(str(args.model / "ngram_embedding.safetensors"), KEY, 0, 1 << 20)
    try:
        layout = reader.layout
        dim, bits, words = layout.dim, layout.bits, layout.words
        print(f"layout dim={dim} bits={bits} words={words} row_bytes={reader.row_bytes}")

        aux = {}
        for name, header in zip(("bias", "offsets", "vocab", "multipliers"), layout.auxiliaries):
            dtype = np.float16 if header.dtype == "F16" else np.int64
            aux[name] = np.frombuffer(bytes(reader.read_tensor(header)), dtype=dtype).reshape(
                header.shape
            )
        bias = aux["bias"].astype(np.float32)
        offsets = aux["offsets"].astype(np.int64).tolist()
        vocab = aux["vocab"].astype(np.int64).tolist()
        print(f"heads={layout.heads} offsets={offsets} vocab={vocab}")

        # Real rows of head 0 plus one global id outside the owner range.
        ids = list(range(args.rows)) + [int(offsets[-1] + vocab[-1] - 1)]
        target = bytearray(len(ids) * reader.row_bytes)
        failed = reader.read_rows(ids, memoryview(target))
        assert not failed, failed
        packed = (
            torch.frombuffer(target, dtype=torch.int16)
            .reshape(len(ids), words)
            .contiguous()
        )

        device = torch.device("cuda", 1)
        gpu = {k: torch.tensor(v).to(device) for k, v in aux.items()}
        packed_gpu = packed.to(device)
        ids_gpu = torch.tensor(ids, dtype=torch.int64, device=device)
        out = torch.empty((len(ids), dim), dtype=torch.bfloat16, device=device)
        owner_hi = int(offsets[-1] + vocab[-1] - 1)  # exclude the last id
        torch.ops._exl3_C.ple_decode_out(
            packed_gpu,
            ids_gpu,
            gpu["bias"],
            gpu["offsets"],
            gpu["vocab"],
            0,
            owner_hi,
            bits,
            out,
        )
        torch.cuda.synchronize(device)

        want = reference(
            packed.numpy(),
            ids,
            bias,
            offsets,
            vocab,
            dim,
            bits,
            0,
            owner_hi,
        )
        want = torch.tensor(want).to(torch.bfloat16).to(device)
        mismatch = (out != want).sum().item()
        print(f"compared {out.numel()} values, mismatches={mismatch}")
        assert mismatch == 0, (out.float() - want.float()).abs().max().item()
        assert torch.count_nonzero(out[-1]) == 0, "unowned row must decode to zero"
    finally:
        reader.close()
    print("OK ple_decode_out matches the CPU reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
