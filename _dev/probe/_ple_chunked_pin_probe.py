"""Can one contiguous host buffer be pinned in chunks and still be usable?

A 47.68 GiB PLE table cannot be pinned on Windows in one call (single
cudaHostAlloc/cudaHostRegister above ~16 GiB fails with cudaErrorMemoryAllocation
-- the same wall the KV tier hit at 17.16 GB). The workaround would be one
VirtualAlloc reservation pinned in <=14 GiB registrations. That only works if
the device view stays contiguous across registration boundaries, because the
lookup kernel indexes rows from a single base pointer.

This probe builds such a buffer, registers it in N chunks, and checks that rows
inside the last chunk read back correctly through the base device pointer.

    python _dev/probe/_ple_chunked_pin_probe.py --gib 2 --chunks 4
"""

from __future__ import annotations

import argparse
import ctypes
import sys

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=2.0)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--dim", type=int, default=160)
    args = ap.parse_args()

    rt = torch.cuda.cudart()
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    rows = int(args.gib * 2**30) // args.dim
    total = rows * args.dim
    base = int(k32.VirtualAlloc(None, total, 0x3000, 0x04))
    if not base:
        print(f"VirtualAlloc failed: {ctypes.get_last_error()}")
        return 1
    print(f"reserved {total / 2**30:.2f} GiB at 0x{base:x} for {rows} rows x {args.dim} B")

    step = ((rows // args.chunks) * args.dim // 4096) * 4096  # page-aligned chunks
    for i in range(args.chunks):
        off = i * step
        n = min(step, total - off)
        rc = rt.cudaHostRegister(base + off, n, 0)
        if rc != 0:
            print(f"chunk {i} register {n / 2**30:.2f} GiB -> failed rc={rc}")
            return 1
        print(f"  chunk {i}: 0x{base + off:x} +{n / 2**30:.2f} GiB registered")

    # UVA: the device pointer of registered host memory is the host pointer. The
    # kernel needs a typed pointer, so wrap the reservation as a torch tensor and
    # take the accelerator view the engine uses.
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    raw = torch.frombuffer((ctypes.c_uint8 * total).from_address(base), dtype=torch.uint8)
    uva = get_accelerator_view_from_cpu_tensor(raw)
    dev = uva
    print(f"UVA base 0x{base:x} -> torch view {tuple(uva.shape)} on {uva.device} ptr=0x{uva.data_ptr():x} same={uva.data_ptr() == base}")

    pat = np.frombuffer(bytes((i * 37 + 11) & 0xFF for i in range(4096)), dtype=np.uint8)
    view = np.ctypeslib.as_array((ctypes.c_uint8 * total).from_address(base))
    block = 64 << 20
    for off in range(0, total, block):
        n = min(block, total - off)
        reps = n // 4096 + 1
        view[off : off + n] = np.tile(pat, reps)[:n]
    print("filled the whole reservation with a 4 KiB periodic pattern")

    n = 512
    last_chunk_start = (args.chunks - 1) * step
    ids = torch.randint(last_chunk_start // args.dim, rows, (n,), dtype=torch.int64, device="cuda:0")
    out = torch.empty(n, args.dim, dtype=torch.uint8, device="cuda:0")

    import triton

    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        _lookup_ple_embedding_from_pinned_kernel,
    )

    _lookup_ple_embedding_from_pinned_kernel[(n,)](
        dev, ids, out, args.dim, 0, rows, BLOCK_D=triton.next_power_of_2(args.dim)
    )
    got = out.cpu().numpy()
    offs = ids.cpu().numpy()[:, None] * args.dim + np.arange(args.dim)[None, :]
    want = pat[offs % 4096]
    ok = bool(np.array_equal(got, want))
    print(f"rows from the last chunk read through the base pointer: {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        bad = np.nonzero((got != want).any(axis=1))[0][:3].tolist()
        print(f"  first bad rows: {bad}")

    for i in range(args.chunks):
        rt.cudaHostUnregister(base + i * step)
    k32.VirtualFree(ctypes.c_void_p(base), 0, 0x8000)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())



