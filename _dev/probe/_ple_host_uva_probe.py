"""Probe the pinned-host PLE lookup path on Windows.

The recipe serves the n-gram table from pinned host RAM and looks rows up
through a CUDA view of that memory (UVA). This project has only ever run the
SSD offload path, so whether the engine's pinned path works on Windows -- and
whether a table-sized pinned allocation is possible -- is unproven.

This probe answers the first question on a small table: allocate a pinned FP8
table, take the engine's device view of it, run the engine's lookup kernel, and
compare against a CPU gather. It also reports how much pinned memory the
allocator actually reserves, which is what decides whether the real 47.7 GiB
table fits.

    python _dev/probe/_ple_host_uva_probe.py --rows 200000 --dim 2560 --gib 0

--gib N instead allocates N GiB of pinned memory and stops (sizing test only).
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--dim", type=int, default=2560)
    ap.add_argument("--gib", type=float, default=0.0, help="only size-test a pinned allocation")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch

    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        _lookup_ple_embedding_from_pinned_kernel,
    )
    from vllm.utils.platform_utils import is_uva_available
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(args.device)}")
    print(f"is_uva_available: {is_uva_available()}")

    if args.gib:
        nbytes = int(args.gib * 2**30)
        t0 = time.perf_counter()
        buf = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        alloc_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        view = get_accelerator_view_from_cpu_tensor(buf)
        map_s = time.perf_counter() - t0
        print(
            f"pinned {nbytes / 2**30:.2f} GiB: allocate {alloc_s:.1f} s, "
            f"device view {map_s:.1f} s, device {view.device}"
        )
        del view, buf
        return 0

    rows, dim = args.rows, args.dim
    ref = torch.randint(-120, 120, (rows, dim), dtype=torch.int8)
    weight = torch.empty(rows, dim, dtype=torch.float8_e4m3fn, device="cpu", pin_memory=True)
    weight.view(torch.int8).copy_(ref)
    uva = get_accelerator_view_from_cpu_tensor(weight)
    print(f"table {rows}x{dim} fp8 = {rows * dim / 2**20:.1f} MiB pinned; device view ok")

    n = 4096
    ids = torch.randint(0, rows, (n,), dtype=torch.int64, device=args.device)
    out = torch.empty(n, dim, dtype=torch.float8_e4m3fn, device=args.device)

    import triton

    block_d = triton.next_power_of_2(dim)
    _lookup_ple_embedding_from_pinned_kernel[(n,)](
        uva, ids, out, dim, 0, rows, BLOCK_D=block_d
    )
    got = out.view(torch.int8).cpu()
    want = ref[ids.cpu()]
    ok = torch.equal(got, want)
    print(f"lookup of {n} rows: {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        bad = (got != want).any(dim=1).nonzero().flatten()[:5].tolist()
        print(f"  first differing rows: {bad}")
        return 1

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _lookup_ple_embedding_from_pinned_kernel[(n,)](
            uva, ids, out, dim, 0, rows, BLOCK_D=block_d
        )
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    print(f"UVA gather {n} rows: {dt * 1e3:.3f} ms/call ({n / dt / 1e6:.1f} M rows/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
