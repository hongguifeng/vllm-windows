"""Can a shared section be pinned and read by the GPU?

The two-instance plan shares one 47.68 GiB host table between two servers. On
Windows that means a named, pagefile-backed section (CreateFileMapping with an
invalid file handle), mapped in each process, then cudaHostRegister'ed per
process in <=14 GiB chunks. cudaHostRegister normally takes private committed
pages (VirtualAlloc); whether it accepts a section view is the open question
this probe answers. It maps the same section twice (as two instances would),
registers both views, and reads rows through each.

    python _dev/probe/_section_pin_probe.py --mib 512
"""

from __future__ import annotations

import argparse
import ctypes
import sys

import numpy as np
import torch

INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
PAGE_READWRITE = 0x04
FILE_MAP_ALL_ACCESS = 0xF001F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=512)
    ap.add_argument("--dim", type=int, default=160)
    ap.add_argument("--spawn-child", action="store_true", help="also read it from a second process")
    args = ap.parse_args()
    size = args.mib << 20

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileMappingW.restype = ctypes.c_void_p
    k32.CreateFileMappingW.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_wchar_p,
    ]
    k32.MapViewOfFile.restype = ctypes.c_void_p
    k32.MapViewOfFile.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t,
    ]
    name = "Local\\dsh-ple-table-probe"
    h = k32.CreateFileMappingW(INVALID_HANDLE_VALUE, None, PAGE_READWRITE, 0, size, name)
    if not h:
        print(f"CreateFileMapping failed: {ctypes.get_last_error()}")
        return 1
    print(f"section {name} {size / 2**20:.0f} MiB created")

    rt = torch.cuda.cudart()
    views = []
    for i in (0, 1):
        # The second view stands in for the other instance's mapping.
        v = int(k32.MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size))
        if not v:
            print(f"MapViewOfFile[{i}] failed: {ctypes.get_last_error()}")
            return 1
        rc = rt.cudaHostRegister(v, size, 0)
        print(f"view {i} 0x{v:x}: cudaHostRegister -> rc={rc}")
        if rc != 0:
            return 1
        views.append(v)

    rows = size // args.dim
    pat = np.frombuffer(bytes((i * 37 + 11) & 0xFF for i in range(4096)), dtype=np.uint8)
    base_view = np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(views[0]))
    reps = size // 4096 + 1
    base_view[:] = np.tile(pat, reps)[:size]
    print("filled the section through view 0")

    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        _lookup_ple_embedding_from_pinned_kernel,
    )
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    import triton

    if args.spawn_child:
        import subprocess
        import sys as _sys

        child = str(__file__).replace("_section_pin_probe.py", "_section_pin_child.py")
        r = subprocess.run([_sys.executable, child, name, str(args.mib), str(args.dim)])
        print(f"child exit: {r.returncode}")

    n = 512
    ok = True
    for i, v in enumerate(views):
        raw = torch.frombuffer((ctypes.c_uint8 * size).from_address(v), dtype=torch.uint8)
        uva = get_accelerator_view_from_cpu_tensor(raw)
        ids = torch.randint(0, rows, (n,), dtype=torch.int64, device="cuda:0")
        out = torch.empty(n, args.dim, dtype=torch.uint8, device="cuda:0")
        _lookup_ple_embedding_from_pinned_kernel[(n,)](
            uva, ids, out, args.dim, 0, rows, BLOCK_D=triton.next_power_of_2(args.dim)
        )
        got = out.cpu().numpy()
        offs = ids.cpu().numpy()[:, None] * args.dim + np.arange(args.dim)[None, :]
        want = pat[offs % 4096]
        same = bool(np.array_equal(got, want))
        ok = ok and same
        print(f"view {i} gather {n} rows: {'MATCH' if same else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

