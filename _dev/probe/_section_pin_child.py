"""Second process for _section_pin_probe: open the shared table, pin it, read it.

    python _dev/probe/_section_pin_child.py <section-name> <mib> <dim>

Exits 0 only if the gather through this process's own registration matches the
periodic pattern the parent wrote, i.e. one physical copy served two processes.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import torch

FILE_MAP_ALL_ACCESS = 0xF001F


def main() -> int:
    name, mib, dim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    size = mib << 20
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenFileMappingW.restype = ctypes.c_void_p
    k32.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    k32.MapViewOfFile.restype = ctypes.c_void_p
    k32.MapViewOfFile.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t,
    ]
    h = k32.OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, name)
    if not h:
        print(f"[child] OpenFileMapping failed: {ctypes.get_last_error()}")
        return 1
    v = int(k32.MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size))
    if not v:
        print(f"[child] MapViewOfFile failed: {ctypes.get_last_error()}")
        return 1
    rt = torch.cuda.cudart()
    rc = rt.cudaHostRegister(v, size, 0)
    print(f"[child] view 0x{v:x} cudaHostRegister -> rc={rc}")
    if rc != 0:
        return 1

    rows = size // dim
    pat = np.frombuffer(bytes((i * 37 + 11) & 0xFF for i in range(4096)), dtype=np.uint8)
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        _lookup_ple_embedding_from_pinned_kernel,
    )
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    import triton

    raw = torch.frombuffer((ctypes.c_uint8 * size).from_address(v), dtype=torch.uint8)
    uva = get_accelerator_view_from_cpu_tensor(raw)
    n = 512
    ids = torch.randint(0, rows, (n,), dtype=torch.int64, device="cuda:0")
    out = torch.empty(n, dim, dtype=torch.uint8, device="cuda:0")
    _lookup_ple_embedding_from_pinned_kernel[(n,)](
        uva, ids, out, dim, 0, rows, BLOCK_D=triton.next_power_of_2(dim)
    )
    got = out.cpu().numpy()
    offs = ids.cpu().numpy()[:, None] * dim + np.arange(dim)[None, :]
    ok = bool(np.array_equal(got, pat[offs % 4096]))
    print(f"[child] gather through the shared section: {'MATCH' if ok else 'MISMATCH'}")
    rt.cudaHostUnregister(v)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
