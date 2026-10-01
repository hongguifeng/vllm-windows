"""Single-allocation ceiling probe: can one cudaMalloc exceed 41 GiB?

The 15:25 vLLM OOM was ONE torch.zeros(40.99 GiB) refused while this card
accepts 64x1 GiB chunked+written (see _vram_ceiling.py). Test whether the
MCDM segment caps SINGLE allocation size: try one big alloc of N GiB,
verify writable, free, next N. Commit-guarded like the ceiling probe.
"""
import ctypes
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

COMMIT_FLOOR = 15 << 30
SIZES_GIB = [41, 45, 50, 55, 58, 60, 62, 64]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def avail_commit_bytes():
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPageFile, st.dwMemoryLoad


def main():
    total, free = torch.cuda.mem_get_info(0)
    print(f"(mem_get_info says total {total / 2**30:.2f} / free {free / 2**30:.2f} GiB -- known to be nonsense on this stack)")
    print("-" * 60, flush=True)

    for n in SIZES_GIB:
        ac, _ = avail_commit_bytes()
        if ac < COMMIT_FLOOR:
            print(f"[ABORT] commit headroom {ac / 2**30:.1f} GiB < 15 GiB")
            break
        nbytes = n * (1 << 30)
        label = f"single torch.zeros({n} GiB)"
        try:
            t = torch.zeros(nbytes // 4, dtype=torch.float32, device="cuda:0")
            t[-100_000_000:].fill_(1.0)   # touch the TAIL
            torch.cuda.synchronize()
            s = t[:100_000_000].sum().item()  # read back head
            print(f"[OK      ] {label}  (head sum {s:.3e})")
            del t
            torch.cuda.empty_cache()
        except RuntimeError as e:
            msg = str(e).splitlines()[0][:100]
            print(f"[REFUSED ] {label}: {msg}")
            torch.cuda.empty_cache()
    print("-" * 60)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
