"""Single-allocation cudaHostAlloc ceiling, without torch's pinned allocator.

`torch.empty(..., pin_memory=True)` fails at 12 GiB here while 1 GiB
registrations reach 41 GiB. This calls the CUDA runtime directly so the two
limits can be told apart: a torch allocator cap, or the driver's host-pinning
budget.

    python _dev/probe/_cudart_hostalloc_probe.py 12 24 40
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch

DLL = Path(torch.__file__).parent / "lib" / "cudart64_13.dll"


def main() -> int:
    sizes = [float(x) for x in sys.argv[1:]] or [12.0]
    rt = ctypes.CDLL(str(DLL))
    rt.cudaSetDevice(0)
    rt.cudaHostAlloc.restype = ctypes.c_int
    rt.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    rt.cudaFreeHost.argtypes = [ctypes.c_void_p]
    rt.cudaGetErrorString.restype = ctypes.c_char_p
    rt.cudaGetErrorString.argtypes = [ctypes.c_int]
    rt.cudaHostGetDevicePointer.restype = ctypes.c_int
    rt.cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    for gib in sizes:
        nbytes = int(gib * 2**30)
        ptr = ctypes.c_void_p()
        rc = rt.cudaHostAlloc(ctypes.byref(ptr), nbytes, 0)
        if rc != 0:
            print(f"cudaHostAlloc {gib:.2f} GiB -> {rc}: {rt.cudaGetErrorString(rc).decode()}")
            continue
        dev = ctypes.c_void_p()
        rc2 = rt.cudaHostGetDevicePointer(ctypes.byref(dev), ptr, 0)
        print(
            f"cudaHostAlloc {gib:.2f} GiB -> ok host=0x{ptr.value:x} "
            f"dev=0x{dev.value:x} (getdev {rc2})"
        )
        rt.cudaFreeHost(ptr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
