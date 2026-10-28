"""How much host memory can this box pin for the GPU, and in what shape?

The recipe serves the n-gram table from one 47.68 GiB pinned host allocation.
A single `torch.empty(..., pin_memory=True)` dies well below that here, so the
question is whether the ceiling is per allocation (chunking could live with it)
or total (the host-table path cannot fit this box at all).

    python _dev/probe/_pin_memory_cap_probe.py --mode chunks --chunk-mb 1024
    python _dev/probe/_pin_memory_cap_probe.py --mode register --gib 24
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time

import torch


def err_name(rc: int) -> str:
    return {0: "ok", 2: "cudaErrorMemoryAllocation", 1: "cudaErrorInvalidValue"}.get(rc, str(rc))


def mode_chunks(chunk_mb: int, limit_gib: float) -> int:
    rt = torch.cuda.cudart()
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    held = []
    total = 0.0
    t0 = time.perf_counter()
    while total < limit_gib:
        ptr = k32.VirtualAlloc(None, chunk_mb * 2**20, 0x3000, 0x04)
        if not ptr:
            print(f"stopped at {total:.2f} GiB: VirtualAlloc error {ctypes.get_last_error()}")
            break
        rc = rt.cudaHostRegister(int(ptr), chunk_mb * 2**20, 0)
        if rc != 0:
            print(f"stopped at {total:.2f} GiB: cudaHostRegister {err_name(rc)}")
            break
        held.append(ptr)
        total += chunk_mb / 1024
        print(f"  +{chunk_mb} MiB -> {total:.2f} GiB pinned ({time.perf_counter() - t0:.1f} s)")
    print(f"RESULT chunks: {total:.2f} GiB in {len(held)} registrations of {chunk_mb} MiB")
    for ptr in held:
        rt.cudaHostUnregister(int(ptr))
        k32.VirtualFree(ctypes.c_void_p(ptr), 0, 0x8000)
    return 0


def mode_register(gib: float) -> int:
    rt = torch.cuda.cudart()
    nbytes = int(gib * 2**30)
    # A pagefile-backed section is a normal user allocation; registering it is
    # the pattern the recipe's shared table uses.
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    MEM_COMMIT_RESERVE = 0x3000
    PAGE_READWRITE = 0x04
    t0 = time.perf_counter()
    ptr = k32.VirtualAlloc(None, nbytes, MEM_COMMIT_RESERVE, PAGE_READWRITE)
    if not ptr:
        print(f"VirtualAlloc failed: {ctypes.get_last_error()}")
        return 1
    print(f"VirtualAlloc {gib:.2f} GiB ok in {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    rc = rt.cudaHostRegister(int(ptr), nbytes, 0)
    print(f"cudaHostRegister -> {err_name(rc)} after {time.perf_counter() - t0:.1f} s")
    if rc == 0:
        rc2, dev = rt.cudaHostGetDevicePointer(int(ptr), 0)
        print(f"cudaHostGetDevicePointer -> {err_name(rc2)} device={dev}")
        rt.cudaHostUnregister(int(ptr))
    return 0 if rc == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chunks", "register"], default="chunks")
    ap.add_argument("--chunk-mb", type=int, default=1024)
    ap.add_argument("--gib", type=float, default=48.0)
    args = ap.parse_args()
    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    if args.mode == "chunks":
        return mode_chunks(args.chunk_mb, args.gib)
    return mode_register(args.gib)


if __name__ == "__main__":
    sys.exit(main())

