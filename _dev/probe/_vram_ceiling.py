"""Find the exact allocatable VRAM ceiling on GPU0 (CMP 170HX, MCDM).

History: the UNCAPPED 15:33 version of this probe caused bugcheck 0xEF --
not via VRAM but via SYSTEM COMMIT: on this stack every VRAM allocation
mirrors ~1:1 into commit, and with a 40 GB bench live on GPU1 the account
hit 130/139.9 GB. This version carries a commit guard (abort below 15 GB
avail) and is only run with GPU1 idle. 2026-09-26 16:3x: user asked for a
direct test of the full 64 GiB; commit headroom verified before launch.

Allocates 1 GiB chunks up to 64 GiB and WRITES each chunk (fill_ forces
real residency; a pure reservation proves nothing). Two stop conditions:
  - driver refuses an allocation -> clean OOM, that IS the ceiling answer
  - commit headroom drops below the red line -> abort, no repeat of 15:34

Run from OUTSIDE the repo:
    D:/code/vllm-windows/.venv/Scripts/python.exe D:/code/vllm-windows/_dev/probe/_vram_ceiling.py
"""

import ctypes
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

CHUNK_BYTES = 1 << 30            # 1 GiB
TARGET_CHUNKS = 64               # nominal 64 GiB
COMMIT_FLOOR = 15 << 30          # abort if avail commit < 15 GB


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
    ac, load = avail_commit_bytes()
    print(f"driver-reported total : {total / 2**30:.2f} GiB")
    print(f"driver-reported free  : {free / 2**30:.2f} GiB")
    print(f"commit avail at start : {ac / 2**30:.1f} GiB (mem load {load}%)")
    print("-" * 56, flush=True)

    chunks = []
    peak = 0
    write_ms = []
    for i in range(1, TARGET_CHUNKS + 1):
        ac, load = avail_commit_bytes()
        if ac < COMMIT_FLOOR:
            print(f"[ABORT] commit headroom {ac / 2**30:.1f} GiB < 15 GiB at chunk {i}")
            break
        try:
            t = torch.empty(CHUNK_BYTES // 4, dtype=torch.float32, device="cuda:0")
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            t.fill_(1.0)          # force residency, timed
            torch.cuda.synchronize()
            t1.record()
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1)
            gbps = CHUNK_BYTES / (ms / 1000) / 1e9
            write_ms.append((i, ms, gbps))
            chunks.append(t)
            peak = i
            if i % 4 == 0 or i >= 56:
                print(f"chunk {i:2d}  write {ms:8.2f} ms  ({gbps:7.1f} GB/s)   total {i} GiB   commit avail {ac / 2**30:.1f} GiB", flush=True)
        except RuntimeError as e:
            msg = str(e).splitlines()[0][:90]
            print(f"[REFUSED] at chunk {i} (total {i - 1} GiB allocated): {msg}")
            break
    else:
        print("[DONE] reached 64 chunks with no refusal?!")

    # second pass: re-read every chunk; demoted (system-RAM-backed) pages will
    # read back at PCIe speed instead of HBM speed -> slow chunks = not in VRAM
    if peak:
        print("-" * 56)
        print("second pass: read-back timing (slow = demoted to system RAM)")
        hbm, slow = [], []
        for i, t in enumerate(chunks, 1):
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            s = t.sum().item()
            torch.cuda.synchronize()
            t1.record()
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1)
            gbps = CHUNK_BYTES / (ms / 1000) / 1e9
            (hbm if gbps > 300 else slow).append(i)
            if i % 4 == 0 or i >= 56 or gbps <= 300:
                print(f"chunk {i:2d}  read {ms:8.2f} ms  ({gbps:7.1f} GB/s)" + ("   <-- SLOW: demoted?" if gbps <= 300 else ""), flush=True)
        print(f"HBM-speed chunks : {hbm[0]}..{hbm[-1]} ({len(hbm)} chunks)" if hbm else "none")
        print(f"slow chunks      : {slow[0]}..{slow[-1]} ({len(slow)} chunks)" if slow else "no slow chunks")

    print("-" * 56)
    print(f"RESULT: max allocated+written = {peak} GiB")
    if peak:
        print(f"        ghost vs declared 64 GiB = {64 - peak:.1f} GiB")
    del chunks
    torch.cuda.empty_cache()
    print("cleaned up")


if __name__ == "__main__":
    sys.exit(main())
