"""Cold-device random-read probe for the Windows PLE-SSD fallback path.

`_ple_winio_probe.py` established that `FILE_FLAG_NO_BUFFERING` -- the Windows
stand-in for the O_DIRECT that `ple_ssd_io.c` relies on -- is refused outright on
this machine, so the Windows reader has to be buffered and lean on the system
file cache.  That leaves the one number that decides whether the whole SSD
backend is worth porting: **what does a cache-miss row read actually cost?**

A 95.37 GiB PLE table cannot be cached by 88 GiB of RAM, so some share of lookups
will always fall through to the device.  This measures that fall-through by
reading a file bigger than RAM that was written with `FILE_FLAG_WRITE_THROUGH`
(so it never entered the cache to begin with).

Two shapes are timed:
  * 4 KiB random reads -- what the SSD really fetches per row (one LBA block),
    and what the Linux O_DIRECT path asks for.
  * 320 B random reads -- what a buffered Windows `pread` stand-in would ask for.
    The device still moves a whole page, so this only differs in syscall count.

Run:  D:\\code\\vllm-windows\\.venv\\Scripts\\python.exe D:\\code\\vllm-windows\\_dev\\probe\\_ple_coldread_probe.py [--make 100] [--keep]

Writes ~100 GiB to D:\\_ple_cold_probe.bin by default.  Check free space first:
the ext4.vhdx for WSL lives on the same NVMe and the running WSL server is
SSD-delivery-bound, so this probe competes with it.
"""

import argparse
import ctypes
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
REPO = DEV.parent

ROW_BYTES = 320
GENERIC_WRITE = 0x40000000
GENERIC_READ = 0x80000000
FILE_SHARE_READ_WRITE = 0x00000007
CREATE_ALWAYS = 2
OPEN_EXISTING = 3
FILE_FLAG_WRITE_THROUGH = 0x80000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.argtypes = [
    ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
]
k32.CreateFileW.restype = ctypes.c_void_p
k32.WriteFile.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
    ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p,
]
k32.WriteFile.restype = ctypes.c_int
k32.ReadFile.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
    ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p,
]
k32.ReadFile.restype = ctypes.c_int
k32.CloseHandle.argtypes = [ctypes.c_void_p]
k32.GetDiskFreeSpaceExW.argtypes = [
    ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
]
k32.GetDiskFreeSpaceExW.restype = ctypes.c_int


def make_cold_file(path: Path, gib: float) -> int:
    """Sequential write-through: bytes land on the device, not in the cache."""
    flags = FILE_FLAG_WRITE_THROUGH | FILE_FLAG_RANDOM_ACCESS
    h = k32.CreateFileW(str(path), GENERIC_WRITE, FILE_SHARE_READ_WRITE, None,
                        CREATE_ALWAYS, flags, None)
    if h in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    chunk = bytes(1 << 20)
    written = 0
    t0 = time.perf_counter()
    while written < int(gib * 2**30):
        got = ctypes.c_uint(0)
        for _ in range(16):                      # 16 MiB per WriteFile call
            if not k32.WriteFile(h, chunk, len(chunk), ctypes.byref(got), None):
                raise ctypes.WinError(ctypes.get_last_error())
        written += 16 << 20
        if written % (4 << 30) == 0:
            el = time.perf_counter() - t0
            print(f"    wrote {written/2**30:5.1f} GiB  "
                  f"{written/2**20/el:7.1f} MiB/s", flush=True)
    el = time.perf_counter() - t0
    k32.CloseHandle(h)
    print(f"    created {written/2**30:.1f} GiB in {el:.1f} s "
          f"({written/2**20/el:.1f} MiB/s sustained write-through)")
    return written


def random_reads(path: Path, n_threads: int, reads_per_thread: int,
                 span: int, size: int) -> tuple[list[float], float, int]:
    lat: list[float] = []
    lock = threading.Lock()
    total_bytes = 0
    total_reads = 0

    def worker(tid: int):
        nonlocal total_bytes, total_reads
        buf = ctypes.create_string_buffer(size)
        got = ctypes.c_uint(0)
        h = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ_WRITE, None,
                            OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, None)
        local: list[float] = []
        my_bytes = 0
        my_reads = 0
        try:
            # Distinct stride per thread so the threads do not collapse onto the
            # same pages and inflate the cache-hit rate.
            pos = (tid * 104729 + 7) % span
            step = max(4096, span // reads_per_thread)
            for i in range(reads_per_thread):
                off = (pos + i * step) & ~4095 if size == 4096 else (pos + i * step)
                off %= span
                t0 = time.perf_counter()
                ok = k32.ReadFile(h, buf, size, ctypes.byref(got), None)
                t1 = time.perf_counter()
                if not ok or got.value != size:
                    raise ctypes.WinError(ctypes.get_last_error())
                local.append((t1 - t0) * 1e3)
                my_bytes += size
                my_reads += 1
        finally:
            k32.CloseHandle(h)
        with lock:
            lat.extend(local)
            total_bytes += my_bytes
            total_reads += my_reads

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    return lat, wall, total_reads, total_bytes


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(int(q * len(s)), len(s) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=r"D:\_ple_cold_probe.bin")
    ap.add_argument("--make", type=float, default=100.0,
                    help="GiB to create; 0 reuses whatever is there")
    ap.add_argument("--reads", type=int, default=30000, help="reads per thread")
    ap.add_argument("--keep", action="store_true", help="leave the file behind")
    args = ap.parse_args()

    path = Path(args.path)
    free = 0
    avail = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    root = str(path.parent.parent) if path.parent else None
    if root and k32.GetDiskFreeSpaceExW(root, ctypes.byref(avail),
                                        ctypes.byref(total), None):
        free = avail.value / 2**30
    print("==== PLE cold-read probe ====")
    print(f"target      : {path}")
    print(f"free on vol : {free:.0f} GiB")
    if args.make > 0:
        if free < args.make + 40:
            print(f"ABORT: need {args.make + 40:.0f} GiB free, only {free:.0f} GiB")
            return 1
        print("-- 1. build a write-through file (bypasses the file cache) --")
        make_cold_file(path, args.make)
    if not path.exists():
        print("no file at --path and --make 0; nothing to measure")
        return 1
    size = path.stat().st_size
    if size < 1 << 30:
        print("file too small to be colder than RAM; nothing to measure")
        return 1
    span = size - 4096
    print(f"file        : {size/2**30:.1f} GiB, measuring over its whole span")
    print()

    print("-- 2. buffered random reads, sync ReadFile --")
    for label, size_ in (("4096 B (one LBA, = what the SSD moves per row)", 4096),
                         ("320 B  (a buffered pread stand-in, row-shaped)", ROW_BYTES)):
        print(f"  {label}")
        for n_threads in (1, 4, 8, 16, 32):
            lat, wall, n, nbytes = random_reads(path, n_threads, args.reads, span, size_)
            print(f"    threads {n_threads:2d}: {n/wall:7.0f} reads/s  "
                  f"p50 {pct(lat, 0.5):6.3f} ms  p90 {pct(lat, 0.9):6.3f} ms  "
                  f"p99 {pct(lat, 0.99):6.3f} ms  max {max(lat):6.2f} ms")
    print()

    print("-- 3. what that means for PLE --")
    print("  one PLE layer, 16 n-gram heads per token, 320 B rows; a 4 KiB page")
    print("  holds 12.8 rows, so unique *pages* -- not rows -- set the device work.")
    print("  rows per decode step = batch x 16; rows per 8K prefill chunk = 2048 x 16.")
    print()

    if not args.keep:
        try:
            os.unlink(str(path))
            print(f"removed {path}")
        except OSError as e:
            print(f"could not remove {path}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
