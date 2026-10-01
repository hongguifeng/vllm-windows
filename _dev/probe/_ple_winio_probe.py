"""Portability + speed probe for the WSL PLE-SSD row reader on Windows.

The reference implementation (qwen3.8-flash-next-cmp170hx, `ple_ssd.py` +
`ple_ssd_io.c`) reads single 320-byte BF16 rows out of a 95.37 GiB safetensors
shard.  Every row read is widened to a 4 KiB-aligned page so O_DIRECT stays
happy, and the rows are pulled out of that page afterwards.

Two questions, answered without booting vLLM:
  1. Which Linux-only calls does the reader depend on, and does a Windows
     stand-in produce the same bytes?
  2. What IOPS does the Windows file stack give for that access pattern --
     plain threaded reads, and an IOCP queue that mirrors `rows_read`'s
     depth-256 Linux AIO batch?

Run:  D:\\code\\vllm-windows\\.venv\\Scripts\\python.exe D:\\code\\vllm-windows\\_dev\\probe\\_ple_winio_probe.py [--file PATH] [--rows N]

The default target is the 27B shard on the NVMe, which is far smaller than RAM.
That makes the *buffered* numbers page-cache hits, not device reads; only the
`NO_BUFFERING` rows measure the disk.  Point --file at something bigger than
RAM to measure a cold buffered path.
"""

import argparse
import ctypes
import os
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = HERE.parent
REPO = DEV.parent

DEFAULT_FILE = REPO / "build"  # placeholder, replaced below
DEFAULT_FILE = Path(
    r"D:\models\Qwen3.8-27B-W4A16-AutoRound-fast\model-00001-of-00007.safetensors"
)

ROW_BYTES = 320          # dim 160 x bf16, exactly what PLE SSD expects
n_rows = 40000           # set from the target file in main()
GENERIC_READ = 0x80000000
FILE_SHARE_READ_WRITE = 0x00000007
OPEN_EXISTING = 3
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_NO_BUFFERING = 0x40000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
MEM_COMMIT_RESERVE = 0x00003000
PAGE_READWRITE = 0x04

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = ctypes.c_void_p
k32.CreateFileW.argtypes = [
    ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
]
k32.CloseHandle.argtypes = [ctypes.c_void_p]
k32.ReadFile.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
    ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p,
]
k32.ReadFile.restype = ctypes.c_int
k32.CreateIoCompletionPort.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint,
]
k32.CreateIoCompletionPort.restype = ctypes.c_void_p
k32.GetQueuedCompletionStatus.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_uint,
]
k32.GetQueuedCompletionStatus.restype = ctypes.c_int
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint,
]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]


class OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_ulong),
        ("InternalHigh", ctypes.c_ulong),
        ("Offset", ctypes.c_uint),
        ("OffsetHigh", ctypes.c_uint),
        ("hEvent", ctypes.c_void_p),
    ]


def page_for(offset: int, row_bytes: int = ROW_BYTES) -> tuple[int, int, int]:
    """The same page widening ple_ssd_io.c does for one row."""
    page = offset & ~4095
    delta = offset - page
    nbytes = (delta + row_bytes + 4095) & ~4095
    return page, delta, nbytes


def alloc_aligned(nbytes: int) -> ctypes.c_void_p:
    # VirtualAlloc hands back 64 KiB-granular, therefore 4096-aligned, memory.
    addr = k32.VirtualAlloc(None, nbytes, MEM_COMMIT_RESERVE, PAGE_READWRITE)
    if not addr:
        raise ctypes.WinError(ctypes.get_last_error())
    return addr


# ---------------------------------------------------------------- linux-only --
def probe_linux_apis() -> dict[str, bool]:
    checks = {
        "os.pread": hasattr(os, "pread"),
        "os.posix_fadvise": hasattr(os, "posix_fadvise"),
        "os.O_DIRECT": hasattr(os, "O_DIRECT"),
        "/proc/self/fd/3 readable": os.path.exists("/proc/self/fd/3"),
    }
    return checks


def win_pread(fd: int, count: int, offset: int) -> bytes:
    """Windows stand-in for os.pread: seek+read under a per-fd lock."""
    with _pread_lock:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, count)


_pread_lock = threading.Lock()


def reference_rows(path: Path, rows: list[int], base: int) -> bytes:
    """Ground truth straight from the file, one row at a time."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        out = bytearray()
        for row in rows:
            out += win_pread(fd, ROW_BYTES, base + row * ROW_BYTES)
        return bytes(out)
    finally:
        os.close(fd)


def extract_from_pages(pages: dict[int, bytes], offsets: list[int]) -> bytes:
    out = bytearray()
    for off in offsets:
        page, delta, _ = page_for(off)
        out += pages[page][delta:delta + ROW_BYTES]
    return bytes(out)


# --------------------------------------------------- synchronous, many threads --
def sync_read(path: Path, n_threads: int, rows_per_thread: int, base: int,
              no_buffering: bool) -> tuple[list[float], int, int]:
    flags = (FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS) if no_buffering else 0
    handle = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ_WRITE, None,
                             OPEN_EXISTING, flags, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    lat: list[float] = []
    bytes_read = 0
    rows_read = 0
    lock = threading.Lock()

    def worker(tid: int):
        nonlocal bytes_read, rows_read
        buf = alloc_aligned(8192)
        read_len = ctypes.c_uint(0)
        local: list[float] = []
        my_bytes = 0
        my_rows = 0
        h = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ_WRITE, None,
                           OPEN_EXISTING, flags, None)
        try:
            rng = (hash(tid) * 2654435761) % 1000003
            for i in range(rows_per_thread):
                row = (rng + i * 7919) % n_rows
                off = base + row * ROW_BYTES
                page, delta, nbytes = page_for(off)
                t0 = time.perf_counter()
                ok = k32.ReadFile(h, buf, nbytes, ctypes.byref(read_len), None)
                t1 = time.perf_counter()
                if not ok or read_len.value != nbytes:
                    raise ctypes.WinError(ctypes.get_last_error())
                local.append((t1 - t0) * 1e3)
                my_bytes += nbytes
                my_rows += 1
        finally:
            k32.CloseHandle(h)
            k32.VirtualFree(buf, 0, 0x8000)
        with lock:
            lat.extend(local)
            bytes_read += my_bytes
            rows_read += my_rows

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    k32.CloseHandle(handle)
    return lat, wall, rows_read, bytes_read


# ------------------------------------------------------- IOCP, one deep queue --
def iocp_read(path: Path, depth: int, batches: int, base: int) -> tuple[int, int, float, list[float]]:
    flags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED
    handle = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ_WRITE, None,
                             OPEN_EXISTING, flags, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    iocp = k32.CreateIoCompletionPort(handle, None, 0, 0)
    if not iocp:
        raise ctypes.WinError(ctypes.get_last_error())

    total_rows = 0
    total_bytes = 0
    submit_lat: list[float] = []
    done_lat: list[float] = []
    keep: list[OVERLAPPED] = []   # OVERLAPPED must outlive its completion
    t_start = time.perf_counter()

    for b in range(batches):
        # Spread the batch over the table so it looks like real PLE traffic.
        offsets = [
            base + ((b * 7919 + i * 104729) % n_rows) * ROW_BYTES
            for i in range(depth)
        ]
        pending: dict[int, tuple[int, ctypes.c_void_p, int, float]] = {}
        t_submit = time.perf_counter()
        for i, off in enumerate(offsets):
            page, delta, nbytes = page_for(off)
            ov = OVERLAPPED()
            ov.Offset = page & 0xFFFFFFFF
            ov.OffsetHigh = (page >> 32) & 0xFFFFFFFF
            ov.hEvent = None
            buf = alloc_aligned(8192)
            ok = k32.ReadFile(handle, buf, nbytes, None, ctypes.byref(ov))
            if not ok:
                err = ctypes.get_last_error()
                if err != 997:  # ERROR_IO_PENDING
                    raise ctypes.WinError(err)
            pending[ctypes.addressof(ov)] = (i, buf, nbytes, time.perf_counter())
            keep.append(ov)
        submitted = 0
        while submitted < depth:
            n_bytes = ctypes.c_uint(0)
            key = ctypes.c_ulong(0)
            ovlp = ctypes.c_void_p(0)
            ok = k32.GetQueuedCompletionStatus(
                iocp, ctypes.byref(n_bytes), ctypes.byref(key),
                ctypes.byref(ovlp), 5000)
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())
            i, buf, nbytes, t_sub = pending.pop(ovlp.value)
            done_lat.append((time.perf_counter() - t_sub) * 1e3)
            total_bytes += n_bytes.value
            total_rows += 1
            k32.VirtualFree(buf, 0, 0x8000)  # MEM_RELEASE
            submitted += 1
        submit_lat.append((time.perf_counter() - t_submit) * 1e3)

    wall = time.perf_counter() - t_start
    k32.CloseHandle(iocp)
    k32.CloseHandle(handle)
    return total_rows, total_bytes, wall, done_lat


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(int(q * len(s)), len(s) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--rows", type=int, default=40000, help="rows to touch")
    ap.add_argument("--depth", type=int, default=256, help="IOCP queue depth")
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()

    path = Path(args.file)
    size = path.stat().st_size
    global n_rows
    n_rows = min(args.rows, max(1, (size - 8) // ROW_BYTES))

    print("==== PLE Windows I/O probe ====")
    print(f"file      : {path}  ({size/2**30:.2f} GiB)")
    print(f"rows      : {n_rows} x {ROW_BYTES} B "
          f"(~{(n_rows*4300)/2**20:.1f} MiB of page traffic)")
    print(f"ram       : {os.cpu_count()} cores")
    print()

    print("-- 1. Linux-only calls the reader uses --")
    for name, ok in probe_linux_apis().items():
        verdict = "present" if ok else "ABSENT -> needs a Windows stand-in"
        print(f"  {name:28s}: {verdict}")
    print()

    # ---- correctness: page widening reproduces pread byte for byte ----
    base = 8
    sample = [(i * 104729) % n_rows for i in range(512)]
    ref = reference_rows(path, sample, base)
    fd = os.open(str(path), os.O_RDONLY)
    pages: dict[int, bytes] = {}
    offsets = [base + r * ROW_BYTES for r in sample]
    try:
        for off in offsets:
            page, _, nbytes = page_for(off)
            if page not in pages:
                os.lseek(fd, page, os.SEEK_SET)
                pages[page] = os.read(fd, nbytes)
    finally:
        os.close(fd)
    got = extract_from_pages(pages, offsets)
    print(f"-- 2. page-widening row extraction --")
    print(f"  reference vs extracted : {'IDENTICAL' if got == ref else 'MISMATCH'}"
          f"  ({len(ref)} B)")
    print()

    print("-- 3. threaded reads, sync ReadFile --")
    per_thread = max(200, n_rows // 8)
    for label, no_buffer in (("NO_BUFFERING (device)", True),
                             ("buffered (page cache)", False)):
        for n_threads in (1, 4, 8, 16, 32):
            lat, wall, rows_read, bytes_read = sync_read(
                path, n_threads, per_thread, base, no_buffer)
            iops = rows_read / wall
            mibs = bytes_read / 2**20 / wall
            print(f"  {label:26s} threads {n_threads:2d}: "
                  f"{iops:7.0f} rows/s  {mibs:6.1f} MiB/s  "
                  f"p50 {pct(lat, 0.5):6.2f} ms  p90 {pct(lat, 0.9):6.2f} ms  "
                  f"p99 {pct(lat, 0.99):6.2f} ms")
    print()

    print(f"-- 4. IOCP queue, depth {args.depth}, {args.batches} batches "
          f"(the shape of rows_read's AIO batch) --")
    total_rows, total_bytes, wall, done_lat = iocp_read(
        path, args.depth, args.batches, base)
    print(f"  {total_rows} rows in {wall:.2f} s -> {total_rows/wall:7.0f} rows/s, "
          f"{total_bytes/2**20/wall:6.1f} MiB/s")
    print(f"  completion latency p50 {pct(done_lat, 0.5):.2f} ms, "
          f"p90 {pct(done_lat, 0.9):.2f} ms, max {max(done_lat):.2f} ms")
    print()

    print("-- 5. what a decode step would pay --")
    for batch in (1, 4, 8, 16):
        need = batch * 16  # 16 n-gram heads per token, one PLE layer
        rows_per_s = total_rows / wall if total_rows else 1e9
        print(f"  decode batch {batch:2d} -> {need:4d} uncached rows, "
              f"~{need/rows_per_s*1e3:6.2f} ms of device time at the measured rate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
