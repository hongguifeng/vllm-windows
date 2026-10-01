"""Exercise the Windows PLE/SSD helper directly against the real PLE shard.

Checks row bytes against buffered reads, includes the final row of the file so
the end-of-file path is covered, and reports how fast batches go.

  python _dev/probe/_ple_ssd_io_win_test.py [--rows 4096] [--depth 256]
"""

import argparse
import ctypes
import json
import os
import struct
import time

DLL = r"D:\code\vllm-windows\_dev\out\ple_ssd_io\ple_ssd_io_win.dll"
SHARD = (
    r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
    r"\model-00001-of-00011.safetensors"
)
ROW = 320


def load(path):
    lib = ctypes.CDLL(path, use_errno=True)
    lib.rows_open.restype = ctypes.c_void_p
    lib.rows_open.argtypes = [ctypes.c_uint]
    lib.rows_close.argtypes = [ctypes.c_void_p]
    lib.rows_bind.restype = ctypes.c_int
    lib.rows_bind.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    lib.rows_read.restype = ctypes.c_int
    lib.rows_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    lib.rows_fault.restype = ctypes.c_int
    lib.rows_fault.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
                               ctypes.POINTER(ctypes.c_uint),
                               ctypes.POINTER(ctypes.c_int)]
    return lib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", default=DLL)
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--depth", type=int, default=256)
    ap.add_argument("--check", type=int, default=64,
                    help="rows whose bytes are compared with a buffered read")
    args = ap.parse_args()
    if not os.path.exists(args.dll):
        print(f"missing {args.dll}; run _dev/bin/_ple_ssd_io_win_build.ps1")
        return 1

    with open(args.shard, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        f.seek(8)
        header = json.loads(f.read(hl).decode("utf-8"))
    data_start = 8 + hl
    keys = sorted(k for k in header if k != "__metadata__")
    last_key = keys[-1]
    print(f"shard: {args.shard}")
    print(f"  header {hl} B, {len(keys)} tensors; last tensor {last_key}")
    lo, hi = header[last_key]["data_offsets"]
    file_rows = (os.path.getsize(args.shard) - data_start - lo) // ROW
    print(f"  last tensor spans {file_rows} rows to end of file")

    import random
    rng = random.Random(7)
    rows = sorted(rng.randrange(0, file_rows) for _ in range(args.rows))
    rows[-1] = file_rows - 1          # the row whose page crosses EOF
    rows.insert(0, 0)
    rows = sorted(set(rows))
    offsets = [data_start + lo + r * ROW for r in rows]

    lib = load(args.dll)
    reader = lib.rows_open(args.depth)
    if not reader:
        print(f"rows_open -> NULL errno={ctypes.get_errno()}")
        return 1
    with open(args.shard, "rb") as f:
        fd = f.fileno()
        rc = lib.rows_bind(reader, fd, args.shard)
        print(f"rows_bind fd={fd} -> {rc}")
        if rc:
            lib.rows_close(reader)
            return 1

        import numpy as np
        fds = np.asarray([fd] * len(rows), dtype=np.int32)
        offs = np.asarray(offsets, dtype=np.uint64)
        out = np.empty((len(rows), ROW), dtype=np.uint8)
        t0 = time.perf_counter()
        status = lib.rows_read(reader, fds.ctypes.data, offs.ctypes.data,
                               len(rows), ROW, out.ctypes.data)
        dt = time.perf_counter() - t0
        print(f"rows_read({len(rows)}) -> {status} in {dt:.3f} s "
              f"({len(rows)/dt:,.0f} rows/s, "
              f"{len(rows)*ROW/2**20/dt:,.1f} MiB/s of row data)")
        if status:
            j = ctypes.c_uint(0)
            base = ctypes.c_uint(0)
            exact = ctypes.c_int(0)
            lib.rows_fault(reader, ctypes.byref(j), ctypes.byref(base),
                           ctypes.byref(exact))
            row_at = rows[base.value + j.value]
            off_at = offsets[base.value + j.value]
            print(f"  errno={ctypes.get_errno()} batch base={base.value} "
                  f"j={j.value} "
                  f"{'exact buffered' if exact.value else 'page unbuffered'} "
                  f"at row {row_at} offset {off_at} "
                  f"(page {off_at & ~4095}, delta {off_at & 4095}, "
                  f"file size {os.path.getsize(args.shard)})")
            lib.rows_close(reader)
            return 1

        mismatch = 0
        checked = 0
        for i in list(range(args.check)) + [-1, len(rows) // 2]:
            row = rows[i]
            f.seek(offsets[i])
            want = f.read(ROW)
            got = out[i].tobytes()
            checked += 1
            if want != got:
                mismatch += 1
                if mismatch == 1:
                    print(f"  MISMATCH row {row} at offset {offsets[i]}: "
                          f"{sum(1 for a, b in zip(want, got) if a != b)}/{ROW} "
                          "bytes differ")
                    print(f"    helper: {got[:16]!r}")
                    print(f"    file  : {want[:16]!r}")
        print(f"  compared {checked} rows against buffered reads: "
              f"{mismatch} mismatched")
        zero = sum(1 for i in range(len(rows)) if not out[i].any())
        print(f"  all-zero rows in the batch: {zero} of {len(rows)}")
    lib.rows_close(reader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
