# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
import tempfile
import time
from pathlib import Path

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
    lib.rows_depth.restype = ctypes.c_int
    lib.rows_depth.argtypes = [ctypes.c_void_p]
    lib.rows_bind.restype = ctypes.c_int
    lib.rows_bind.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    lib.rows_read.restype = ctypes.c_int
    lib.rows_read.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    lib.rows_fault.restype = ctypes.c_int
    lib.rows_fault.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_int),
    ]
    return lib


def synthetic_check(dll, depth):
    """Check queue depth, row order/EOF bytes and failure cleanup on a tiny file."""
    lib = load(dll)
    payload = bytes(range(256)) * 65 + bytes(range(64))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.bin"
        path.write_bytes(payload)
        reader = lib.rows_open(depth)
        if not reader:
            raise RuntimeError(f"rows_open failed: {ctypes.get_errno()}")
        try:
            assert lib.rows_depth(reader) == depth
            assert lib.rows_bind(reader, 7, str(path)) == 0
            count = 2 * depth + 13
            fds = (ctypes.c_int * count)(*([7] * count))
            for width in (1, ROW, 4096):
                candidates = (4095, 0, len(payload) - width, 4096, 1)
                offsets = [candidates[i % len(candidates)] for i in range(count)]
                offs = (ctypes.c_uint64 * count)(*offsets)
                out = (ctypes.c_ubyte * (count * width))()
                assert lib.rows_read(reader, fds, offs, count, width, out) == 0
                assert bytes(out) == b"".join(
                    payload[offset : offset + width] for offset in offsets
                ), f"row byte/order mismatch at width {width}"
            fds[depth // 2] = -1
            assert lib.rows_read(reader, fds, offs, count, width, out) < 0
            assert lib.rows_read(reader, fds, offs, count, width, out) < 0
        finally:
            lib.rows_close(reader)
        reader = lib.rows_open(depth)
        if not reader:
            raise RuntimeError(f"rows_open failed: {ctypes.get_errno()}")
        try:
            assert lib.rows_bind(reader, 7, str(path)) == 0
            fds[depth // 2] = 7
            offs[depth // 2] = len(payload) - 1
            assert lib.rows_read(reader, fds, offs, count, width, out) < 0
        finally:
            lib.rows_close(reader)
    print(f"Synthetic correctness passed: depth {depth}, {count} rows, EOF/cancel")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", default=DLL)
    ap.add_argument(
        "--synthetic", action="store_true", help="CPU-only check; no model required"
    )
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--depth", type=int, default=256)
    ap.add_argument(
        "--check",
        type=int,
        default=64,
        help="rows whose bytes are compared with a buffered read",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="time repeated reads of the same offsets before validation",
    )
    args = ap.parse_args()
    if args.rows < 1 or args.repeat < 1 or args.check < 0:
        ap.error("rows and repeat must be positive; check must be nonnegative")
    if not os.path.exists(args.dll):
        print(f"missing {args.dll}; run _dev/bin/_ple_ssd_io_win_build.ps1")
        return 1

    if args.synthetic:
        return synthetic_check(args.dll, args.depth)

    with open(args.shard, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        f.seek(8)
        header = json.loads(f.read(hl).decode("utf-8"))
    data_start = 8 + hl
    keys = sorted(k for k in header if k != "__metadata__")
    ple_keys = [k for k in keys if ".ngram_embedding.shard_" in k]
    if not ple_keys:
        print("no PLE tensors found")
        return 1
    ranges = []
    for key in ple_keys:
        lo, hi = header[key]["data_offsets"]
        ranges.append((key, lo, (hi - lo) // ROW))
    last_key, last_lo, last_rows = max(ranges, key=lambda item: item[1] + item[2] * ROW)
    total_rows = sum(item[2] for item in ranges)
    print(f"shard: {args.shard}")
    print(
        f"  header {hl} B, {len(keys)} tensors; PLE tensors "
        f"{len(ranges)}, last tensor {last_key}"
    )
    print(f"  PLE table spans {total_rows:,} rows across the shard")
    print(f"  last tensor spans {last_rows} rows to end of file")

    import random

    rng = random.Random(args.seed)
    rows = []
    offsets = []
    for _ in range(args.rows):
        _, lo, rows_in_tensor = ranges[rng.randrange(len(ranges))]
        row = rng.randrange(rows_in_tensor)
        rows.append(row)
        offsets.append(data_start + lo + row * ROW)
    # Include the final row so the EOF fallback remains covered.
    rows.append(last_rows - 1)
    offsets.append(data_start + last_lo + (last_rows - 1) * ROW)
    order = sorted(range(len(rows)), key=offsets.__getitem__)
    rows = [rows[i] for i in order]
    offsets = [offsets[i] for i in order]
    pages = {off // 4096 for off in offsets}
    pages.update((off + ROW - 1) // 4096 for off in offsets)
    print(
        f"  seed {args.seed}, {len(pages):,} distinct 4 KiB pages; "
        f"repeat reads do not establish cold-cache throughput"
    )

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
        for iteration in range(args.repeat):
            t0 = time.perf_counter()
            status = lib.rows_read(
                reader,
                fds.ctypes.data,
                offs.ctypes.data,
                len(rows),
                ROW,
                out.ctypes.data,
            )
            dt = time.perf_counter() - t0
            print(
                f"pass {iteration + 1}: rows_read({len(rows)}) -> {status} "
                f"in {dt:.6f} s ({len(rows) / dt:,.0f} rows/s, "
                f"{len(rows) * ROW / 2**20 / dt:,.1f} MiB/s of row data)",
                flush=True,
            )
            if status:
                break
        if status:
            j = ctypes.c_uint(0)
            base = ctypes.c_uint(0)
            exact = ctypes.c_int(0)
            lib.rows_fault(
                reader, ctypes.byref(j), ctypes.byref(base), ctypes.byref(exact)
            )
            row_at = rows[base.value]
            off_at = offsets[base.value]
            print(
                f"  errno={ctypes.get_errno()} ordinal={base.value} "
                f"slot={j.value} "
                f"{'exact' if exact.value else 'page-shaped'} "
                f"at row {row_at} offset {off_at} "
                f"(page {off_at & ~4095}, delta {off_at & 4095}, "
                f"file size {os.path.getsize(args.shard)})"
            )
            lib.rows_close(reader)
            return 1

        mismatch = 0
        checked = 0
        for i in sorted(
            set(range(min(args.check, len(rows)))) | {len(rows) - 1, len(rows) // 2}
        ):
            row = rows[i]
            f.seek(offsets[i])
            want = f.read(ROW)
            got = out[i].tobytes()
            checked += 1
            if want != got:
                mismatch += 1
                if mismatch == 1:
                    print(
                        f"  MISMATCH row {row} at offset {offsets[i]}: "
                        f"{sum(1 for a, b in zip(want, got) if a != b)}/{ROW} "
                        "bytes differ"
                    )
                    print(f"    helper: {got[:16]!r}")
                    print(f"    file  : {want[:16]!r}")
        print(
            f"  compared {checked} rows against buffered reads: {mismatch} mismatched"
        )
        zero = sum(1 for i in range(len(rows)) if not out[i].any())
        print(f"  all-zero rows in the batch: {zero} of {len(rows)}")
    lib.rows_close(reader)
    return int(mismatch != 0)


if __name__ == "__main__":
    raise SystemExit(main())
