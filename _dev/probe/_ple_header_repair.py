"""Restore the safetensors header of a damaged shard from a pristine copy.

The header is the region [0, 8 + header_len). Data regions are untouched, so
rewriting only the header is enough. The write goes through the shim so it can
use a full-share handle even when another process holds the file.

In-place writes keep hardlinks valid: the PLE-free view points at the same file
object, so repairing the source repairs the view too.

  python _dev/probe/_ple_header_repair.py --target <file> --source <file>
  python _dev/probe/_ple_header_repair.py --target <file> --source <file> --check-only
"""

import argparse
import ctypes
import json
import os
import struct

SHIM = r"D:\code\vllm-windows\_dev\out\nbshim\nbshim.dll"
GENERIC_ALL = 0xC0000000
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
SHARE_RWD = 0x00000007
OPEN_EXISTING = 3
RANDOM_ACCESS = 0x10000000
MEM_COMMIT_RESERVE = 0x00003000
PAGE_READWRITE = 0x04

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
                             ctypes.c_uint]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
k32.VirtualFree.restype = ctypes.c_int


def shim_load():
    shim = ctypes.CDLL(SHIM)
    shim.nb_open.restype = ctypes.c_void_p
    shim.nb_open.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                             ctypes.c_uint]
    shim.nb_close.argtypes = [ctypes.c_void_p]
    shim.nb_size.restype = ctypes.c_longlong
    shim.nb_size.argtypes = [ctypes.c_void_p]
    shim.nb_read.restype = ctypes.c_longlong
    shim.nb_read.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_uint,
                             ctypes.c_void_p]
    shim.nb_write.restype = ctypes.c_longlong
    shim.nb_write.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_uint,
                              ctypes.c_void_p]
    shim.nb_flush.argtypes = [ctypes.c_void_p]
    shim.nb_flush.restype = ctypes.c_int
    shim.nb_last_error.restype = ctypes.c_uint
    return shim


BS = chr(92)
WSL_ROOT = BS * 2 + "wsl.localhost" + BS + "Ubuntu" + BS + "home" + BS \
    + "hong" + BS + "models"


def header_len(path):
    with open(path, "rb") as f:
        raw = f.read(8)
    if len(raw) < 8:
        raise ValueError(f"{path} shorter than 8 bytes")
    return struct.unpack("<Q", raw)[0]


def summary(path, label):
    n = header_len(path)
    with open(path, "rb") as f:
        f.seek(8)
        blob = f.read(n)
    try:
        meta = json.loads(blob.decode("utf-8"))
        names = sorted(k for k in meta if k != "__metadata__")
        print(f"  {label}: header {n} B parses, {len(names)} tensors")
        odd = [k for k in names if ".lal." in k or "pleg" in k or ".ple.ple" in k]
        if odd:
            print(f"    suspicious names: {odd[:4]}")
        return blob, meta
    except Exception as e:
        print(f"  {label}: header {n} B does NOT parse: "
              f"{type(e).__name__}: {str(e)[:100]}")
        return blob, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.source:
        d, leaf = os.path.split(args.target)
        args.source = os.path.join(WSL_ROOT, os.path.basename(d), leaf)
        print(f"source defaulted to the WSL copy: {args.source}")

    tn = header_len(args.target)
    sn = header_len(args.source)
    print(f"target {args.target}\n  header length {tn} "
          f"({os.path.getsize(args.target)} B total)")
    print(f"source {args.source}\n  header length {sn} "
          f"({os.path.getsize(args.source)} B total)")
    summary(args.target, "target before")
    summary(args.source, "source")
    if tn != sn:
        print("header lengths differ; refusing to write")
        return 2
    if args.check_only:
        with open(args.source, "rb") as f:
            f.seek(8)
            good = f.read(sn)
        with open(args.target, "rb") as f:
            f.seek(8)
            cur = f.read(tn)
        print(f"header bytes identical: {good == cur}")
        return 0

    if not os.path.exists(SHIM):
        print(f"missing {SHIM}; run _dev/bin/_nbshim_build.ps1")
        return 1
    shim = shim_load()
    with open(args.source, "rb") as f:
        f.seek(0)
        blob = f.read(8 + sn)
    print(f"read {len(blob)} B of header+length from source")

    length = 8 + sn
    buf = k32.VirtualAlloc(None, length, MEM_COMMIT_RESERVE, PAGE_READWRITE)
    ctypes.memmove(buf, blob, length)

    h = shim.nb_open(args.target, GENERIC_READ | GENERIC_WRITE, SHARE_RWD,
                     RANDOM_ACCESS)
    if not h:
        print(f"open for write failed err={shim.nb_last_error()}")
        k32.VirtualFree(buf, 0, 0x8000)
        return 1
    put = shim.nb_write(h, 0, length, buf)
    err = shim.nb_last_error() if put < 0 else 0
    flushed = shim.nb_flush(h)
    shim.nb_close(h)
    k32.VirtualFree(buf, 0, 0x8000)
    print(f"wrote {put} B at offset 0 (err={err}, flush={flushed})")
    if put != length:
        print("write incomplete")
        return 1
    summary(args.target, "target after")
    with open(args.target, "rb") as f:
        f.seek(0)
        back = f.read(8 + sn)
    print(f"target header now identical to source: {back == blob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
