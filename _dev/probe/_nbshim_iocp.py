"""Throughput of the IOCP batch path, unbuffered vs buffered, on the PLE shard.

The synchronous NO_BUFFERING read is refused with ERROR_INVALID_PARAMETER inside
python.exe, but an overlapped one was accepted (ERROR_IO_PENDING). This asks the
question that matters for PLE: can we sustain random 4096 B reads through a
completion port, and how fast?

  python _dev/probe/_nbshim_iocp.py [--reads 20000] [--depth 32] [--buffered]
"""

import argparse
import ctypes
import os
import random
import statistics
import sys
import time

GENERIC_READ = 0x80000000
SHARE_RW = 0x00000006
NO_BUFFERING = 0x40000000
RANDOM_ACCESS = 0x10000000
SEQUENTIAL = 0x00000000
MEM_COMMIT_RESERVE = 0x00003000
PAGE_READWRITE = 0x04
MEM_RELEASE = 0x00008000

SHIM = r"D:\code\vllm-windows\_dev\out\nbshim\nbshim.dll"
TARGET = (
    r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
    r"\model-00001-of-00011.safetensors"
)
HEADER = 8
ROW = 320
PAGE = 4096

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
                             ctypes.c_uint]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]


def load_shim():
    shim = ctypes.CDLL(SHIM)
    shim.nb_open.restype = ctypes.c_void_p
    shim.nb_open.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                             ctypes.c_uint]
    shim.nb_close.argtypes = [ctypes.c_void_p]
    shim.nb_size.restype = ctypes.c_longlong
    shim.nb_size.argtypes = [ctypes.c_void_p]
    shim.nb_last_error.restype = ctypes.c_uint
    shim.nb_read.restype = ctypes.c_longlong
    shim.nb_read.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_uint,
                             ctypes.c_void_p]
    shim.nb_iocp_new.restype = ctypes.c_void_p
    shim.nb_iocp_new.argtypes = [ctypes.c_uint]
    shim.nb_iocp_bind.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    shim.nb_iocp_bind.restype = ctypes.c_int
    shim.nb_iocp_submit.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_longlong, ctypes.c_uint,
                                    ctypes.c_void_p, ctypes.c_uint]
    shim.nb_iocp_submit.restype = ctypes.c_int
    shim.nb_iocp_wait.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                  ctypes.POINTER(ctypes.c_uint),
                                  ctypes.POINTER(ctypes.c_uint),
                                  ctypes.POINTER(ctypes.c_void_p)]
    shim.nb_iocp_wait.restype = ctypes.c_int
    shim.nb_iocp_release.argtypes = [ctypes.c_void_p]
    shim.nb_iocp_close.argtypes = [ctypes.c_void_p]
    return shim


def percentiles(xs):
    if not xs:
        return "n/a"
    q = statistics.quantiles(xs, n=100, method="inclusive")
    return (f"min {min(xs)*1e3:.1f} p50 {q[49]*1e3:.1f} "
            f"p90 {q[89]*1e3:.1f} p99 {q[98]*1e3:.1f} "
            f"max {max(xs)*1e3:.1f} ms")


def run(path, total, depth, use_buffering, page, verify=64, verify_from=None):
    shim = load_shim()
    flags = (RANDOM_ACCESS | (0 if use_buffering else NO_BUFFERING))
    h = shim.nb_open(path, GENERIC_READ, SHARE_RW, flags)
    if not h:
        print(f"  open failed err={shim.nb_last_error()}")
        return
    size = shim.nb_size(h)
    n_rows = (size - HEADER) // ROW
    iocp = shim.nb_iocp_new(depth)
    if not iocp:
        print(f"  iocp create failed err={shim.nb_last_error()}")
        shim.nb_close(h)
        return
    if not shim.nb_iocp_bind(iocp, h, 1):
        print(f"  bind failed err={shim.nb_last_error()}")
        shim.nb_iocp_close(iocp)
        shim.nb_close(h)
        return

    region = depth * page
    pool = k32.VirtualAlloc(None, region, MEM_COMMIT_RESERVE, PAGE_READWRITE)
    slots = [pool + i * page for i in range(depth)]
    verify_slot = 1
    # a slot may only be reused once its own read has been harvested; cycling
    # slots by submission index lets an outstanding read be overwritten
    free = list(range(depth))
    outstanding = 0
    print(f"  pool 0x{pool:x}, {depth} slots of {page} B "
          f"(slot mod 4096 = {slots[1] % 4096})")

    rng = random.Random(1234)
    order = [rng.randrange(n_rows) for _ in range(total)]
    next_req = 0
    inflight = {}
    latencies = []
    short = 0
    failed = 0
    bytes_ok = 0

    # verify against an independent copy of the same weights (the WSL ext4 copy
    # reached through \\wsl.localhost). A second handle on the same NTFS file is
    # not usable while an unbuffered handle is live.
    vf = None
    if verify_from and os.path.exists(verify_from):
        vf = open(verify_from, "rb")
    elif verify:
        print(f"  no independent copy at {verify_from!r}; content unverified")
    verified = 0
    mismatch = 0
    verified_bytes = 0
    t_start = time.perf_counter()
    submitted = 0
    done = 0

    while done < total:
        while free and outstanding < depth and next_req < total:
            row = order[next_req]
            slot = free.pop()
            off = (HEADER + row * ROW) & ~(PAGE - 1)
            key = submitted
            if shim.nb_iocp_submit(iocp, h, off, page, slots[slot], key):
                inflight[key] = (time.perf_counter(), slot)
                submitted += 1
                outstanding += 1
                next_req += 1
            else:
                free.append(slot)
                outstanding -= 1
                e = shim.nb_last_error()
                failed += 1
                if failed == 1:
                    print(f"  first submit failure err={e} (stopping submits)")
                break
        key = ctypes.c_uint(0)
        nb = ctypes.c_uint(0)
        tok = ctypes.c_void_p(0)
        r = shim.nb_iocp_wait(iocp, 5000, ctypes.byref(key), ctypes.byref(nb),
                              ctypes.byref(tok))
        if r == 1:
            info = inflight.pop(key.value, None)
            if info:
                outstanding -= 1
                free.append(info[1])
                latencies.append(time.perf_counter() - info[0])
                if verified < verify and info[1] == verify_slot:
                    row = order[key.value]
                    page_off = (HEADER + row * ROW) & ~(PAGE - 1)
                    if vf is not None:
                        vf.seek(page_off)
                        expect = vf.read(page)
                        verified += 1
                        a = ctypes.string_at(slots[info[1]], page)
                        if len(expect) != page:
                            print(f"  independent read short: {len(expect)} B "
                                  f"at {page_off}")
                            mismatch += 1
                        elif a == expect:
                            verified_bytes += page
                        else:
                            mismatch += 1
                            if mismatch == 1:
                                diff = sum(1 for x, y in zip(a, expect)
                                           if x != y)
                                print(f"  MISMATCH page {page_off}: "
                                      f"{diff}/{page} bytes differ")
                                print(f"    unbuffered: {a[:24]!r}")
                                print(f"    buffered  : {expect[:24]!r}")
            if nb.value != page:
                short += 1
            else:
                bytes_ok += 1
            shim.nb_iocp_release(tok)
            done += 1
        elif r == -1:
            e = shim.nb_last_error()
            info = inflight.pop(key.value, None)
            print(f"  completion error {e} on key {key.value}")
            if info:
                outstanding -= 1
                free.append(info[1])
            shim.nb_iocp_release(tok)
            done += 1
        else:
            print(f"  wait timed out err={shim.nb_last_error()} "
                  f"(submitted {submitted}, done {done})")
            break

    dt = time.perf_counter() - t_start
    rate = done / dt if dt > 0 else 0.0
    print(f"  {'buffered' if use_buffering else 'NO_BUFFERING'} "
          f"depth={depth} page={page} reads={done}/{total} "
          f"in {dt:.2f} s -> {rate:,.0f} reads/s, "
          f"{rate*page/2**20:,.1f} MiB/s")
    print(f"  latency (submit to completion): {percentiles(latencies)}")
    print(f"  exact {page} B: {bytes_ok}; short: {short}; "
          f"submit failures: {failed}")
    print(f"  verified {verified} pages against buffered reads: "
          f"{verified_bytes} B identical, {mismatch} mismatched")
    if vf is not None:
        vf.close()
    k32.VirtualFree(pool, 0, MEM_RELEASE)
    shim.nb_iocp_close(iocp)
    shim.nb_close(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=TARGET)
    ap.add_argument("--reads", type=int, default=20000)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--buffered", action="store_true")
    ap.add_argument("--twice", action="store_true",
                    help="run the same offsets again to see cached behaviour")
    ap.add_argument("--verify", type=int, default=64)
    ap.add_argument("--verify-from", default=None,
                    help="path of an independent copy to compare bytes against")
    args = ap.parse_args()
    if not os.path.exists(SHIM):
        print(f"missing {SHIM} -- run _dev/bin/_nbshim_build.ps1")
        return 1
    print(f"file: {args.file} "
          f"({os.path.getsize(args.file)/2**30:.2f} GiB)")
    if not args.verify_from:
        d, leaf = os.path.split(args.file)
        root = "\\\\wsl.localhost\\Ubuntu\\home\\hong\\models"
        args.verify_from = os.path.join(root, os.path.basename(d), leaf)
    run(args.file, args.reads, args.depth, args.buffered, args.page,
        args.verify, args.verify_from)
    if args.twice:
        print()
        run(args.file, args.reads, args.depth, args.buffered, args.page,
            args.verify, args.verify_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
