"""Dump the same sampled pages three ways, then compare.

The unbuffered IOCP path reports ~190k reads/s but its bytes did not match an
independent copy. Either the reads return the wrong bytes, or the two copies
differ in the data region. Three dumps, three-way comparison:

  python _dev/probe/_ple_page_compare.py dump --mode win  --out a.bin
  python _dev/probe/_ple_page_compare.py dump --mode wsl  --out b.bin
  python _dev/probe/_ple_page_compare.py dump --mode iocp --out c.bin
  python _dev/probe/_ple_page_compare.py compare a.bin b.bin c.bin
"""

import argparse
import ctypes
import hashlib
import json
import os
import random
import struct

BS = chr(92)
SHIM = r"D:\code\vllm-windows\_dev\out\nbshim\nbshim.dll"
WIN_ROOT = r"D:\models"
WSL_ROOT = BS * 2 + "wsl.localhost" + BS + "Ubuntu" + BS + "home" + BS \
    + "hong" + BS + "models"
SUBDIR = "Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
GENERIC_READ = 0x80000000
SHARE_RW = 0x00000006
NO_BUFFERING = 0x40000000
RANDOM_ACCESS = 0x10000000
MEM_COMMIT_RESERVE = 0x00003000
PAGE_READWRITE = 0x04
HEADER = 8
ROW = 320
PAGE = 4096

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
                             ctypes.c_uint]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
k32.VirtualFree.restype = ctypes.c_int


def pages_for(path, count, seed=99):
    """Sample pages that cover real ngram rows, via the header's own ranges."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8)
        meta = json.loads(f.read(n).decode("utf-8"))
    base = 8 + n
    rng = random.Random(seed)
    out = []
    keys = sorted(k for k in meta if k != "__metadata__")
    for k in keys:
        lo, hi = meta[k]["data_offsets"]
        if hi - lo < 2 * PAGE:
            continue
        for _ in range(max(1, count // min(8, len(keys)))):
            off = base + lo + rng.randrange(0, hi - lo - PAGE)
            out.append((off & ~(PAGE - 1)))
    out = out[:count]
    return sorted(set(out))


def dump_plain(path, offs, out):
    with open(path, "rb") as f:
        blob = bytearray()
        for off in offs:
            f.seek(off)
            blob += f.read(PAGE)
    with open(out, "wb") as f:
        f.write(blob)
    print(f"  {len(offs)} pages via buffered read -> {out} "
          f"({len(blob)} B)")


def dump_iocp(path, offs, out, depth=8):
    shim = ctypes.CDLL(SHIM)
    shim.nb_open.restype = ctypes.c_void_p
    shim.nb_open.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                             ctypes.c_uint]
    shim.nb_close.argtypes = [ctypes.c_void_p]
    shim.nb_last_error.restype = ctypes.c_uint
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

    h = shim.nb_open(path, GENERIC_READ, SHARE_RW, NO_BUFFERING | RANDOM_ACCESS)
    if not h:
        print(f"  open failed err={shim.nb_last_error()}")
        return
    iocp = shim.nb_iocp_new(depth)
    if not iocp:
        print(f"  iocp failed err={shim.nb_last_error()}")
        shim.nb_close(h)
        return
    if not shim.nb_iocp_bind(iocp, h, 1):
        print(f"  bind failed err={shim.nb_last_error()}")
        shim.nb_iocp_close(iocp)
        shim.nb_close(h)
        return
    pool = k32.VirtualAlloc(None, depth * PAGE, MEM_COMMIT_RESERVE,
                            PAGE_READWRITE)
    slots = [pool + i * PAGE for i in range(depth)]
    got = {off: None for off in offs}
    pend = {}
    # a slot may only be reused once its previous read has been harvested, so
    # submit from a free list rather than cycling slots by index
    free = list(range(depth))
    queued = 0
    done = 0
    for off in offs:
        if not free:
            while done < queued:
                key = ctypes.c_uint(0)
                nb = ctypes.c_uint(0)
                tok = ctypes.c_void_p(0)
                r = shim.nb_iocp_wait(iocp, 5000, ctypes.byref(key),
                                      ctypes.byref(nb), ctypes.byref(tok))
                if r != 1:
                    print(f"  wait r={r} err={shim.nb_last_error()}")
                    break
                slot, off_done = pend.pop(key.value)
                got[off_done] = ctypes.string_at(slots[slot], nb.value)
                shim.nb_iocp_release(tok)
                free.append(slot)
                done += 1
        slot = free.pop()
        if shim.nb_iocp_submit(iocp, h, off, PAGE, slots[slot], queued):
            pend[queued] = (slot, off)
            queued += 1
        else:
            print(f"  submit failed err={shim.nb_last_error()}")
            free.append(slot)
            break
    while done < queued:
        key = ctypes.c_uint(0)
        nb = ctypes.c_uint(0)
        tok = ctypes.c_void_p(0)
        r = shim.nb_iocp_wait(iocp, 5000, ctypes.byref(key), ctypes.byref(nb),
                              ctypes.byref(tok))
        if r == 1:
            slot, off_done = pend.pop(key.value)
            got[off_done] = ctypes.string_at(slots[slot], nb.value)
            shim.nb_iocp_release(tok)
            free.append(slot)
            done += 1
        else:
            print(f"  wait r={r} err={shim.nb_last_error()}")
            break
    k32.VirtualFree(pool, 0, 0x8000)
    shim.nb_iocp_close(iocp)
    shim.nb_close(h)
    blob = bytearray()
    missing = 0
    for off in offs:
        b = got.get(off)
        if b is None:
            missing += 1
            b = b"\x00" * PAGE
        blob += b
    with open(out, "wb") as f:
        f.write(blob)
    print(f"  {len(offs)} pages via unbuffered IOCP -> {out} "
          f"({len(blob)} B), missing {missing}")


def compare(paths):
    blobs = []
    for p in paths:
        with open(p, "rb") as f:
            blobs.append(f.read())
    names = [os.path.basename(p) for p in paths]
    print(f"  sizes: {[len(b) for b in blobs]}")
    if len(set(len(b) for b in blobs)) > 1:
        print("  sizes differ; cannot compare page by page")
        return
    n = len(blobs[0]) // PAGE
    print(f"  {n} pages each")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = sum(1 for pg in range(n)
                       if blobs[i][pg*PAGE:(pg+1)*PAGE]
                       != blobs[j][pg*PAGE:(pg+1)*PAGE])
            print(f"  {names[i]} vs {names[j]}: {diff}/{n} pages differ")
    for i, name in enumerate(names):
        print(f"  {name} sha256 {hashlib.sha256(blobs[i]).hexdigest()[:16]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "compare"])
    ap.add_argument("--way", default="win",
                    choices=["win", "wsl", "iocp"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="model-00003-of-00011.safetensors")
    ap.add_argument("--pages", type=int, default=64)
    ap.add_argument("paths", nargs="*", default=[])
    args = ap.parse_args()

    if args.mode == "compare":
        print(f"compare: {[os.path.basename(p) for p in args.paths]}")
        compare(args.paths)
        return 0

    win = os.path.join(WIN_ROOT, SUBDIR, args.name)
    wsl = os.path.join(WSL_ROOT, SUBDIR, args.name)
    src = wsl if args.way == "wsl" else win
    offs = pages_for(src, args.pages)
    print(f"file {args.name}: {len(offs)} sampled pages, "
          f"first {offs[0]}, last {offs[-1]}")
    out = args.out or f"C:\\Users\\hong\\AppData\\Local\\Temp\\psprobe" \
        f"\\dump_{args.way}.bin"
    if args.way == "iocp":
        dump_iocp(win, offs, out)
    else:
        dump_plain(src, offs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
