"""Check and repair the damage the unbuffered-IO C test did to shard 1.

_nbtest.c opened the PLE shard with GENERIC_WRITE and wrote an uninitialized
4096-byte buffer at offset 4096. In a safetensors file that region is part of
the JSON header, so the header is now partly zeroed.

  python _dev/probe/_ple_shard1_header_check.py check
  python _dev/probe/_ple_shard1_header_check.py repair --source <pristine file>

Repair writes in place so hardlinks (the PLE-free view) keep pointing at the
same file object.

"""

import argparse
import json
import os
import struct
import sys

DAMAGE_START = 4096
DAMAGE_LEN = 4096

TARGET = (
    r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
    r"\model-00001-of-00011.safetensors"
)


def header_of(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return None, "file shorter than 8 bytes"
        n = struct.unpack("<Q", raw)[0]
        f.seek(8)
        blob = f.read(n)
        return n, blob


def describe(path, label):
    n, blob = header_of(path)
    if blob is None:
        print(f"{label}: {blob}")
        return None
    print(f"{label}: header length {n} B")
    if isinstance(blob, bytes):
        zeros = blob.count(0)
        print(f"  bytes 8..{8+n}: {zeros} zero bytes "
              f"({100.0*zeros/len(blob):.2f}% of header)")
        seg = blob[DAMAGE_START-8:DAMAGE_START-8+DAMAGE_LEN]
        print(f"  the damaged window (file offset {DAMAGE_START}, {DAMAGE_LEN} B): "
              f"{seg.count(0)} zeros")
        try:
            meta = json.loads(blob.decode("utf-8"))
            names = [k for k in meta if k != "__metadata__"]
            print(f"  header parses: {len(names)} tensors")
            return meta
        except Exception as e:
            print(f"  header does NOT parse: {type(e).__name__}: "
                  f"{str(e)[:120]}")
    return None


def repair(target, source, start, length):
    with open(source, "rb") as f:
        f.seek(start)
        good = f.read(length)
    if len(good) != length:
        print(f"source gave only {len(good)} B")
        return 1
    with open(target, "r+b") as f:
        f.seek(start)
        f.write(good)
        f.flush()
        os.fsync(f.fileno())
    print(f"wrote {length} B at offset {start} from {source}")
    with open(target, "rb") as f:
        f.seek(start)
        back = f.read(length)
    print(f"readback identical: {back == good}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "repair"])
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--source")
    ap.add_argument("--start", type=int, default=DAMAGE_START)
    ap.add_argument("--length", type=int, default=DAMAGE_LEN)
    args = ap.parse_args()

    print(f"target: {args.target} ({os.path.getsize(args.target)/2**30:.2f} GiB)")
    meta = describe(args.target, "target")
    if args.source:
        describe(args.source, "source")
    if args.mode == "check":
        print()
        print("verdict:", "header OK" if meta else "header BROKEN")
        return 0
    if not args.source:
        print("repair needs --source")
        return 2
    return repair(args.target, args.source, args.start, args.length)


if __name__ == "__main__":
    raise SystemExit(main())
