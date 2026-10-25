#!/usr/bin/env python3
"""Compare the safetensors headers of the two PLE shards.

If the tensor name, dtype, shape and data length agree, the payload is the
same table and the Windows copy can hardlink the existing file instead of
copying 95 GiB again.
"""
import json
import struct

OURS = ("/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound/"
        "model-00016-of-00017.safetensors")
THEIRS = ("/mnt/d/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP/"
          "model-00001-of-00011.safetensors")


def head(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
        import os
        size = os.path.getsize(path)
    meta = hdr.pop("__metadata__", None)
    return n, hdr, meta, size


for label, path in (("ours", OURS), ("theirs", THEIRS)):
    n, hdr, meta, size = head(path)
    print(f"=== {label}: {path.split('/')[-1]}")
    print(f"    header json {n} B, file {size} B, data {size - 8 - n} B, meta {meta}")
    for k, v in hdr.items():
        print(f"    tensor: {k!r}")
        print(f"      dtype={v['dtype']} shape={v['shape']} offsets={v['data_offsets']}")
