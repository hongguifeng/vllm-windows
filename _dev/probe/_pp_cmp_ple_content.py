#!/usr/bin/env python3
"""Compare three PLE tensors by name between the two shard files.

Both files carry the same 128 tensors but serialise them in a different order,
so the block offsets differ. Hashing a tensor means seeking to that file's own
offset for that name.
"""
import hashlib
import json
import struct

FILES = {
    "ours": ("/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound/"
             "model-00016-of-00017.safetensors"),
    "theirs": ("/mnt/d/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP/"
               "model-00001-of-00011.safetensors"),
}
NAMES = ["0", "64", "127"]


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
    hdr.pop("__metadata__", None)
    base = 8 + n
    return {k.split(".shard_")[-1].split(".")[0]: (v["data_offsets"], base, path)
            for k, v in hdr.items()}


def digest(path, base, off):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(base + off[0])
        remaining = off[1] - off[0]
        while remaining:
            chunk = f.read(min(64 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


index = {label: load(path) for label, path in FILES.items()}
for name in NAMES:
    a_off, a_base, a_path = index["ours"][name]
    b_off, b_base, b_path = index["theirs"][name]
    ha = digest(a_path, a_base, a_off)
    hb = digest(b_path, b_base, b_off)
    size = a_off[1] - a_off[0]
    print(f"{name}: {size/2**20:.0f} MiB  "
          f"{'IDENTICAL' if ha == hb else 'DIFFERENT'}  {ha[:16]} / {hb[:16]}")
