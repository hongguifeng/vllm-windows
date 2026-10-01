"""Are the D: copy and the WSL copy the same bytes?

If they are, then every IOCP unbuffered page that mismatched is a read defect.
If they are not, the comparison was meaningless and the mismatch is data, not I/O.

  python _dev/probe/_ple_copy_compare.py [--shard 1]
"""

import argparse
import hashlib
import json
import os
import struct

WIN_ROOT = r"D:\models"
WSL_ROOT = "\\\\wsl.localhost\\Ubuntu\\home\\hong\\models"
SUBDIR = "Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"


def header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        n = struct.unpack("<Q", raw)[0]
        f.seek(8)
        blob = f.read(n)
        meta = json.loads(blob.decode("utf-8"))
    return n, blob, meta


def sha(data):
    return hashlib.sha256(data).hexdigest()[:16]


def region(path, start, length):
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(length)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default=SUBDIR)
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    for k in range(args.shards):
        idx = args.shard + k
        name = f"model-{idx:05d}-of-00011.safetensors"
        wpath = os.path.join(WIN_ROOT, args.subdir, name)
        spath = os.path.join(WSL_ROOT, args.subdir, name)
        print(f"--- {name} ---")
        for label, path in (("windows", wpath), ("wsl", spath)):
            if not os.path.exists(path):
                print(f"  {label}: missing {path}")
                continue
            size = os.path.getsize(path)
            n, blob, meta = header(path)
            names = sorted(k2 for k2 in meta if k2 != "__metadata__")
            print(f"  {label}: {path}")
            print(f"    size {size} B ({size/2**30:.2f} GiB)  header {n} B  "
                  f"sha {sha(blob)}  tensors {len(names)}")
            md = meta.get("__metadata__")
            if md:
                print(f"    metadata: {md}")
            if names:
                first = names[0]
                info = meta[first]
                print(f"    first tensor {first}: dtype={info['dtype']} "
                      f"shape={info['shape']} offsets={info['data_offsets']}")
        if not (os.path.exists(wpath) and os.path.exists(spath)):
            continue
        wn, wb, wmeta = header(wpath)
        sn, sb, smeta = header(spath)
        print(f"  header identical: {wb == sb} (lengths {wn} vs {sn})")
        if wb != sb:
            wnames = set(k2 for k2 in wmeta if k2 != "__metadata__")
            snames = set(k2 for k2 in smeta if k2 != "__metadata__")
            print(f"    names only on windows: "
                  f"{sorted(wnames - snames)[:5]}")
            print(f"    names only on wsl: {sorted(snames - wnames)[:5]}")
            common = wnames & snames
            differing = [k2 for k2 in sorted(common)
                         if wmeta[k2] != smeta[k2]]
            print(f"    tensors present in both: {len(common)}, "
                  f"with differing descriptors: {len(differing)}")
            for k2 in differing[:5]:
                print(f"      {k2}: win {wmeta[k2]} != wsl {smeta[k2]}")
        # compare the first tensor's data bytes
        try:
            info = wmeta[sorted(k2 for k2 in wmeta
                                if k2 != "__metadata__")[0]]
            lo, hi = info["data_offsets"]
            n = min(hi - lo, 1 << 20)
            a = region(wpath, 8 + wn + lo, n)
            b = region(spath, 8 + sn + lo, n)
            print(f"  first {n} B of the first tensor: identical={a == b}, "
                  f"win sha {sha(a)}, wsl sha {sha(b)}")
        except Exception as e:
            print(f"  data compare failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
