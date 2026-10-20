"""Check that PLESSDTable serves only its rank's rows and zero-fills the rest.

CPU only. Reads a handful of real rows from the AutoRound checkpoint with two
disjoint owner ranges and compares them against a full-table read.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

MODEL = Path(r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP")

index = json.loads((MODEL / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]

# Discover the PLE table: contiguous ".shard_N.weight" keys of one prefix.
by_prefix: dict[str, list[str]] = {}
for key in weight_map:
    if ".shard_" in key and key.endswith(".weight"):
        prefix = key[: key.rindex(".shard_")]
        by_prefix.setdefault(prefix, []).append(key)
if len(by_prefix) != 1:
    print(f"expected one PLE prefix, found {sorted(by_prefix)}")
    sys.exit(1)
prefix, keys = next(iter(by_prefix.items()))
keys.sort(key=lambda k: int(k.rsplit("shard_", 1)[1].split(".")[0]))
print(f"prefix {prefix}")
print(f"shards {len(keys)}")

first_file = MODEL / weight_map[keys[0]]
with open(first_file, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))
shape = header[keys[0]]["shape"]
dim = shape[1]
dtype = header[keys[0]]["dtype"]
print(f"dtype {dtype} dim {dim} shard0 rows {shape[0]}")
if dtype != "BF16":
    print("not a BF16 table; the SSD path does not apply")
    sys.exit(1)

rows = 0
for key in keys:
    filename = MODEL / weight_map[key]
    with open(filename, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(header_len))[key]
    rows += meta["shape"][0]
print(f"rows {rows} row_bytes {dim * 2} table {rows * dim * 2 / 2**30:.2f} GiB")

from vllm.models.qwen4_exp.nvidia.ple_ssd import PLESSDTable  # noqa: E402

half = rows // 2
full = PLESSDTable(str(MODEL), prefix, rows, dim, 1, 0)
low = PLESSDTable(str(MODEL), prefix, rows, dim, 1, 0, owner=(0, half))
high = PLESSDTable(str(MODEL), prefix, rows, dim, 1, 0, owner=(half, rows))

ids = np.array(
    [0, 7, half - 1, half, half + 1, rows - 1, 12345678, rows - 7919], dtype=np.int64
)
ref = np.empty((ids.size, full.row_bytes), dtype=np.uint8)
full.read(ids, ref)

failures = []
for name, table, mine in (("low", low, ids < half), ("high", high, ids >= half)):
    out = np.empty((ids.size, table.row_bytes), dtype=np.uint8)
    table.read(ids, out)
    for i, row in enumerate(ids.tolist()):
        got = out[i].tobytes()
        want = ref[i].tobytes() if mine[i] else bytes(table.row_bytes)
        label = "owned" if mine[i] else "foreign"
        if got != want:
            failures.append(f"{name} row {row} ({label}) mismatch")
    print(f"{name}: owned {int(mine.sum())} foreign {int((~mine).sum())} "
          f"foreign_occ {table.foreign_occ}")

# A row is only served by its owner, so the summed halves reproduce the table.
out_low = np.empty((ids.size, low.row_bytes), dtype=np.uint8)
out_high = np.empty((ids.size, high.row_bytes), dtype=np.uint8)
low.read(ids, out_low)
high.read(ids, out_high)
summed = out_low.astype(np.uint16) + out_high.astype(np.uint16)
if not np.array_equal(summed, ref.astype(np.uint16)):
    failures.append("owner sum does not reproduce the full table")
else:
    print("owner sum reproduces the table")

full.close()
low.close()
high.close()

if failures:
    print(f"RESULT FAIL ({len(failures)})")
    for line in failures:
        print(f"  - {line}")
    sys.exit(1)
print("RESULT OK")
