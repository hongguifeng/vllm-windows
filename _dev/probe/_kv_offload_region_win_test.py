"""Windows check for the CPU KV-offload shared region.

Covers what a rank actually does: create the region, write through a worker
view, let a second handle see the same bytes, clean up, and start again over a
stale file left by an unclean exit.

  python _dev/probe/_kv_offload_region_win_test.py
"""

import os
import sys

import torch

from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

ENGINE = "win-test"
CHUNKS = 8
ROW = 4096
PAGE = 2048
TENSOR = 1024

fail = []


def make(rank: int | None = 0) -> SharedOffloadRegion:
    return SharedOffloadRegion(
        engine_id=ENGINE,
        num_chunks=CHUNKS,
        rank=rank,
        kv_bytes_per_chunk=ROW,
        cpu_page_size=PAGE,
    )


creator = make()
path = creator.mmap_path
print(f"path        {path}")
print(f"size        {os.path.getsize(path)} bytes (expected {CHUNKS * ROW})")
if os.path.getsize(path) != CHUNKS * ROW:
    fail.append("region file has the wrong size")

view = creator.create_next_worker_view(TENSOR)
view.fill_(7)

joiner = make(rank=1)
print(f"joiner creator flag: {joiner._creator} (expected False)")
if joiner._creator:
    fail.append("second open became the creator instead of joining")
joinerview = joiner._base.view(CHUNKS, ROW)
if not torch.all(joinerview[0, :TENSOR] == 7):
    fail.append(
        f"joiner does not see the creator's rows: {joinerview[0, :4].tolist()}"
    )
else:
    print("shared view  joiner sees the creator's bytes")

joinerview[0, :TENSOR].fill_(9)
if int(view[0, 0]) != 9:
    fail.append("write through the joiner mapping is not visible to the creator")
else:
    print("shared write creator sees the joiner's write")

# Drop every reference before cleanup: torch keeps the mmap buffer exported.
del joinerview
joiner.cleanup()
print(f"file after joiner cleanup: {os.path.exists(path)} (expected True)")

del view
creator.cleanup()
removed = not os.path.exists(path)
print(f"file after creator cleanup removed: {removed}")
if not removed:
    fail.append("region file survived cleanup")

# A file left behind by an unclean exit must not be joined by the next start.
with open(path, "wb") as fh:
    fh.write(b"\x01" * 4096)
again = make()
print(f"stale file cleared, became creator: {again._creator}")
if not again._creator:
    fail.append("a stale region file was joined instead of replaced")
if os.path.getsize(path) != CHUNKS * ROW:
    fail.append("stale region file was not resized")
again.cleanup()

# Files from other (dead) engines have to be reclaimed too: every run gets a
# fresh engine id, so its own cleanup never sees them again.
other = os.path.join(
    os.path.dirname(path), "vllm_offload_dead-engine-test.mmap"
)
with open(other, "wb") as fh:
    fh.write(b"\x02" * 4096)
third = make()
print(f"foreign stale region reclaimed: {not os.path.exists(other)}")
if os.path.exists(other):
    fail.append("a region file from an earlier run was left in place")
third.cleanup()

if fail:
    print(f"RESULT FAIL ({len(fail)})")
    for line in fail:
        print(f"  - {line}")
    sys.exit(1)
print("RESULT OK")
