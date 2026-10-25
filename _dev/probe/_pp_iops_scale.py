#!/usr/bin/env python3
"""Random 4 KiB read throughput at different queue depths on the PLE shard.

Threads block on pread, so the thread count is the achieved queue depth. The
engine's mmap gather uses VLLM_PLE_MMAP_WORKERS (default 32) the same way.
"""
import os
import random
import sys
import threading
import time

PATH = ("/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound/"
        "model-00016-of-00017.safetensors")
SIZE = os.path.getsize(PATH)
DUR = 3.0


def run(threads: int) -> tuple[int, int]:
    counts = [0] * threads
    bytes_ = [0] * threads

    def worker(idx: int) -> None:
        fd = os.open(PATH, os.O_RDONLY)
        rnd = random.Random(idx)
        end = time.time() + DUR
        n = 0
        b = 0
        while time.time() < end:
            off = rnd.randrange(0, SIZE - 4096) & ~4095
            b += len(os.pread(fd, 4096, off))
            n += 1
        counts[idx] = n
        bytes_[idx] = b
        os.close(fd)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.time() - t0
    return int(sum(counts) / dt), int(sum(bytes_) / dt)


if __name__ == "__main__":
    print(f"file {SIZE/2**30:.2f} GiB, {DUR:.0f}s per level, threads == queue depth")
    for qd in (1, 8, 32, 64, 128):
        iops, bps = run(qd)
        print(f"  qd {qd:3d}: {iops:6d} IOPS  {bps/1e6:6.1f} MB/s  "
              f"({iops*4096/1e6:.1f} MB/s of 4 KiB payload)")
