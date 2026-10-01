"""Dual-stream cost/corruption probe, instrumented: one kind, N iterations.

Companion to _repro_topk_streams.py, which only printed at the very end -- too
coarse when a single iteration can take seconds.  Prints per-iteration wall time
so the flashinfer / torch difference is visible immediately.

Usage: python _repro_topk_streams2.py <flashinfer|torch> [iters]
"""

import os
import sys
import time

import torch
import flashinfer

V = 248320
K = 16
N_STEPS = 7
DTYPE = torch.float32
ROWS = 8 * N_STEPS


def make_call(kind):
    if kind == "torch":
        return lambda x: torch.topk(x, K, dim=-1)
    os.environ["FLASHINFER_TOPK_ALGO"] = "multi_cta"
    return lambda x: flashinfer.top_k(x, K, sorted=True, deterministic=True)


def main():
    kind = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    call = make_call(kind)
    x1 = torch.randn(ROWS, V, device="cuda", dtype=DTYPE)
    x2 = torch.randn(ROWS, V, device="cuda", dtype=DTYPE)
    for _ in range(3):
        call(x1)
    torch.cuda.synchronize()
    # Two graphs must NOT share a memory pool: torch.cuda.graph() defaults every
    # capture into the same pool, and then two concurrent replays write into the
    # same addresses and clobber each other.  (vLLM shares a pool on purpose but
    # only ever replays one graph at a time.)
    pool1, pool2 = torch.cuda.graph_pool_handle(), torch.cuda.graph_pool_handle()
    g1, g2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool1):
        o1 = call(x1)
    with torch.cuda.graph(g2, pool=pool2):
        o2 = call(x2)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    print(f"{kind}: {iters} dual-stream replays, rows={ROWS}, V={V}, K={K}", flush=True)
    stats = {"oob": 0, "iset": 0}
    for i in range(iters):
        t0 = time.time()
        # keep each stream's own input write ordered before its own replay;
        # the two streams themselves stay concurrent
        with torch.cuda.stream(s1):
            x1.normal_()
            g1.replay()
        with torch.cuda.stream(s2):
            x2.normal_()
            g2.replay()
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000
        bad = 0
        for x, o in ((x1, o1), (x2, o2)):
            idx = o[1]
            bad += int(((idx < 0) | (idx >= V)).sum().item())
            safe = idx.clamp(0, V - 1)
            ref = torch.topk(x, K, dim=-1).indices
            if not torch.equal(torch.sort(safe, -1).values, torch.sort(ref, -1).values):
                stats["iset"] += 1
        stats["oob"] += bad
        print(f"  iter {i:>3}  {dt:>9.1f} ms   oob_this_iter={bad}", flush=True)
    print(f"{kind} done: oob={stats['oob']} idxset_mismatch={stats['iset']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
