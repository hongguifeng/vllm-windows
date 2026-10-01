"""Hostile-but-fair repro for the flashinfer radix top-k graph-replay defect.

The one-op graph in _repro_topk_graph.py caught a wrong-index row but only once
in 2400 replays, while a 24000-replay run with a deep host queue caught nothing.
Real vLLM differs from both: the drafter CUDA graph holds a whole model forward
in front of the top-k, and the engine overlaps host work with the GPU.

Two harnesses here:
  burst : the graph runs several large GEMMs before the top-k, so the top-k's
          77-CTA grid starts while the SMs are still draining the GEMMs -- the
          radix kernel syncs its CTAs through a software barrier in a global
          workspace, so late block scheduling is exactly what it cannot take.
  sync  : plain one-op graph, but the host drains the queue every replay.

Both are run against `flashinfer` (vLLM's current call) and `torch` (the
candidate replacement).  Usage: python _repro_topk_burst.py [iters]
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
BUCKETS = range(1, 9)
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


def make_call(kind):
    if kind == "torch":
        return lambda x: torch.topk(x, K, dim=-1)
    os.environ["FLASHINFER_TOPK_ALGO"] = "multi_cta"
    return lambda x: flashinfer.top_k(x, K, sorted=True, deterministic=True)


def verify(x, vals, idx, stats):
    bad = (idx < 0) | (idx >= V)
    n_oob = int(bad.sum().item())
    if n_oob:
        stats["oob"] += n_oob
        if stats["first"] is None:
            r = int(torch.nonzero(bad.any(-1), as_tuple=False)[0])
            stats["first"] = ("oob", r, idx[r].tolist())
    safe = idx.clamp(0, V - 1)
    stats["dup"] += int((torch.sort(safe, dim=-1).values.diff(dim=-1) == 0).sum().item())
    ref_v, ref_i = torch.topk(x, K, dim=-1)
    if not torch.equal(vals, ref_v):
        stats["vmm"] += 1
    if not torch.equal(torch.sort(safe, dim=-1).values, torch.sort(ref_i, dim=-1).values):
        stats["iset"] += 1
        if stats["first"] is None:
            rows = torch.nonzero(
                (torch.sort(safe, dim=-1).values != torch.sort(ref_i, dim=-1).values).any(-1),
                as_tuple=False,
            ).flatten().tolist()
            r = rows[0]
            stats["first"] = ("iset", r, (safe[r].tolist(), ref_i[r].tolist()))
    stats["n"] += 1


def build_graph(kind, n_reqs, burst):
    call = make_call(kind)
    x = torch.randn(n_reqs * N_STEPS, V, device="cuda", dtype=DTYPE)
    # a fat weight so the graph runs real GEMMs in front of the top-k
    w = torch.randn(2048, 5120, device="cuda", dtype=DTYPE)
    h = torch.randn(n_reqs * N_STEPS, 5120, device="cuda", dtype=DTYPE)
    for _ in range(3):
        call(x) if not burst else (h @ w.T, call(x))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if burst:
            for _ in range(6):
                _ = h @ w.T
        out = call(x)
    return g, x, out, h


def run(kind, burst, iters):
    stats = dict(oob=0, dup=0, vmm=0, iset=0, n=0, first=None)
    graphs = {n: build_graph(kind, n, burst) for n in BUCKETS}
    t0 = time.time()
    for _ in range(iters):
        for _, (g, x, out, h) in graphs.items():
            x.normal_()
            h.normal_()
            g.replay()
            if not burst:
                torch.cuda.synchronize()
            verify(x, out[0], out[1], stats)
    torch.cuda.synchronize()
    bad = stats["oob"] or stats["dup"] or stats["vmm"] or stats["iset"]
    print(
        f"  {kind:<8} burst={int(burst)} replays={stats['n']:<6} oob={stats['oob']:<5} "
        f"dup={stats['dup']:<5} val_mismatch={stats['vmm']:<4} idxset_mismatch={stats['iset']:<4} "
        f"-> {'BROKEN' if bad else 'clean'} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    if stats["first"]:
        print(f"           first: {stats['first']}", flush=True)
    return bad


def main():
    print(f"flashinfer {getattr(flashinfer, '__version__', '?')} / torch {torch.__version__}", flush=True)
    print(f"V={V} K={K} rows=n_reqs*{N_STEPS} buckets={list(BUCKETS)} iters={ITERS}", flush=True)
    total = 0
    for burst in (True, False):
        for kind in ("flashinfer", "torch"):
            total += run(kind, burst, ITERS)
    print(f"\nbroken runs: {total}/4")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
