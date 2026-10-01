"""Long-run confidence test: which top-k path survives sustained graph replay?

Short runs hit the flashinfer bug ~1/2000 replays, so 2000 replays is not enough
to clear a variant.  This runs 8*3000 = 24000 replays per variant with a strict
check (index range, in-row duplicates, value match, index-set match, and
value/index self-consistency via gather).

Usage: python _repro_topk_long.py [variant ...]
  variants: multi_cta | filtered | torch  (default: multi_cta filtered)
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
ITERS = 3000
BUCKETS = range(1, 9)


def make_call(kind, deterministic=True, sorted_=True):
    if kind == "torch":
        def f(x):
            return torch.topk(x, K, dim=-1, sorted=sorted_)
        return f
    os.environ["FLASHINFER_TOPK_ALGO"] = kind

    def f(x):
        return flashinfer.top_k(x, K, sorted=sorted_, deterministic=deterministic)

    return f


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
    if not torch.equal(x.gather(1, safe), vals):
        stats["incons"] += 1
        if stats["first"] is None:
            stats["first"] = ("incons", -1, None)
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


def run(kind, iters=ITERS):
    stats = dict(oob=0, dup=0, vmm=0, iset=0, incons=0, n=0, first=None)
    call = make_call(kind)
    for _ in range(20):
        call(torch.randn(4 * N_STEPS, V, device="cuda", dtype=DTYPE))
    torch.cuda.synchronize()

    graphs = {}
    for n_reqs in BUCKETS:
        x = torch.randn(n_reqs * N_STEPS, V, device="cuda", dtype=DTYPE)
        for _ in range(3):
            call(x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = call(x)
        graphs[n_reqs] = (g, x, out)

    t0 = time.time()
    for _ in range(iters):
        for _, (g, x, out) in graphs.items():
            x.normal_()
            g.replay()
            verify(x, out[0], out[1], stats)
    torch.cuda.synchronize()

    bad = stats["oob"] or stats["dup"] or stats["vmm"] or stats["iset"] or stats["incons"]
    rate = bad / max(stats["n"], 1)
    print(
        f"{kind:<10} replays={stats['n']:<6} oob={stats['oob']:<5} dup={stats['dup']:<5} "
        f"val_mismatch={stats['vmm']:<4} idxset_mismatch={stats['iset']:<4} "
        f"inconsistent={stats['incons']:<4} -> {'BROKEN' if bad else 'clean'} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    if stats["first"]:
        print(f"           first: {stats['first']}", flush=True)
    return bad


def main():
    kinds = sys.argv[1:] or ["multi_cta", "filtered"]
    print(f"flashinfer {getattr(flashinfer, '__version__', '?')} torch {torch.__version__}", flush=True)
    total = 0
    for k in kinds:
        total += run(k)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
