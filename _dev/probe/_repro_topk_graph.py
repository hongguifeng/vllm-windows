"""Offline repro: does flashinfer.top_k emit unwritten rows under CUDA-graph replay?

vLLM's DFlash2 drafter calls LogitsProcessor.get_top_k_tokens() ->
flashinfer.top_k(scores, k, sorted=True, deterministic=True) from *inside* a
captured CUDA graph.  The engine then dies on `predecessor_table[candidate_ids]`
with 0 <= id < 248320 asserted, so some rows of the top-k output are garbage.

This script calls the exact same op with the exact same shapes, first eagerly and
then from a replayed graph, and checks both the index range and correctness
against torch.topk.

Run it from anywhere (this file only imports torch + flashinfer, never vllm), but
put `.venv/Scripts` on PATH first so flashinfer's JIT can find ninja.
GPU: RTX 3090 (sm_86), flashinfer 0.6.11.
"""

import sys

import torch
import flashinfer

V = 248320          # candidate_selector vocab (codebook rows)
K = 16              # selector_top_k
N_STEPS = 7         # num_speculative_steps -> num_rows = num_reqs * N_STEPS
DTYPE = torch.float32


def top_k(x):
    return flashinfer.top_k(x, K, sorted=True, deterministic=True)


def verify(x, vals, idx, label):
    """Check an already-produced (vals, idx) pair against torch.topk."""
    torch.cuda.synchronize()
    oob = int(((idx < 0) | (idx >= V)).sum().item())
    ref_v, ref_i = torch.topk(x, K, dim=-1)
    vals_ok = bool(torch.equal(vals, ref_v))
    idx_ok = bool(
        torch.equal(
            torch.sort(idx.clamp(0, V - 1), dim=-1).values,
            torch.sort(ref_i, dim=-1).values,
        )
    )
    if oob or not vals_ok or not idx_ok:
        bad_rows = torch.nonzero(
            ((idx < 0) | (idx >= V)).any(dim=-1), as_tuple=False
        ).flatten().tolist()
        print(
            f"  [FAIL] {label}: oob={oob} vals_ok={vals_ok} idx_ok={idx_ok} "
            f"bad_rows={bad_rows[:8]} (of {idx.shape[0]})"
        )
        if bad_rows:
            r = bad_rows[0]
            print(f"         row {r} got={idx[r].tolist()}")
            print(f"         row {r} ref={ref_i[r].tolist()}")
        return False
    return True


def fresh(n_reqs):
    return torch.randn(n_reqs * N_STEPS, V, device="cuda", dtype=DTYPE)


def main():
    print(f"flashinfer {getattr(flashinfer, '__version__', '?')} / torch {torch.__version__}")
    print(f"rows/bucket = n_reqs * {N_STEPS}, V={V}, K={K}")

    # ---------------- phase 0: eager ----------------
    print("\n== phase 0: eager, varying n_reqs ==")
    bad = 0
    for it in range(120):
        n_reqs = 1 + (it * 7) % 8
        x = fresh(n_reqs)
        v, i = top_k(x)
        if not verify(x, v, i, f"eager it={it} n_reqs={n_reqs}"):
            bad += 1
    print(f"  eager done, failures={bad}")

    # ---------------- phase 1: one graph per bucket ----------------
    print("\n== phase 1: CUDA graph replay per n_reqs bucket ==")
    buckets = {}
    for n_reqs in range(1, 9):
        x = fresh(n_reqs)
        for _ in range(3):          # warmup on the side stream, like vllm does
            top_k(x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            v, i = top_k(x)
        buckets[n_reqs] = (g, x, v, i)
    print(f"  captured {len(buckets)} graphs")

    bad = 0
    for it in range(300):
        for n_reqs, (g, x, v, i) in buckets.items():
            x.normal_()             # new logits every step, like real decode
            g.replay()
            if not verify(x, v, i, f"graph it={it} n_reqs={n_reqs}"):
                bad += 1
                if bad >= 5:
                    print("  stopping after 5 failures")
                    return 1
    print(f"  graph done, failures={bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
