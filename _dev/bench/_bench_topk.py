"""Cost of replacing flashinfer.top_k with torch.topk in the DFlash2 drafter.

The drafter calls get_top_k_tokens() once per draft step on
(num_reqs * num_speculative_steps, vocab) fp32 logits with k = selector_top_k.
Timed here both eagerly and from inside a captured graph.

Usage: python _bench_topk.py
"""

import os
import statistics
import sys
import time

import torch
import flashinfer

V = 248320
K = 16
N_STEPS = 7
DTYPE = torch.float32
REPS = 50


def cuda_ms(fn, reps=REPS):
    fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    return statistics.median(ts)


def main():
    print(f"flashinfer {getattr(flashinfer, '__version__', '?')} / torch {torch.__version__}")
    print(f"V={V} K={K} fp32  (median ms of {REPS} reps, eager + graph replay)\n")
    print(f"{'rows':>5} {'n_reqs':>6} | {'flex_eager':>10} {'torch_eager':>11} "
          f"| {'flex_graph':>10} {'torch_graph':>11}")
    for n_reqs in (1, 2, 4, 8):
        rows = n_reqs * N_STEPS
        x = torch.randn(rows, V, device="cuda", dtype=DTYPE)

        flex = lambda: flashinfer.top_k(x, K, sorted=True, deterministic=True)   # noqa: E731
        tor = lambda: torch.topk(x, K, dim=-1)                                   # noqa: E731

        fe, te = cuda_ms(flex), cuda_ms(tor)

        # graph replay
        gs = {}
        for name, fn in (("flex", flex), ("torch", tor)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            gs[name] = g
        fg = cuda_ms(gs["flex"].replay)
        tg = cuda_ms(gs["torch"].replay)

        print(f"{rows:>5} {n_reqs:>6} | {fe:>10.3f} {te:>11.3f} "
              f"| {fg:>10.3f} {tg:>11.3f}   torch/flex = {tg / max(fg, 1e-6):.2f}x")
        del x
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
