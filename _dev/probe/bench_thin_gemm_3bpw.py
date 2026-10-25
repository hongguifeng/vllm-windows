"""Microbenchmark: fork sm_80 thin GEMM vs cuBLAS on the 3bpw checkpoint's BF16 shapes.

Shapes come from `probe_3bpw_bf16_shapes.py` (the checkpoint's real BF16 weights), so
this measures the shapes the model actually calls, on the part actually served
(CMP 170HX, sm_80, 74 SMs).

Method: per (shape, M) time cuBLAS `F.linear` and the Triton `thin_gemm` in the same
process, alternating implementations, with a set of weight copies rotated between
reps so L2 cannot hold a weight, plus a flush buffer read between reps. Median of
`REPS` CUDA-event timings. Eager launches (that is the conservative case for
thin_gemm: a graph replay removes its launch overhead, not cuBLAS's work).

  $env:CUDA_VISIBLE_DEVICES=1; .venv\\Scripts\\python.exe _dev\\probe\\bench_thin_gemm_3bpw.py
"""

import importlib.util
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", "_scratch", "recipe_src", "sm80",
                                   "kernels", "thin_gemm"))
SHAPES_JSON = os.path.join(HERE, "_3bpw_bf16_shapes.json")

REPS = 30
BLOCKS = 3
COPIES = 4
FLUSH_MB = 320

# (shape, calls per layer pass, layers) for the step-level estimate. Layer-pass
# counts from the checkpoint inventory: 48 layers, 36 GDN + 12 full attention.
STEP_CALLS = {
    (320, 10240): 96,   # attn+mlp hyper-connection down, 2 per layer
    (10240, 320): 96,   # ... up, 2 per layer
    (4, 10240): 96,     # ... block inject, 2 per layer
    (512, 2560): 48,    # MoE router
    (1, 2560): 48,      # shared-expert scalar gate
    (640, 2560): 96,    # shared-expert gate + up
    (2560, 640): 48,    # shared-expert down
    (48, 2560): 72,     # GDN in_proj_a + in_proj_b, 36 layers
    (12800, 2560): 1,   # PLE fused kv_proj (only when the PLE path runs)
    (10240, 2560): 1,   # PLE key_proj (our model keeps key/value separate)
    (2560, 2560): 1,    # PLE value_proj
    (248320, 2560): 1,  # lm_head, fully bf16 in this checkpoint
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def timeit(fn, flush, sink):
    times = []
    for _ in range(REPS):
        sink.add_(flush[::64].float().sum())
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
        del y
    return statistics.median(times)


def main():
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    print(f"{props.name} sm_{props.major}{props.minor} {props.multi_processor_count} SMs "
          f"| {props.total_memory / 2**30:.1f} GiB")

    v74 = load("tgv74", os.path.join(PKG, "thin_gemm_v74.py"))
    # Gluon needs gl.barrier, absent from Triton 3.6.0's gluon language; keep it
    # off by default so the arm does not fail compile on every shape.
    gluon = None
    if os.environ.get("WITH_GLUON"):
        try:
            gluon = load("tggluon", os.path.join(PKG, "thin_gemm_gluon.py"))
        except Exception as exc:  # noqa: BLE001
            print(f"gluon module not importable: {type(exc).__name__}: {exc}")

    inventory = json.load(open(SHAPES_JSON))["shapes"]
    shapes = [(n, k, c) for n, k, c in inventory
              if (n, k) in STEP_CALLS and k >= 128]
    shapes.sort(key=lambda t: -(t[0] * t[1]))
    print(f"benchmarking {len(shapes)} shapes reaching the decode GEMM path\n")

    flush = torch.empty(FLUSH_MB * 2**20 // 4, device=dev, dtype=torch.float32)
    sink = torch.zeros(1, device=dev)
    ms_all = {"cuBLAS": 0.0, "v74": 0.0, "gluon": 0.0}
    rows = []
    ms_list = [1, 2, 4, 8, 16, 32]

    for n, k, _calls in shapes:
        copies = COPIES if n * k * 2 < 8 * 2**20 else 2
        weights = [torch.randn(n, k, device=dev, dtype=torch.bfloat16)
                   for _ in range(copies)]
        if gluon is not None:
            try:
                gluon.warmup([(n, k)], ms_list, device=dev)
            except Exception as exc:  # noqa: BLE001
                print(f"  gluon warmup failed for N={n} K={k}: "
                      f"{type(exc).__name__}: {exc}")
        for m in ms_list:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            outs = {}

            def run(impl):
                def f():
                    w = weights[impl_state[0] % copies]
                    impl_state[0] += 1
                    if impl == "cuBLAS":
                        return torch.nn.functional.linear(x, w)
                    if impl == "v74":
                        return v74.thin_gemm(x, w)
                    return gluon.thin_gemm(x, w)
                return f

            impl_state = [0]
            ref = torch.nn.functional.linear(x, weights[0])
            # Three alternating blocks: one cuBLAS timing and one thin-GEMM
            # timing drift by up to 2x between runs on this part (clock and L2
            # state), so take the median of the per-block medians and report the
            # block-to-block spread.
            blocks = {"cuBLAS": [], "v74": []}
            if gluon is not None:
                blocks["gluon"] = []
            for block in range(BLOCKS):
                for impl in ("cuBLAS", "v74") if block % 2 == 0 else ("v74", "cuBLAS"):
                    impl_state[0] = 0
                    blocks[impl].append(timeit(run(impl), flush, sink))
                if gluon is not None:
                    impl_state[0] = 0
                    blocks["gluon"].append(timeit(run("gluon"), flush, sink))
            for impl, ts in blocks.items():
                if ts:
                    rows.append((n, k, m, impl, statistics.median(ts),
                                 max(ts) - min(ts)))
            del outs, x, ref

    by = {}
    for n, k, m, impl, t, spread in rows:
        by.setdefault((n, k, m), {})[impl] = (t, spread)

    print(f"\n{'N':>6} {'K':>6} {'M':>3} | {'cuBLAS us':>10} {'v74 us':>8} {'x':>6} "
          f"{'calls':>6} {'save us/step':>12} {'spread':>7}")
    step_gain = {}
    for (n, k, m) in sorted(by, key=lambda t: (-t[0] * t[1], t[2])):
        r = by[(n, k, m)]
        cb, v = r["cuBLAS"][0], r["v74"][0]
        calls = STEP_CALLS.get((n, k), 0)
        gain = (cb - v) * calls
        step_gain[m] = step_gain.get(m, 0.0) + gain
        print(f"{n:>6} {k:>6} {m:>3} | {cb:>10.1f} {v:>8.1f} {cb / v:>6.2f} "
              f"{calls:>6} {gain:>12.0f} "
              f"{max(r['cuBLAS'][1], r['v74'][1]):>7.1f}")

    print()
    for m in sorted(step_gain):
        print(f"  step-level saving at M={m:<3}: {step_gain[m] / 1000:+.2f} ms/step")
    print("(eager launches; graph replay removes launch gaps for both, so the "
          "small shapes' saving is the optimistic end)")

    out = os.path.join(HERE, "_thin_gemm_3bpw_bench.json")
    with open(out, "w") as f:
        json.dump([[n, k, m, impl, t, spread] for n, k, m, impl, t, spread in rows], f,
                  indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
