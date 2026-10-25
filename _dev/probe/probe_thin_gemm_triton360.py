r"""Probe: does the fork's sm_80 thin-GEMM kernel compile and run under Triton 3.6.0?

The vendored kernel documents "Triton 3.7.1". Our venv has 3.6.0 (Windows wheel).
If this probe fails to compile, porting thin GEMM needs a Triton upgrade first,
which on Windows may not exist -- that changes the plan, so check it directly.

Run on a CMP 170HX (sm_80) only:
  $env:CUDA_VISIBLE_DEVICES=1; .venv\Scripts\python.exe _dev\probe\probe_thin_gemm_triton360.py
"""

import importlib.util
import os
import sys

import torch

SRC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "_scratch",
    "recipe_src",
    "sm80",
    "kernels",
    "thin_gemm",
    "thin_gemm_v74.py",
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import triton

    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    print(f"triton {triton.__version__} | {props.name} sm_{props.major}{props.minor} "
          f"{props.multi_processor_count} SMs")

    tg = load(SRC, "thin_gemm_v74_probe")

    cases = [
        (4, 512, 2560, "moe router"),
        (4, 336, 10240, "mHC down"),
        (1, 12800, 2560, "PLE kv_proj"),
        (4, 10240, 320, "mHC up"),
        (4, 4096, 4096, "not in table"),
    ]
    worst = 0.0
    for m, n, k, label in cases:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
        try:
            y = tg.thin_gemm(x, w)
            ref = torch.nn.functional.linear(x, w)
            err = (y.float() - ref.float()).abs().max().item()
            rel = err / ref.float().abs().max().item()
            worst = max(worst, rel)
            print(f"  OK  M={m:<3} N={n:<6} K={k:<6} {label:<16} "
                  f"max_abs_err={err:.4f} rel={rel:.2e}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL M={m:<3} N={n:<6} K={k:<6} {label}: "
                  f"{type(exc).__name__}: {exc}")
            return 1

    # warmup() is the pre-capture path the port would call.
    shapes = [(512, 2560), (336, 10240), (12800, 2560)]
    try:
        tg.warmup(shapes, [1, 2, 4, 8], device=dev)
        print(f"  OK  warmup{shapes} x M 1..8")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL warmup: {type(exc).__name__}: {exc}")
        return 1

    # Re-run to confirm the warmed workspace path is reusable.
    x = torch.randn(4, 2560, device=dev, dtype=torch.bfloat16)
    w = torch.randn(12800, 2560, device=dev, dtype=torch.bfloat16)
    y = tg.thin_gemm(x, w)
    ref = torch.nn.functional.linear(x, w)
    bitwise = torch.equal(y, tg.thin_gemm(x, w))
    print(f"  rerun after warmup: max_abs_err="
          f"{(y.float() - ref.float()).abs().max().item():.4f} "
          f"deterministic={bitwise} worst_rel={worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
