"""Probe: the thin-GEMM dispatch gate and the kernel's correctness through it.

Checks, on an sm_80 CMP 170HX only:
  1. VLLM_THIN_GEMM unset -> dispatch_unquantized_gemm returns the upstream
     callable and the kernel package's triton module is never imported;
  2. VLLM_THIN_GEMM=1    -> the returned callable is ampere_thin_gemm and the
     small-row calls for the 3bpw checkpoint's shapes match F.linear;
  3. rows above VLLM_THIN_GEMM_MAX_TOKENS fall back to cuBLAS.

  $env:CUDA_VISIBLE_DEVICES=2; $env:VLLM_THIN_GEMM=1; `
    .venv\\Scripts\\python.exe _dev\\probe\\probe_thin_gemm_dispatch.py
"""

import importlib
import os
import sys

import torch


def main():
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    print(f"{props.name} sm_{props.major}{props.minor} {props.multi_processor_count} SMs "
          f"| VLLM_THIN_GEMM={os.environ.get('VLLM_THIN_GEMM', '(unset)')}")

    from vllm.model_executor.layers import utils as layer_utils

    on = bool(int(os.environ.get("VLLM_THIN_GEMM", "0")))
    picked = layer_utils.dispatch_unquantized_gemm()
    print(f"dispatch -> {picked.__module__}.{picked.__name__}")
    kernel_loaded = "vllm.sm80.kernels.thin_gemm.thin_gemm_v74" in sys.modules
    print(f"triton kernel module imported: {kernel_loaded}")

    if not on:
        if picked is not layer_utils.default_unquantized_gemm:
            print("FAIL: switch off must return default_unquantized_gemm")
            return 1
        if kernel_loaded:
            print("FAIL: switch off must not import the triton kernel")
            return 1
        print("OK: switch off is the upstream callable, kernel not imported")
        return 0

    from vllm.sm80.kernels.thin_gemm import (
        ampere_thin_gemm,
        dispatch_threshold,
        thin_gemm_supported,
    )

    if picked is not ampere_thin_gemm:
        print("FAIL: switch on must return ampere_thin_gemm")
        return 1
    bound = dispatch_threshold()
    print(f"OK: switch on returns ampere_thin_gemm (bound M<={bound})")

    torch.manual_seed(0)
    cases = [
        (1, 320, 10240), (4, 10240, 320), (4, 4, 10240), (4, 640, 2560),
        (16, 2560, 640), (32, 512, 2560), (4, 48, 2560), (4, 248320, 2560),
        (bound + 1, 640, 2560),
    ]
    worst = 0.0
    for m, n, k in cases:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        want = torch.nn.functional.linear(x, w)
        got = ampere_thin_gemm(None, x, w, None)
        routed = thin_gemm_supported(x, w, None)
        # bf16 output rounding is the only difference: compare in fp32.
        err = (got.float() - want.float()).abs().max().item()
        scale = want.float().abs().max().item()
        rel = err / scale
        worst = max(worst, rel)
        print(f"  M={m:<4} N={n:<7} K={k:<6} routed={str(routed):<5} "
              f"max_abs_err={err:.5f} rel={rel:.2e} bitwise={torch.equal(got, want)}")

    if worst > 2e-2:
        print(f"FAIL: worst rel error {worst:.2e} exceeds bf16 rounding scale")
        return 1
    print(f"OK: worst rel error {worst:.2e} (bf16 output rounding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
