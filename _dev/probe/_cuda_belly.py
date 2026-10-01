#!/usr/bin/env python3
"""Do a few CUDA operations so a profiler can be tested for kernel capture.

A profiler that reports no kernels has to be told whether the tool or the
application wiring is at fault. This runs work that is large enough to name on
GPU and finishes on its own, so a capture tool can be checked in about a minute
instead of through a full model launch.
"""
import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=200)
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("cuda not available")
    print(f"device: {torch.cuda.get_device_name(0)}")
    a = torch.randn(args.size, args.size, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(args.size, args.size, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(a)
    start = time.perf_counter()
    for _ in range(args.ops):
        torch.matmul(a, b, out=out)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    copy_src = torch.randn(1024, 4096, dtype=torch.float32, device="cuda")
    copy_dst = torch.empty(1024, 4096, dtype=torch.float32, device="cpu",
                           pin_memory=True)
    for _ in range(20):
        copy_dst.copy_(copy_src, non_blocking=True)
    torch.cuda.synchronize()
    print(f"{args.ops} matmuls of {args.size} in {wall * 1000:.1f} ms; "
          f"{(args.ops * 2 * args.size ** 3) / wall / 1e9:.1f} GFLOP/s")


if __name__ == "__main__":
    main()
