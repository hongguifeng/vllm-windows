#!/usr/bin/env python3
"""Correctness gates for the prepared Triton launch fast path.

The fast path is only allowed to reuse a compiled kernel when the call still
satisfies every assumption Triton baked into it. These checks force the cases
where that is not true: pointer alignment flips, an integer stops being one or
stops being divisible by sixteen, a constexpr changes, launch hooks appear, the
kernel reads a module global, or compile options change. In every case the
result must match what Triton's own dispatch produces.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["VLLM_TRITON_PREPARED_LAUNCH"] = "1"
if "STRICT" in sys.argv[1:]:
    os.environ["VLLM_TRITON_PREPARED_LAUNCH_STRICT"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton import knobs  # noqa: E402

from vllm.triton_utils import prepared  # noqa: E402


@triton.jit
def add_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(b_ptr + offs, tl.load(a_ptr + offs, mask=mask) + 1, mask=mask)


@triton.jit
def stride_kernel(a_ptr, b_ptr, stride, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(b_ptr + offs * stride, tl.load(a_ptr + offs * stride, mask=mask) + 2,
             mask=mask)


SCALE = tl.constexpr(3)


@triton.jit
def global_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(b_ptr + offs, tl.load(a_ptr + offs, mask=mask) * SCALE, mask=mask)


def make(n, misaligned=False):
    a = torch.zeros(n + 4, device="cuda", dtype=torch.int32)
    b = torch.zeros(n + 4, device="cuda", dtype=torch.int32)
    if misaligned:
        a = a[1:]
        b = b[1:]
    return a, b


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main() -> None:
    torch.cuda.init()
    installed = prepared.install_prepared_launch()
    print(f"installed: {installed}")
    results = []

    a, b = make(256)
    add_kernel[(4,)](a, b, 256, BLOCK=256)
    torch.cuda.synchronize()
    stats_before = prepared.prepared_launch_stats()["full"]
    cache = getattr(add_kernel, "_vllm_prepared", {})
    results.append(check("first call compiled through Triton", len(cache) == 1))

    b.zero_()
    add_kernel[(4,)](a, b, 256, BLOCK=256)
    torch.cuda.synchronize()
    hits = sum(e.hits for e in cache.values())
    results.append(check("identical call served from cache", hits == 1))
    results.append(check("cached launch still correct", bool((b[:256] == 1).all())))

    # Different data, same call shape: pointer values change, alignment does not.
    a2, b2 = make(256)
    a2.fill_(7)
    add_kernel[(4,)](a2, b2, 256, BLOCK=256)
    torch.cuda.synchronize()
    results.append(check("different tensors correct", bool((b2[:256] == 8).all())))

    # Constexpr changed: must not reuse the kernel baked with the old value.
    b.zero_()
    add_kernel[(2,)](a, b, 256, BLOCK=128)
    torch.cuda.synchronize()
    results.append(check("changed constexpr correct", bool((b[:256] == 1).all())))
    results.append(check("changed constexpr got its own entry", len(cache) == 2))

    # n stops being divisible by sixteen: specialization no longer holds.
    n = 255
    b.zero_()
    add_kernel[(4,)](a, b, n, BLOCK=256)
    torch.cuda.synchronize()
    expect = torch.ones(255, dtype=torch.int32, device="cuda")
    results.append(check("non-divisible n correct", bool((b[:n] == expect).all())))

    # Pointer alignment flips: Triton compiled assuming 16 byte alignment.
    am, bm = make(256, misaligned=True)
    bm.zero_()
    add_kernel[(4,)](am, bm, 256, BLOCK=256)
    torch.cuda.synchronize()
    results.append(check("misaligned pointer correct", bool((bm[:256] == 1).all())))

    # stride == 1 is its own specialization class.
    a, b = make(256)
    b.zero_()
    stride_kernel[(4,)](a, b, 1, 256, BLOCK=256)
    torch.cuda.synchronize()
    results.append(check("stride one correct", bool((b[:256] == 2).all())))
    b.zero_()
    stride_kernel[(4,)](a, b, 2, 256, BLOCK=256)
    torch.cuda.synchronize()
    results.append(check("stride two correct", bool((b[:512:2] == 2).all())))

    # A kernel that reads a module global must never be cached.
    a, b = make(256)
    b.zero_()
    global_kernel[(4,)](a, b, 256, BLOCK=256)
    torch.cuda.synchronize()
    results.append(check("global kernel correct", bool((b[:256] == 0).all())))
    results.append(
        check("global kernel not cached",
              len(getattr(global_kernel, "_vllm_prepared", {})) == 0)
    )

    # Launch hooks appear: the fast path has to stand down so profilers see it.
    seen = []
    knobs.runtime.launch_enter_hook = lambda md: seen.append(1)
    b.zero_()
    add_kernel[(4,)](a, b, 256, BLOCK=256)
    torch.cuda.synchronize()
    results.append(check("hooks installed are honoured", bool(seen)))
    results.append(check("hook path correct", bool((b[:256] == 1).all())))
    knobs.runtime.launch_enter_hook = None

    # Compile options are part of the identity.
    b.zero_()
    add_kernel[(4,)](a, b, 256, BLOCK=256, num_warps=8)
    torch.cuda.synchronize()
    results.append(check("changed options correct", bool((b[:256] == 1).all())))

    stats = prepared.prepared_launch_stats()
    print(f"stats: {stats}")
    saved_check = prepared._strict_check
    results.append(check("fast path actually used", stats["fast"] > 0))
    if os.environ.get("VLLM_TRITON_PREPARED_LAUNCH_STRICT") == "1":
        results.append(check("strict verification ran", stats["checked"] > 0))
        results.append(check("cheap key never picked a wrong kernel",
                            stats["mismatch"] == 0))
        # Force a mismatch so the fallback path itself gets exercised: it must
        # produce the same result as Triton's dispatch and must not launch the
        # cached kernel.
        before = stats["fast"]
        prepared._strict_check = lambda jit, entry, args, kwargs: False
        b.zero_()
        add_kernel[(4,)](a, b, 256, BLOCK=256)
        torch.cuda.synchronize()
        after = prepared.prepared_launch_stats()
        results.append(check("forced mismatch falls back to Triton",
                            after["fallback"] > 0))
        results.append(check("forced fallback did not use the cache",
                            after["fast"] == before))
        results.append(check("forced fallback correct", bool((b[:256] == 1).all())))
        prepared._strict_check = saved_check
    results.append(check("no crash, all correct", all(results)))
    print(f"\n{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
