#!/usr/bin/env python3
"""Why does the strict check see a different CompiledKernel for the same args?

Replays three calls of one kernel, prints the cache entry we made, what Triton
resolves for the same arguments, and whether the two kernels are actually
equivalent (same name, same metadata, same cubin) or genuinely different.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["VLLM_TRITON_PREPARED_LAUNCH"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.runtime import driver  # noqa: E402
from triton.runtime.jit import compute_cache_key  # noqa: E402

from vllm.triton_utils import prepared  # noqa: E402


@triton.jit
def add_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(b_ptr + offs, tl.load(a_ptr + offs, mask=mask) + 1, mask=mask)


def triton_target(jit, args, kwargs):
    device = driver.active.get_current_device()
    kernel_cache, kernel_key_cache, _t, _b, binder = jit.device_caches[device]
    _bound, specialization, options = binder(*args, **kwargs)
    key = compute_cache_key(kernel_key_cache, specialization, options)
    print(f"  device        : {device}")
    print(f"  kernel_cache  : {len(kernel_cache)} entries")
    for k, v in kernel_cache.items():
        print(f"    key={k!r} -> {v.name} id={id(v)}")
    print(f"  kernel_key_cache: {len(kernel_key_cache)} entries")
    for k, v in kernel_key_cache.items():
        print(f"    {k} -> {v!r}")
    return key, specialization, kernel_cache.get(key)


def main() -> None:
    torch.cuda.init()
    prepared.install_prepared_launch()

    a = torch.zeros(260, device="cuda", dtype=torch.int32)
    b = torch.zeros(260, device="cuda", dtype=torch.int32)

    calls = [
        ("first", (a, b, 256), {"BLOCK": 256}),
        ("second", (a, b, 256), {"BLOCK": 256}),
        ("other tensors", (a[4:], b[4:], 256), {"BLOCK": 256}),
    ]
    for label, args, kwargs in calls:
        add_kernel[(4,)](*args, **kwargs)
        torch.cuda.synchronize()
        cheap = prepared._key(add_kernel, args, kwargs)
        entry = getattr(add_kernel, "_vllm_prepared", {}).get(cheap)
        tkey, spec, tkernel = triton_target(add_kernel, args, kwargs)
        print(f"\n--- {label}")
        print(f"  cheap key : {cheap}")
        print(f"  entry     : {id(entry.compiled) if entry else 'none'}")
        print(f"  triton key: {tkey}")
        print(f"  spec      : {spec}")
        print(f"  triton kernel: {id(tkernel) if tkernel else 'none'}")
        if entry is not None and tkernel is not None:
            same_obj = entry.compiled is tkernel
            print(f"  same object      : {same_obj}")
            print(f"  same name        : {entry.compiled.name == tkernel.name}")
            print(f"  same hash        : {entry.compiled.hash == tkernel.hash}")
            print(f"  same metadata    : {str(entry.compiled.metadata) == str(tkernel.metadata)}")
            cu1 = getattr(entry.compiled, "asm", {}).get("cubin", "")
            cu2 = getattr(tkernel, "asm", {}).get("cubin", "")
            print(f"  same cubin       : {cu1 == cu2}  "
                  f"({len(cu1)} vs {len(cu2)} bytes)")
            print(f"  same function    : {entry.compiled.function == tkernel.function}")
        stats = prepared.prepared_launch_stats()
        print(f"  stats: {stats}")


if __name__ == "__main__":
    main()
