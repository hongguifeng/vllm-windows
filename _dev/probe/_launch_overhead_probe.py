"""Measure CPU-side launch overhead: Triton JIT dispatch vs a prepared launch,
and the per-step mamba block-table index math (six aten ops)."""
import time

import torch
import triton
import triton.language as tl


def bench(label: str, fn, iters: int) -> float:
    fn()  # warm
    torch.cuda.synchronize()
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters * 1e6
        best = dt if best is None else min(best, dt)
    torch.cuda.synchronize()
    print(f"{label:38s} {best:8.2f} us/call  ({iters} iters, best of 3)")
    return best


@triton.jit
def prep_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr, g_ptr, h_ptr,
                n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(c_ptr + offs, tl.load(a_ptr + offs, mask=mask) + 1, mask=mask)


@triton.jit(do_not_specialize=["n"])
def prep_kernel_ns(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr, g_ptr, h_ptr,
                   n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(c_ptr + offs, tl.load(a_ptr + offs, mask=mask) + 1, mask=mask)


def main() -> None:
    torch.cuda.init()
    n = 1024
    a = torch.zeros(n, device="cuda", dtype=torch.int32)
    b = torch.zeros(n, device="cuda", dtype=torch.int32)
    c = torch.zeros(n, device="cuda", dtype=torch.int32)
    scalars = [1, 2, 3, 4, 5, 6, 7]

    args = (a, b, c, b, a, c, a, b)

    def jit_call():
        prep_kernel[(4,)](*args, n=256, BLOCK=256)

    jit_call()
    torch.cuda.synchronize()
    jit_us = bench("triton JIT call (8 ptr + 2 constexpr)", jit_call, 2000)

    # Prepared launch: reuse the CompiledKernel and skip binder, cache key and
    # the used-globals check.
    ck = prep_kernel.warmup(*args, n=256, BLOCK=256, grid=(4, 1, 1))
    ck._init_handles()
    stream = torch.cuda.current_stream().cuda_stream

    def prepared_call():
        ck.run(4, 1, 1, stream, ck.function, ck.packed_metadata,
               None, None, None, *args, 256, 256)

    prepared_call()
    torch.cuda.synchronize()
    prepared_us = bench("triton prepared launch (bypass JIT)", prepared_call, 2000)
    print(f"  -> saving {jit_us - prepared_us:.2f} us/call "
          f"({100*(jit_us-prepared_us)/jit_us:.0f}% of the dispatch)")

    # The low-risk variant: keep the JIT but skip arg specialization.
    ns_us = bench("triton JIT with do_not_specialize", lambda: prep_kernel_ns[
        (4,)](*args, n=256, BLOCK=256), 2000)

    # A prepared launch is only safe if the specialization still holds, so the
    # realistic helper re-checks pointer alignment and the int special cases.
    aligned = [t.data_ptr() % 16 == 0 for t in args]

    def prepared_checked():
        for t, was in zip(args, aligned):
            if t.data_ptr() % 16 != 0 and was:
                raise RuntimeError("alignment changed")
        ck.run(4, 1, 1, stream, ck.function, ck.packed_metadata,
               None, None, None, *args, 256, 256)

    checked_us = bench("prepared launch + alignment check",
                       prepared_checked, 2000)

    # torch.cuda.synchronize() per call is what the profile showed as 3.5%.
    bench("torch.cuda.synchronize()", lambda: torch.cuda.synchronize(), 2000)

    # What mamba_get_block_table_tensor does in "align" mode, every step.
    batch, cols = 4, 33
    block_size = 262144 // 1  # any large block size; only the math matters
    block_table = torch.arange(batch * cols, device="cuda",
                               dtype=torch.int32).reshape(batch, cols)
    seq_lens = torch.full((batch,), 4096, device="cuda", dtype=torch.int32)
    num_spec_blocks = 1

    def align_math():
        start = (seq_lens - 1) // block_size
        start.clamp_(min=0)
        offsets = torch.arange(1 + num_spec_blocks, device="cuda",
                               dtype=torch.int32)
        idx = (start.unsqueeze(1) + offsets).to(torch.int64)
        return torch.gather(block_table, 1, idx)

    align_us = bench("align block-table math (6 aten ops)", align_math, 3000)

    offsets_cached = torch.arange(1 + num_spec_blocks, device="cuda",
                                  dtype=torch.int32)
    idx_cached = ((seq_lens - 1) // block_size).clamp_(min=0).unsqueeze(1) \
        .add(offsets_cached).to(torch.int64)
    start_t = (seq_lens - 1) // block_size

    print("  per-op breakdown of those 6 aten ops:")
    bench("  (seq_lens - 1)", lambda: seq_lens - 1, 3000)
    bench("  // block_size", lambda: start_t // block_size, 3000)
    bench("  clamp_(min=0)", lambda: start_t.clamp_(min=0), 3000)
    bench("  torch.arange(2)", lambda: torch.arange(2, device="cuda",
                                                    dtype=torch.int32), 3000)
    bench("  unsqueeze(1) + offsets", lambda: start_t.unsqueeze(1)
                                            .add(offsets_cached), 3000)
    bench("  .to(int64)", lambda: start_t.to(torch.int64), 3000)
    bench("  torch.gather", lambda: torch.gather(block_table, 1, idx_cached),
          3000)

    def align_math_cached():
        start = (seq_lens - 1) // block_size
        start.clamp_(min=0)
        idx = (start.unsqueeze(1) + offsets_cached).to(torch.int64)
        return torch.gather(block_table, 1, idx)

    cached_us = bench("same, arange hoisted out", align_math_cached, 3000)

    @triton.jit
    def gather_kernel(table_ptr, seq_ptr, out_ptr, reqs: tl.constexpr,
                      cols: tl.constexpr, bs: tl.constexpr,
                      nspec: tl.constexpr):
        r = tl.program_id(0)
        start = (tl.load(seq_ptr + r) - 1) // bs
        start = tl.maximum(start, 0)
        for j in tl.static_range(nspec + 1):
            v = tl.load(table_ptr + r * cols + start + j)
            tl.store(out_ptr + r * (nspec + 1) + j, v.to(tl.int64))

    out = torch.empty(batch, 1 + num_spec_blocks, device="cuda",
                      dtype=torch.int64)

    def fused():
        gather_kernel[(batch,)](block_table, seq_lens, out, reqs=batch,
                                cols=cols, bs=block_size,
                                nspec=num_spec_blocks)

    fused_us = bench("one fused triton kernel instead", fused, 3000)

    # And the small aten ops the metadata build does per step.
    static = torch.zeros(8, device="cuda", dtype=torch.int32)
    src = torch.ones(4, device="cuda", dtype=torch.int32)
    bench("copy_ of a 4-elem slice", lambda: static[:4].copy_(src, non_blocking=True), 3000)
    bench("fill_ on a slice", lambda: static[4:].fill_(1), 3000)
    bench("slice [:n] (view only)", lambda: static[:6], 3000)


main()
