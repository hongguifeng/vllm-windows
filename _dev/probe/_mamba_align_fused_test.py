#!/usr/bin/env python3
"""Check the fused "align" block-table path against the aten path.

Two things have to hold before this can be used for real:
  * the output is bit-identical to the six-aten-op path whenever the index is
    inside the block table (clamping must be a no-op in the normal case);
  * it is actually cheaper per call, which is the whole reason for the change.

Cases that run off the end of the table are reported separately because there
the fused kernel clamps to the last column while the aten path reads whatever
is behind the array.
"""
import os
import time

os.environ["VLLM_MAMBA_ALIGN_FUSED"] = "0"

import torch  # noqa: E402

from vllm.v1.attention.backends.utils import (  # noqa: E402
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

MAX_MODEL_LEN = 4096


def spec(block_size: int, num_spec: int) -> MambaSpec:
    return MambaSpec(
        shapes=((2, 8),),
        dtypes=(torch.bfloat16,),
        block_size=block_size,
        num_speculative_blocks=num_spec,
    )


def call(block_table, seq_lens, sp, fused: bool):
    os.environ["VLLM_MAMBA_ALIGN_FUSED"] = "1" if fused else "0"
    return mamba_get_block_table_tensor(
        block_table, seq_lens, sp, "align")


def bench(fn, iters=2000):
    fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters * 1e6
        best = dt if best is None else min(best, dt)
    torch.cuda.synchronize()
    return best


def main() -> None:
    torch.manual_seed(0)
    mismatches = 0
    clamped_cases = 0
    checked = 0

    for block_size in (512, 1024, 2048):
        for num_spec in (0, 1, 2):
            # Production width for align mode includes the speculative blocks
            # (kv_cache_interface.py: MambaSpec.max_num_blocks_per_req), so the
            # index can never leave the row.
            cols = (MAX_MODEL_LEN + block_size - 1) // block_size + num_spec
            for num_requests in (1, 2, 4, 8):
                block_table = torch.randint(
                    0, 1024, (num_requests, cols),
                    dtype=torch.int32, device="cuda")
                seq_lens = torch.randint(
                    0, MAX_MODEL_LEN + 1, (num_requests,),
                    dtype=torch.int32, device="cuda")
                # Force the interesting edges: empty request, full request.
                seq_lens[0] = 0
                seq_lens[-1] = MAX_MODEL_LEN
                sp = spec(block_size, num_spec)
                aten = call(block_table, seq_lens, sp, fused=False)
                fused = call(block_table, seq_lens, sp, fused=True)
                checked += 1
                start = (seq_lens.cpu() - 1) // block_size
                overflow = (start + num_spec).max().item() >= cols
                clamped_cases += int(overflow)
                same = torch.equal(aten, fused)
                dtype_ok = aten.dtype == fused.dtype
                if not (same and dtype_ok):
                    mismatches += 1
                    tag = ("block_size=%d num_spec=%d reqs=%d overflow=%s "
                           "dtype %s->%s") % (
                        block_size, num_spec, num_requests, overflow,
                        aten.dtype, fused.dtype)
                    print("MISMATCH", tag)
                    print("  aten:", aten.flatten().tolist()[:12])
                    print("  fused:", fused.flatten().tolist()[:12])

    # What the clamp does when a table is genuinely too narrow: the aten path
    # dies with a device-side assert there, so it is not callable. Only the
    # fused kernel is exercised, and it stays inside the array.
    narrow_cols = MAX_MODEL_LEN // 1024
    sp = spec(1024, 1)
    block_table = torch.randint(0, 1024, (2, narrow_cols), dtype=torch.int32,
                                device="cuda")
    seq_lens = torch.tensor([MAX_MODEL_LEN, MAX_MODEL_LEN], dtype=torch.int32,
                            device="cuda")
    fused_only = call(block_table, seq_lens, sp, fused=True)
    print(f"narrow table (width {narrow_cols}, needs index "
          f"{(MAX_MODEL_LEN - 1) // 1024 + 1}): fused clamped to "
          f"{fused_only.flatten().tolist()}")

    print(f"checked={checked} cases at production width, "
          f"{clamped_cases} of them would have left the row")
    print("mismatches:", mismatches, "=>",
          "IDENTICAL" if mismatches == 0 else "DIFFERENT")

    # Timing at the shape decode actually uses.
    block_size = 1024
    cols = MAX_MODEL_LEN // block_size
    sp = spec(block_size, 1)
    block_table = torch.randint(0, 1024, (4, cols), dtype=torch.int32,
                                device="cuda")
    seq_lens = torch.randint(0, MAX_MODEL_LEN, (4,), dtype=torch.int32,
                             device="cuda")
    aten_us = bench(lambda: call(block_table, seq_lens, sp, fused=False))
    fused_us = bench(lambda: call(block_table, seq_lens, sp, fused=True))
    print(f"per call: aten {aten_us:7.2f} us   fused {fused_us:7.2f} us   "
          f"saving {aten_us - fused_us:6.2f} us/call")


if __name__ == "__main__":
    main()
