#!/usr/bin/env python3
"""Check when a prefix slice equals a boolean mask select in the GDN metadata path.

The proposed fast path replaces ``tensor[cpu_bool_mask]`` with ``tensor[:n]`` to
avoid the device synchronization that boolean indexing forces. Boolean select
returns a new compact tensor while a slice returns a view, so this compares shape,
values, contiguity and strides case by case, including the cases that must reject
the fast path.

Run with CUDA_VISIBLE_DEVICES pointing at a free device and no server running.
"""
import torch


def compare(name: str, block_table: torch.Tensor, mask: torch.Tensor, n: int,
            k: int, accepted: torch.Tensor) -> None:
    """Print whether the masked select and the prefix slice agree."""
    selected = block_table[mask, :k]
    sliced = block_table[:n, :k]
    values_ok = selected.shape == sliced.shape and bool(
        torch.equal(selected, sliced))
    accepted_masked = accepted[mask]
    accepted_sliced = accepted[:n]
    acc_ok = accepted_masked.shape == accepted_sliced.shape and bool(
        torch.equal(accepted_masked, accepted_sliced))
    print(
        f"{name:28s} shape {tuple(selected.shape)} vs {tuple(sliced.shape)} "
        f"values {'OK ' if values_ok else 'DIFF'} "
        f"| accepted {'OK ' if acc_ok else 'DIFF'} "
        f"| masked contig={selected.is_contiguous()} slice contig={sliced.is_contiguous()} "
        f"| slice stride={sliced.stride()} is_view={sliced._base is not None}"
    )


def main() -> None:
    device = torch.cuda.current_device()
    print(f"device: {torch.cuda.get_device_name(device)}")
    mask_dtype = dict(dtype=torch.bool, device="cpu")

    cases = []

    # All rows are real speculative requests and there is no graph padding.
    block = torch.arange(4 * 6, dtype=torch.int32, device=device).reshape(4, 6)
    mask = torch.tensor([True, True, True, True], **mask_dtype)
    cases.append(("all true, no padding", block, mask, 4, 3))

    # A true prefix followed by padded rows, which is the expected convention.
    block = torch.arange(8 * 6, dtype=torch.int32, device=device).reshape(8, 6)
    mask = torch.tensor([True, True, True, False, False, False, False, False],
                        **mask_dtype)
    cases.append(("true prefix plus padding", block, mask, 3, 3))

    # A hole in the middle must reject the fast path.
    mask = torch.tensor([True, False, True, True], **mask_dtype)
    cases.append(("false row in the middle", block[:4], mask, 3, 3))

    # A zero length row in the prefix must reject it as well.
    mask = torch.tensor([True, True, False, False], **mask_dtype)
    cases.append(("short prefix, two padded rows", block[:4], mask, 2, 3))

    # Wider rows make the slice a non contiguous view.
    wide = torch.arange(4 * 10, dtype=torch.int32, device=device).reshape(4, 10)
    mask = torch.tensor([True, True, True, True], **mask_dtype)
    cases.append(("wide rows, k smaller than width", wide, mask, 4, 3))

    for name, block_table, mask, n, k in cases:
        # In the real path num_accepted_tokens has one entry per request, so it
        # is the same length as the mask.
        accepted = torch.arange(block_table.shape[0], dtype=torch.int32,
                                device=device)
        compare(name, block_table, mask, n, k, accepted)

    print("\nA prefix slice is only a safe substitute when the mask is exactly")
    print("True for rows 0..n-1 and False for every row after that.")


if __name__ == "__main__":
    main()
