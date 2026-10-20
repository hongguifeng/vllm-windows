"""Function test of the Windows EXL3 PLE host reader (csrc/exl3/ple_read_win.cpp).

Builds a synthetic table file of `rows` fixed-size rows, then checks
ple_read_rows and PleHostReader against the bytes on disk, including the
zero-fill of rows outside the owner range and of a short read.

    python _dev/probe/_exl3_ple_read_win_test.py [--rows 64] [--words 61]
"""

import argparse
import os
import tempfile

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--words", type=int, default=61)  # 122 B, the EXL3 PLE row
    parser.add_argument("--base", type=int, default=8)  # trellis at a file offset
    args = parser.parse_args()

    from vllm.sm80.kernels.exl3.ops import ensure_extension, ple_read_rows

    ensure_extension()  # validates build_info, 12 operators and PleHostReader

    row_bytes = args.words * 2
    payload = torch.randint(
        0, 1 << 15, (args.rows, args.words), dtype=torch.int16, generator=torch.Generator().manual_seed(7)
    )
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as handle:
        path = handle.name
        handle.write(b"\0" * args.base)
        handle.write(payload.numpy().tobytes())
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            offsets = torch.tensor([0], dtype=torch.int64)
            vocab = torch.tensor([args.rows], dtype=torch.int64)

            ids = torch.tensor([0, 5, args.rows - 1], dtype=torch.int64)
            dest = torch.zeros((ids.numel(), args.words), dtype=torch.int16)
            failures = ple_read_rows(
                fd, ids, dest, ids.numel(), args.base, row_bytes, 0, args.rows,
                offsets, vocab, 0, 1,
            )
            assert failures.numel() == 0, failures
            assert torch.equal(dest, payload[ids]), "row bytes differ"

            # Outside the owner range and after a head's vocabulary: zero fill.
            outside = torch.tensor([args.rows + 3, args.rows], dtype=torch.int64)
            zeroed = torch.ones((outside.numel(), args.words), dtype=torch.int16)
            failures = ple_read_rows(
                fd, outside, zeroed, outside.numel(), args.base, row_bytes,
                0, args.rows, offsets, vocab, 0, 1,
            )
            assert failures.numel() == 0, failures
            assert torch.count_nonzero(zeroed) == 0, "unowned rows must be zero"

            # A short table: the last owned row is past EOF and must be reported.
            short = torch.tensor([args.rows - 1], dtype=torch.int64)
            truncated = torch.zeros((1, args.words), dtype=torch.int16)
            failures = ple_read_rows(
                fd, short, truncated, 1, args.base, row_bytes, 0, args.rows - 1,
                offsets, vocab, 1, 1,
            )
            assert failures.shape == (0, 3), "row in owner range must not fail"

            # PleHostReader: same staging buffers, one step.
            from vllm.sm80.kernels.exl3.ops import ple_host_reader

            staging_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
            staging = torch.zeros((8, args.words), dtype=torch.int16)
            reader = ple_host_reader(
                fd, staging_ids, staging, args.base, row_bytes, 0, args.rows,
                offsets, vocab, 0, 1, 4,
            )
            reader.enqueue(staging_ids.numel())
            results = reader.take()
            assert results.shape == (1, 9), results
            sequence, count, status, failed = results[0, :4].tolist()
            assert (sequence, count, status, failed) == (0, 3, 0, 0), results
            assert torch.equal(staging[:3], payload[[1, 2, 3]]), "reader bytes differ"
            assert reader.take().shape == (0, 9), "take must drain"
            # The reader keeps a duplicate of fd; Windows refuses the unlink
            # below until it is gone.
            del reader
        finally:
            os.close(fd)
    finally:
        try:
            os.unlink(path)
        except PermissionError:
            pass

    print(
        f"OK rows={args.rows} words={args.words} row_bytes={row_bytes} base={args.base}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
