"""Exercise the Windows EXL3 PLE row reader against the real checkpoint.

Opens ngram_embedding.safetensors through Exl3PLERowReader (the class the
model uses), reads its auxiliaries and a few table rows, and compares the
bytes with a plain file read at the same offsets. No GPU, no model.

    python _dev/probe/_exl3_ple_reader_win_test.py "D:\\models\\Qwen3.8-Flash-Next-exl3"
"""

import argparse
import os
from pathlib import Path

KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.trellis"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()

    from vllm.models.qwen4_exp.nvidia.exl3_ple import (
        Exl3PLERowReader,
        pread_exact_into,
    )

    path = args.model / "ngram_embedding.safetensors"
    reader = Exl3PLERowReader(str(path), KEY, 0, 1 << 20)
    try:
        layout = reader.layout
        print(
            f"layout rows={layout.rows} dim={layout.dim} words={layout.words} "
            f"row_bytes={reader.row_bytes} heads={layout.heads} base={reader.base}"
        )
        assert reader.row_bytes == 122, reader.row_bytes
        assert reader.page_cache_residency() == (0, 0), "Windows has no mincore"

        # Auxiliary tensor read equals the bytes at its offset.
        header = layout.auxiliaries[0]
        data = reader.read_tensor(header)
        with open(path, "rb") as handle:
            handle.seek(header.offset)
            assert bytes(data) == handle.read(header.nbytes), "auxiliary bytes differ"

        # Table rows equal the bytes at base + row * row_bytes.
        ids = list(range(args.rows)) + [5000, 100000]
        target = bytearray(len(ids) * reader.row_bytes)
        failed = reader.read_rows(ids, memoryview(target))
        assert not failed, failed
        with open(path, "rb") as handle:
            for index, row in enumerate(ids):
                handle.seek(reader.base + row * reader.row_bytes)
                want = handle.read(reader.row_bytes)
                got = bytes(target[index * reader.row_bytes : (index + 1) * reader.row_bytes])
                assert got == want, f"row {row} differs"

        # The read helper must not move the descriptor.
        with open(path, "rb") as handle:
            fd = handle.fileno()
            before = os.lseek(fd, 123, os.SEEK_SET)
            chunk = bytearray(16)
            pread_exact_into(fd, memoryview(chunk), reader.base)
            assert os.lseek(fd, 0, os.SEEK_CUR) == before, "seek position moved"
    finally:
        reader.close()

    print(f"OK rows={args.rows} row_bytes={reader.row_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
