"""Report which tensors live in each safetensors shard of a checkpoint."""
import glob
import json
import os
import sys


def main() -> None:
    d = (sys.argv[1] if len(sys.argv) > 1 else
         r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP")
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    print(f"{len(files)} shards under {d}")
    total_ple = 0
    for path in files:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        names = [k for k in header if k != "__metadata__"]
        ple = [k for k in names if ".ngram_embedding.shard_" in k]
        def nbytes(k: str) -> int:
            shape = header[k]["shape"]
            n = 1
            for dim in shape:
                n *= dim
            dt = header[k]["dtype"]
            size = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4,
                    "U8": 1, "BOOL": 1}.get(dt, 2)
            return n * size

        bytes_total = sum(nbytes(k) for k in names)
        ple_bytes = sum(nbytes(k) for k in ple)
        total_ple += ple_bytes
        print(f"{os.path.basename(path)}: {len(names)} tensors, "
              f"{len(ple)} PLE, {bytes_total / 2**30:.2f} GiB "
              f"({ple_bytes / 2**30:.2f} GiB PLE), "
              f"pure_ple_shard={bool(ple) and len(ple) == len(names)}")
        if ple and len(ple) != len(names):
            others = [k for k in names if k not in ple][:3]
            print(f"    mixed; non-PLE examples: {others}")
    print(f"PLE bf16 total: {total_ple / 2**30:.2f} GiB")


main()
