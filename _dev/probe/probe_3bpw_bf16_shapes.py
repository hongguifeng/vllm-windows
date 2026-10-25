"""Authoritative BF16 (unquantized) 2-D weight inventory of the 3bpw checkpoint.

The previous survey claimed `self_attn.*` is left in fp16 by the 2-bit block. Read
the actual dtypes before planning around that: AutoGPTQ 2-bit leaves `qweight`
I32 + `scales` F16, so a layer is unquantized only where the raw `.weight` is BF16.

Prints, for every distinct (module suffix, B^-)(shape, dtype) found:
  - which module suffixes are BF16 and how many layer instances,
  - which are quantized,
  - per-forward GEMM call count for the BF16 set.

  .venv\\Scripts\\python.exe _dev\\probe\\probe_3bpw_bf16_shapes.py
"""

import collections
import json
import os

from safetensors import safe_open

CKPT = r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
OUT = os.path.join(os.path.dirname(__file__), "_3bpw_bf16_shapes.json")


def load_meta(wanted, index):
    """Read only the safetensors JSON headers (8-byte length + JSON).

    `safe_open` mmaps the shard, and mmapping all 11 shards at once trips
    os error 1455 (paging file) on this box; the header holds dtype and shape.
    """
    by_shard = collections.defaultdict(set)
    for k in wanted:
        by_shard[index[k]].add(k)
    meta = {}
    for shard, keys in by_shard.items():
        with open(os.path.join(CKPT, shard), "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        for k in keys:
            entry = header.get(k)
            if entry is None:
                continue
            meta[k] = (entry["dtype"], tuple(entry["shape"]))
    return meta


def suffix(name):
    """Strip the layer index: `model.language_model.layers.7.mlp.gate.weight`
    -> `L.mlp.gate.weight`; keep the mtp / model-level prefix distinct."""
    for marker, tag in (
        ("model.language_model.layers.", "L"),
        ("mtp.layers.", "MTP"),
    ):
        if marker in name:
            head, rest = name.split(marker, 1)
            idx, _, tail = rest.partition(".")
            return tag + "." + tail
    return name


def main():
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    print(f"checkpoint tensors: {len(index)}")

    # Every 2-D bf16 `.weight`, plus the quantized marker for the rest.
    wanted = [k for k in index if k.endswith(".weight")]
    meta = load_meta(wanted, index)

    bf16 = collections.defaultdict(lambda: collections.Counter())
    quant = collections.Counter()
    for name, (dtype, shape) in meta.items():
        base = name[: -len(".weight")]
        if index.get(base + ".qweight") is not None:
            quant[suffix(name)] += 1
        if dtype == "BF16" and len(shape) == 2:
            bf16[suffix(name)][shape] += 1

    print("\n=== quantized (qweight present) 2-D modules, by suffix ===")
    for s, c in sorted(quant.items()):
        print(f"  {s:<58} x{c}")

    print("\n=== BF16 2-D modules, by suffix ===")
    for s in sorted(bf16):
        for shape, c in bf16[s].items():
            n, k = shape
            print(f"  {s:<58} N={n:<6} K={k:<6} x{c:<4} {n * k * 2 / 2**20:8.2f} MiB")

    # Per-layer call count: one instance of each L.* suffix runs per layer pass.
    per_layer = sum(c for s, rows in bf16.items() if s.startswith("L.")
                    for c in rows.values())
    mtp = sum(c for s, rows in bf16.items() if s.startswith("MTP.")
              for c in rows.values())
    model_level = sum(c for s, rows in bf16.items() if not s.startswith(("L.", "MTP."))
                      for c in rows.values())
    print(f"\nBF16 2-D GEMMs: {per_layer} per layer pass, {mtp} in the MTP layer, "
          f"{model_level} at model level")

    shapes = collections.Counter()
    for s, rows in bf16.items():
        for shape, c in rows.items():
            shapes[shape] += c
    bytes_all = sum(n * k * 2 * c for (n, k), c in shapes.items())
    print(f"distinct BF16 (N, K): {len(shapes)}")
    print(f"BF16 weight bytes read by one full layer pass: {bytes_all / 2**20:.1f} MiB "
          f"({bytes_all / 2**30:.2f} GiB)")

    print("\n=== distinct BF16 (N, K), by bytes per call ===")
    for (n, k), c in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
        print(f"  N={n:<6} K={k:<6} calls={c:<4} {n * k * 2 / 2**20:8.2f} MiB")

    with open(OUT, "w") as f:
        json.dump({"shapes": [[n, k, c] for (n, k), c in sorted(shapes.items())]}, f,
                  indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
