"""Let a W4A16-quantized DFlash2 drafter load: dequantize the context-KV rows.

``precompute_and_store_context_kv`` needs the drafter's KV-projection weights as a
dense matrix -- ``_build_context_kv_buffers`` stacks ``qkv_proj.weight[q_size:]``
from every draft layer into one fused GEMM that turns the target's hidden states
(``dflash_config.target_layer_ids`` = 5/19/33/47/61) into context KV.

A pack-quantized (compressed-tensors W4A16) drafter has no ``.weight``: the
parameters are ``weight_packed`` + ``weight_scale``. So loading dies with

    AttributeError: 'QKVParallelLinear' object has no attribute 'weight'

which is why ``Qwen3.8-27B-DFlash2-W4A16`` -- the 1.19 GB requantized drafter that
fits beside a 27B target on a 24 GB card -- will not load on stock 0.29: the
native DFlash2 code assumes a bf16 drafter. The fork already solved this; the
helper below is lifted verbatim from its ``patches/dflash2-backport.patch``. That
patch is marked RETIRED because its DFlash2 *core* went native in 0.28, and this
hunk was retired along with it, even though 0.29 still needs it.

The dequantization runs inside ``load_weights``, i.e. before the Marlin repack, so
``weight_packed``/``weight_scale`` are still in the plain checkpoint layout. Only
the drafter's k/v rows are materialized dense (a few MB); its q/k/v/o and mlp
linears stay int4, which is the whole point of the requantization.

Python-only, so no rebuild is needed. Patch both the installed package and the
repo source, or a reinstall silently reintroduces the bug.

Usage:
    fix_dflash2_dense_kv.py <vllm_package_dir> [<vllm_package_dir> ...]
e.g.
    fix_dflash2_dense_kv.py .venv/Lib/site-packages/vllm vllm
"""

import os
import sys

TARGET = "qwen3_dflash.py"

HELPER = '''def _dense_kv_rows(attn: nn.Module) -> torch.Tensor:
    """Rows [q_size:] of the qkv projection as a dense bf16 matrix. For a compressed-tensors
    W4A16/W8A16 (pack-quantized, symmetric, group) qkv_proj this is called from load_weights,
    i.e. before the Marlin repack, so weight_packed/weight_scale are still in the plain
    checkpoint layout and can be dequantized here."""
    qkv = attn.qkv_proj
    w = getattr(qkv, "weight", None)
    if w is not None and w.dim() == 2:
        return w[attn.q_size :]
    packed, scale = qkv.weight_packed, qkv.weight_scale
    # (weight_shape holds only the last-loaded shard of a fused qkv; use the tensors.)
    out_f, in_f = int(packed.shape[0]), int(qkv.input_size)
    bits = 32 * packed.shape[1] // in_f
    from compressed_tensors.compressors.pack_quantized.base import unpack_from_int32
    q = unpack_from_int32(packed.data, bits, torch.Size([out_f, in_f]), packed_dim=1)
    group = in_f // scale.shape[1]
    dense = (q.to(torch.float32).reshape(out_f, in_f // group, group)
             * scale.to(torch.float32)[..., None]).reshape(out_f, in_f)
    return dense.to(scale.dtype if scale.dtype.is_floating_point else torch.bfloat16)[attn.q_size :]


'''

ANCHOR = "@support_torch_compile\nclass DFlashQwen3Model(nn.Module):\n"

OLD_LINE = "        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]\n"
NEW_LINE = "        kv_weights = [_dense_kv_rows(a) for a in layers_attn]\n"

patched = 0
for pkg_dir in sys.argv[1:]:
    path = os.path.join(pkg_dir, "model_executor", "models", TARGET)
    if not os.path.exists(path):
        print(f"skip (missing): {path}")
        continue

    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()

    if NEW_LINE in content and "_dense_kv_rows(attn" in content:
        print(f"already patched: {path}")
        continue

    if OLD_LINE not in content:
        print(f"WARN: kv_weights anchor not found, left untouched: {path}")
        continue
    if ANCHOR not in content:
        print(f"WARN: class anchor not found, left untouched: {path}")
        continue

    content = content.replace(ANCHOR, HELPER + ANCHOR, 1)
    content = content.replace(OLD_LINE, NEW_LINE, 1)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    print(f"patched: {path}")
    patched += 1

print(f"done, {patched} file(s) patched")
