"""Build the Qwen4Exp n-gram module with EXL3 on Windows and decode real rows.

Single process, TP=1, one 170HX. Constructs Qwen4ExpNGramEmbedding with the
EXL3 config (so the class selection hook picks Exl3PLEEmbedding), joins the
real ngram_embedding.safetensors, runs start_prefetch + forward, and compares
the output with a direct ple_decode_out of the same rows.

    python _dev/probe/_exl3_ple_module_win_test.py "D:\\models\\Qwen3.8-Flash-Next-exl3"
"""

import argparse
import os
from pathlib import Path

import torch

KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.trellis"
PORT = "tcp://127.0.0.1:29577"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=4)
    args = parser.parse_args()
    path = str(args.model)

    from vllm.config import LoadConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    torch.cuda.set_device(args.device)
    model_config = ModelConfig(
        model=path, tokenizer=path, dtype="bfloat16", max_model_len=4096, seed=0
    )
    vllm_config = VllmConfig(
        model_config=model_config, load_config=LoadConfig(load_format="exl3")
    )
    quant = vllm_config.quant_config
    print("quant_config", type(quant).__name__)
    if quant.manifest is None:
        quant.maybe_update_config(path, hf_config=model_config.hf_config)

    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding

    text_config = model_config.hf_text_config
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29577", RANK="0", WORLD_SIZE="1"
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
        with torch.device(f"cuda:{args.device}"):
            ngram = Qwen4ExpNGramEmbedding(
                text_config,
                int(text_config.ple_embed_dim),
                0,
                32,
                data_parallel_rank=0,
                prefix="model.language_model.layers.1.ple.ple_embedding",
                quant_config=quant,
                params_dtype=torch.bfloat16,
            )
    embedding = ngram.ngram_embedding
    print(
        "embedding",
        type(embedding).__name__,
        "placement",
        embedding.placement,
        "heads",
        ngram.ngram_heads,
        "dim",
        ngram.head_dim,
    )
    assert type(embedding).__name__ == "Exl3PLEEmbedding", type(embedding)

    from vllm.sm80.kernels.exl3.utils import ple_layout, read_headers, read_metadata

    table_path = args.model / "ngram_embedding.safetensors"
    layout = ple_layout(
        read_headers(table_path),
        read_metadata(table_path),
        KEY.removesuffix(".trellis"),
    )
    print(
        "table rows",
        layout.rows,
        "owner",
        embedding.shard_indices.org_vocab_start_index,
        embedding.shard_indices.org_vocab_end_index,
        "org_vocab",
        embedding.org_vocab_size,
    )
    consumed = embedding.setup_file(
        str(table_path),
        KEY,
        ngram.ngram_heads_offsets,
        ngram.ngram_heads_vocab_sizes,
        ngram.layer_multipliers,
    )
    print("setup_file consumed", len(consumed), "keys")

    device = torch.device("cuda", args.device)
    ids = torch.randint(
        0, 1000, (args.tokens, ngram.ngram_heads), dtype=torch.int64, device=device
    )
    hidden = torch.zeros((args.tokens, 1), dtype=torch.bfloat16, device=device)
    embedding.start_prefetch(hidden, ids)
    out = embedding.forward(hidden)
    torch.cuda.synchronize(device)
    print("forward", tuple(out.shape), out.dtype, "finite", bool(out.isfinite().all()))
    assert out.shape == (args.tokens, ngram.ngram_heads * ngram.head_dim)

    # Same rows through the raw kernel: the embedding must be a bit-exact view.
    from vllm.sm80.kernels.exl3.ops import ensure_extension

    ensure_extension()
    reader = embedding._reader
    flat = ids.reshape(-1)
    packed = torch.zeros(
        (flat.numel(), reader.layout.words), dtype=torch.int16, device=device
    )
    host = bytearray(flat.numel() * reader.row_bytes)
    failed = reader.read_rows(flat.tolist(), memoryview(host))
    assert not failed, failed
    packed.copy_(torch.frombuffer(host, dtype=torch.int16).reshape(packed.shape))
    expected = torch.empty(
        (flat.numel(), ngram.head_dim), dtype=torch.bfloat16, device=device
    )
    torch.ops._exl3_C.ple_decode_out(
        packed,
        flat,
        embedding.head_bias,
        embedding.head_offsets,
        embedding.head_vocab_sizes,
        reader.owner_lo,
        reader.owner_hi,
        reader.layout.bits,
        expected,
    )
    torch.cuda.synchronize(device)
    got = out.view(args.tokens, ngram.ngram_heads, ngram.head_dim).reshape(
        flat.numel(), ngram.head_dim
    )
    mismatch = (got != expected).sum().item()
    print(f"raw kernel comparison: {got.numel()} values, mismatches={mismatch}")
    if mismatch:
        same_packed = torch.empty_like(expected)
        torch.ops._exl3_C.ple_decode_out(
            embedding._packed[: flat.numel()],
            flat,
            embedding.head_bias,
            embedding.head_offsets,
            embedding.head_vocab_sizes,
            reader.owner_lo,
            reader.owner_hi,
            reader.layout.bits,
            same_packed,
        )
        torch.cuda.synchronize(device)
        print("vs embedding's own packed rows:", (got != same_packed).sum().item())
        row_bytes = reader.row_bytes
        host_view = bytes(
            embedding._host_rows[: flat.numel()].contiguous().numpy().tobytes()
        )
        print(
            "row bytes equal:",
            host_view == bytes(host[: flat.numel() * row_bytes]),
        )
        bad = (got != expected).any(dim=-1).view(args.tokens, ngram.ngram_heads)
        print("bad (token, head) rows", bad.nonzero().tolist())
        where = tuple((got != expected).nonzero()[0].tolist())
        print(
            "first mismatch",
            where,
            "got",
            got[where].item(),
            "want",
            expected[where].item(),
        )
    assert mismatch == 0
    print("OK EXL3 PLE embedding matches the raw decode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
