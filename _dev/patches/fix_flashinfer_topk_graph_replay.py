"""Keep FlashInfer's radix top-k out of the DFlash2 drafter's CUDA graph.

``LogitsProcessor.get_top_k_tokens`` -- called once per draft step from
``DFlash2Qwen3ForCausalLM.compute_candidates`` -- picks the drafter's k candidate
token ids over the whole vocabulary.  ``_topk`` prefers ``flashinfer.top_k`` over
``torch.topk`` because the radix kernel is roughly twice as fast across 248320
entries.  That kernel must not run from inside the captured drafter graph.

At this model's shapes (vocab 248320, selector_top_k 16) flashinfer's heuristic
picks ``RadixTopKMultiCTA``: ceil(248320 / max_chunk_elements) = 11 CTAs per row
group, synchronised through a software barrier whose counters live in a
process-wide workspace (``_get_cache_buf("radix_topk_row_states_*")``), with
``torch.empty`` output buffers and no guarantee that every output slot is
written.  Replaying that from a CUDA graph intermittently yields candidate ids
that are simply wrong, and sometimes out of range -- the slots nobody wrote are
whatever ``torch.empty`` left behind.  Measured:

* ``_repro_topk_graph.py``: one wrong row in 2400 graph replays of a graph whose
  only node is ``flashinfer.top_k``; the same loop run eagerly is clean.
* In the engine, an in-graph guard on ``candidate_ids`` (``_clamp_exp.py``)
  counted ``neg=15 / ge=23`` out-of-range ids over four passes of
  ``_repro_async_tower.ps1 -Clamp -Repeat 4``, while 20150 eager steps
  (``DFLASH2_NO_DRAFT_CUDAGRAPH=1``) produced none.

Those ids index the ``[vocab, rank]`` codebooks in ``qwen3_dflash2._score_edges``
(``successor_table[candidate_ids]`` / ``predecessor_table[predecessor_ids]``).  An
out-of-range index trips inductor's indirect-index bounds assert, which is a
*device-side* assert: the engine dies and every in-flight request 500s.

``torch.topk`` is stateless and replays safely.  It costs 1.6-2.3x the radix
kernel on these shapes -- +0.06 ms (1 request) to +0.18 ms (8 requests) per draft
step measured from inside a graph, see ``_bench_topk.py`` -- which is well under
1% of a decode step, and it is already what ``_topk`` falls back to when
flashinfer is unavailable.

The env var is ``DFLASH2_``-prefixed on purpose: ``LogitsProcessor.get_top_k_tokens``
has exactly one caller (the DFlash2 candidate selector), and a ``VLLM_`` prefix
would trip ``envs.validate_environ``.  Set ``DFLASH2_TOPK_IMPL=flashinfer`` to get
the old fast-but-crash-prone path back.

Python-only, so no rebuild is needed.  Patch both the installed package and the
repo source, or a reinstall silently reintroduces the bug.

Usage:
    fix_flashinfer_topk_graph_replay.py <vllm_package_dir> [<vllm_package_dir> ...]
e.g.
    fix_flashinfer_topk_graph_replay.py .venv/Lib/site-packages/vllm vllm
"""

import os
import sys

TARGET = os.path.join("model_executor", "layers", "logits_processor.py")

MARKER = "DFLASH2_TOPK_IMPL"

OLD_IMPORT = "from collections.abc import Callable\nfrom functools import cache\n"
NEW_IMPORT = "import os\nfrom collections.abc import Callable\nfrom functools import cache\n"

OLD_GUARD = """    if not current_platform.is_cuda():
        return None
    if not has_flashinfer():
"""

NEW_GUARD = """    # [flashinfer-topk-graph] Default to torch.topk.  The radix kernel is not
    # safe to replay from a captured CUDA graph at this vocabulary size: it
    # syncs 11 CTAs per row group through a software barrier in a process-wide
    # workspace and writes into torch.empty buffers, so a desynchronised round
    # hands back wrong -- and sometimes out-of-range -- candidate ids.  Those
    # ids index a [vocab, rank] codebook, and an out-of-range index is a
    # device-side assert that kills the engine.  See
    # fix_flashinfer_topk_graph_replay.py for the measurements.
    if os.environ.get("DFLASH2_TOPK_IMPL", "torch") != "flashinfer":
        return None
    if not current_platform.is_cuda():
        return None
    if not has_flashinfer():
"""

patched = 0
for pkg_dir in sys.argv[1:]:
    path = os.path.join(pkg_dir, TARGET)
    if not os.path.exists(path):
        print(f"skip (missing): {path}")
        continue

    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()

    if MARKER in content:
        print(f"already patched: {path}")
        continue

    if OLD_GUARD not in content:
        print(f"WARN: _flashinfer_topk guard anchor not found, left untouched: {path}")
        continue

    new_content = content.replace(OLD_GUARD, NEW_GUARD, 1)
    if NEW_IMPORT not in new_content:
        if OLD_IMPORT not in new_content:
            print(f"WARN: import anchor not found, left untouched: {path}")
            continue
        new_content = new_content.replace(OLD_IMPORT, NEW_IMPORT, 1)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(new_content)
    print(f"patched: {path}")
    patched += 1

print(f"done, {patched} file(s) patched")
