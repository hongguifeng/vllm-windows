# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound the spec-decode accepted-token count by the state row, not by the query.

Port of upstream HyperQwen's refresh of `patches/vllm-pr50021-gdn-spec-bounds.patch`
(a86a6d49b, PRs #320/#326), re-cut onto the guard our tree already carries.

`num_accepted_tokens` is the *previous* step's count, so it can exceed what this
step verified -- which happens when a tool-call grammar trims the drafts. The
guard we carry bounds it by `seqlen`, but in the varlen branch `seqlen` has
already been narrowed to this request's own length:

    state_len = state_len - (seqlen - (query_end_index - query_start_index))
    seqlen = query_end_index - query_start_index

so the check rejects a legitimate count and the kernel zeroes its conv output
for that step, on every GDN layer. Upstream's refresh captures the launch-wide
`max_query_len` (the width of the spec-decode state row) before that narrowing
and bounds by it, which is the quantity the state lookup is actually indexed
against. Same class of silent output corruption as the int8 negative-scale bug,
and the same tool-call workload that triggers it.

Usage:
    fix_gdn_accepted_bound.py <vllm_package_dir> [<vllm_package_dir> ...]
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

MARKER = "max_seqlen = seqlen  # [fix_gdn_accepted_bound]"
REL_PATH = Path("model_executor/layers/mamba/ops/causal_conv1d.py")

ROW_ANCHOR = """    if IS_VARLEN:
"""

ROW_PATCHED = """    max_seqlen = seqlen  # [fix_gdn_accepted_bound] launch-wide
    # max_query_len: the width of the spec-decode state row. The varlen branch
    # below narrows `seqlen` to this request's own length, while
    # `num_accepted_tokens` is the previous step's count and may legitimately
    # exceed that, so it has to be bounded by the row.
    if IS_VARLEN:
"""

BOUND_ANCHOR = """        if (num_accepted < 1) | (num_accepted > seqlen):
"""

BOUND_PATCHED = """        if (num_accepted < 1) | (num_accepted > max_seqlen):
"""


def patch_one(pkg_dir: Path, *, check_only: bool = False) -> str:
    target = pkg_dir / REL_PATH
    if not target.is_file():
        return f"SKIP   {target} (no such file)"

    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        return f"OK     {target} (already patched)"
    if source.count(BOUND_ANCHOR) == 0:
        return (
            f"SKIP   {target}: no accepted-count guard here -- this tree does not"
            " carry vllm-pr50021-gdn-spec-bounds, so there is nothing to refresh"
        )

    for name, anchor in (
        ("state-row capture", ROW_ANCHOR),
        ("accepted bound", BOUND_ANCHOR),
    ):
        if source.count(anchor) != 1:
            return (
                f"FAIL   {target}: anchor for {name} matched {source.count(anchor)} "
                "times (expected 1) -- upstream changed, re-derive the patch"
            )

    source = source.replace(ROW_ANCHOR, ROW_PATCHED, 1)
    source = source.replace(BOUND_ANCHOR, BOUND_PATCHED, 1)
    if check_only:
        return f"WOULD  {target}"

    target.write_text(source, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(target), doraise=True)
    except py_compile.PyCompileError as exc:
        return f"FAIL   {target}: written but does not compile: {exc}"
    return f"PATCH  {target}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pkg_dirs", nargs="+", help="vllm package dirs to patch")
    ap.add_argument(
        "--check", action="store_true", help="report what would change, write nothing"
    )
    args = ap.parse_args()

    rc = 0
    for raw in args.pkg_dirs:
        pkg_dir = Path(raw)
        if not pkg_dir.is_dir():
            print(f"FAIL   {pkg_dir} is not a directory")
            rc = 2
            continue
        line = patch_one(pkg_dir, check_only=args.check)
        print(line)
        if line.startswith("FAIL"):
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
