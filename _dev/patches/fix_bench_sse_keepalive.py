# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stop the benchmark client from stripping SSE chunks.

Port of upstream HyperQwen `patches/bench-sse-keepalive.patch` (704a6bedb, PR
#226) onto the in-tree 0.29.1 client.

Every request function starts its chunk loop with:

    chunk_bytes = chunk_bytes.strip()
    if not chunk_bytes:
        continue

A network chunk is not one SSE message -- the blank line between messages is
data. Stripping deletes that blank line, so `StreamedResponseHandler` can no
longer split, and a `: keep-alive` comment gets glued to every message after it.
The combined string starts with ":", so the comment skip two of the three
request functions already have throws the real payload away, and the request is
recorded as "Never received a valid chunk to calculate TTFT".

With the launchers' `--sse-keep-alive-interval 30` that kills every prompt whose
prefill takes longer than 30 s. We measured the same cliff here (AB_WSL_VS_WINDOWS
gotcha 4.1): 32,768 tokens at TTFT 17.7 s pass, 49,152 at 33 s and 65,536 at 48 s
fail -- which is why the prefill matrix here stops at 32k.

Usage:
    fix_bench_sse_keepalive.py <vllm_package_dir> [<vllm_package_dir> ...]
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

import regex as re

MARKER = "[fix_bench_sse_keepalive]"
REL_PATH = Path("benchmarks/lib/endpoint_request_func.py")
EXPECTED_STRIP_SITES = 3

STRIP_RE = re.compile(
    r"^(\s+)chunk_bytes = chunk_bytes\.strip\(\)\n"
    r"\1if not chunk_bytes:\n",
    re.M,
)


def _replacement(indent: str) -> str:
    return (
        f"{indent}# [fix_bench_sse_keepalive] the SSE separators are data: stripping"
        f" a\n"
        f"{indent}# network chunk deletes the blank line between messages and glues"
        f" a\n"
        f"{indent}# keep-alive comment onto every message after it.\n"
        f"{indent}if not chunk_bytes.strip():\n"
    )


AUDIO_ANCHOR = """                            if type(message) is bytes:
                                message = message.decode("utf-8")
                            chunk = message.removeprefix("data: ")
"""

AUDIO_PATCHED = """                            if type(message) is bytes:
                                message = message.decode("utf-8")
                            # [fix_bench_sse_keepalive] SSE comments (keep-alive
                            # pings) are not JSON, as in the two request functions
                            # above.
                            if message.startswith(":"):
                                continue
                            chunk = message.removeprefix("data: ")
"""


def patch_one(pkg_dir: Path, *, check_only: bool = False) -> str:
    target = pkg_dir / REL_PATH
    if not target.is_file():
        return f"SKIP   {target} (no such file)"

    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        return f"OK     {target} (already patched)"

    sites = len(STRIP_RE.findall(source))
    if sites != EXPECTED_STRIP_SITES:
        return (
            f"FAIL   {target}: found {sites} strip-then-skip sites "
            f"(expected {EXPECTED_STRIP_SITES}) -- upstream changed, re-derive"
        )
    if source.count(AUDIO_ANCHOR) != 1:
        return (
            f"FAIL   {target}: audio-request anchor matched "
            f"{source.count(AUDIO_ANCHOR)} times (expected 1) -- upstream changed, "
            "re-derive"
        )

    source = STRIP_RE.sub(lambda match: _replacement(match.group(1)), source)
    source = source.replace(AUDIO_ANCHOR, AUDIO_PATCHED, 1)
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
