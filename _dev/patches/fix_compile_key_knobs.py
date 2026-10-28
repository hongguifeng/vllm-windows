# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep runtime-only env knobs out of the torch.compile cache key.

Port of upstream HyperQwen `patches/compile-key-runtime-knobs.patch` (2206e2f28,
issue #183) onto 0.29.1, listing the knobs *this* tree registers.

`envs.compile_factors()` starts from every registered `VLLM_*` variable and drops
the ones in `ignored_factors`; the result is hashed into the torch.compile cache
key (`compilation/backends.py`, `compilation/caching.py`). A knob that exists
only to be read through `os.environ` at run time -- a watchdog timeout, a weight
repack staging toggle, a regex that picks which GEMMs run in int8 -- therefore
invalidates the whole compile cache every time its value changes, and a cold
compile costs minutes for a setting no compiled graph ever reads. Upstream
measured it at ~227 s against ~150 s.

The names below are the ones this install actually defines, checked against the
file before editing: `VLLM_ENGINE_STALL_SENTINEL_S` (a daemon-thread timeout),
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (a block-manager policy that
`CacheConfig.compute_hash` already leaves out of the hash), `VLLM_MARLIN_REPACK_STAGED`
(a load-time staging buffer) and the two Marlin int8 layer regexes (a load-time
layer selection). `VLLM_MARLIN_INPUT_DTYPE` stays hashed on purpose: it changes
the activation dtype a compiled graph is built around.

Usage:
    fix_compile_key_knobs.py <vllm_package_dir> [<vllm_package_dir> ...]
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

MARKER = "[fix_compile_key_knobs]"
REL_PATH = Path("envs.py")

RUNTIME_ONLY = [
    "VLLM_ENGINE_STALL_SENTINEL_S",
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
    "VLLM_MARLIN_REPACK_STAGED",
    "VLLM_MARLIN_INT8_INCLUDE_RE",
    "VLLM_MARLIN_INT8_EXCLUDE_RE",
]

ANCHOR = """        "VLLM_CACHE_ROOT",
        # Runtime memory-plan persistence; does not affect compiled graphs.
        "VLLM_ENABLE_STARTUP_PLAN",
"""

PATCHED_TAIL = """        # [fix_compile_key_knobs] Knobs this tree registers only to
        # read at run time -- a watchdog timeout, a block-manager policy, a
        # load-time staging buffer, a load-time layer selection. No traced code
        # reads them, but hashing each one meant every change cost a cold
        # torch.compile (~227 s against ~150 s; upstream syv-ai/HyperQwen #183).
"""


def patch_one(pkg_dir: Path, *, check_only: bool = False) -> str:
    target = pkg_dir / REL_PATH
    if not target.is_file():
        return f"SKIP   {target} (no such file)"

    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        return f"OK     {target} (already patched)"
    if source.count(ANCHOR) != 1:
        return (
            f"FAIL   {target}: compile_factors ignore-list anchor matched "
            f"{source.count(ANCHOR)} times (expected 1) -- upstream changed, "
            "re-derive the patch"
        )
    present = [n for n in RUNTIME_ONLY if f'"{n}"' in source]
    if not present:
        return (
            f"SKIP   {target}: none of these knobs is registered here -- this tree"
            " does not carry the patches that define them"
        )

    patched = ANCHOR + PATCHED_TAIL + "".join(f'        "{n}",\n' for n in present)
    source = source.replace(ANCHOR, patched, 1)
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
    ap.add_argument("--check", action="store_true", help="report only")
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
