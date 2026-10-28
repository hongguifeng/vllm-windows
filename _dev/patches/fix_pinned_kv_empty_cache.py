# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Release the allocator cache on the pinned-KV path, before the pool is allocated.

Port of upstream HyperQwen `patches/pinned-kv-empty-cache.patch` (d7ead9c02,
PR #263), re-cut onto the 0.29.1 `determine_available_memory` pinned branch.

The pinned branch (`--kv-cache-memory-bytes`, how every launch here sizes KV)
runs `profile_run()` and returns: the KV cache is allocated right after that, on
top of whatever the run left cached in the allocator. On a cold compile cache
that is the torch.compile and Triton autotune scratch, and the pinned figure
does not budget for it -- so the pool over-commits the card.

Upstream measured it on a 24 GiB RTX 3090: 1.2 GiB of scratch left behind, a
5,008 MiB pool on top of 20,391 MiB already in use, and under WSL2's driver the
overflow does not fail -- about 380 MiB of GPU memory is moved to system RAM for
the life of the engine, the kernels that read it run 14-28x slower, and about
half of fresh boots decoded at 25 against 50 tok/s. Releasing the cache first:
the pool fits (24,183 of 24,576 MiB), nothing lives in system RAM, 4/4 boots at
50.3-53.1 tok/s with the full pool.

Same failure shape we recorded here independently: the 3090 "paging cliff" that
keeps the pin at 5.8e9, and the chunked-bandwidth probe where the last GiB ran at
2.2 GB/s -- PCIe speed, silently demoted to host memory.

The two calls are written in the device-generic form this repo asks for (RFC
#30679): `torch.accelerator`. The file already calls its synchronize twice
before this point, so the patch matches the surrounding style.

Usage:
    fix_pinned_kv_empty_cache.py <vllm_package_dir> [<vllm_package_dir> ...]
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

MARKER = "[fix_pinned_kv_empty_cache]"
REL_PATH = Path("v1/worker/gpu_worker.py")

ANCHOR = """            self.model_runner.profile_run()

            msg = (
"""

PATCHED = """            self.model_runner.profile_run()
            # [fix_pinned_kv_empty_cache] The pool is allocated right after this
            # return, on top of whatever the profile run left cached. On a cold
            # compile cache that is torch.compile + Triton autotune scratch
            # (1.2 GiB on a 24 GiB card), which the pinned size does not budget
            # for: the card ends up over-committed, and the overflow is not an
            # error -- it is served from host memory at PCIe speed. Hand it back
            # before the pool is reserved, like the measured path does.
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()

            msg = (
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
            f"FAIL   {target}: pinned-branch anchor matched {source.count(ANCHOR)} "
            "times (expected 1) -- upstream changed, re-derive the patch"
        )

    source = source.replace(ANCHOR, PATCHED, 1)
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
