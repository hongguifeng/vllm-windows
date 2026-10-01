"""Release the device copy an offloaded tower uploads on every forward.

`UVAOffloader` with UVA disabled (`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`, which
is how 0.29's `--cpu-offload-gb 1 --cpu-offload-params visual` runs on Windows)
uploads the module's weights on *every* forward:

    device_state = {k: v.to(device) for k, v in module.state_dict().items()}
    output = functional_call(module, device_state, args, kwargs, tie_weights=False)

When the call returns the tensors are unreachable, but they are already in the
caching allocator's reserved pool, and vLLM only calls `torch.cuda.empty_cache()`
at CUDA-graph capture (`v1/worker/gpu/model_runner.py`) and at teardown/sleep --
never inside the inference loop.  So the offload buys GPU memory only until the
first image is encoded; from then on the vision tower's upload sits on the card
for the rest of the process lifetime:

    Total CPU offloaded parameters: 0.86 GiB      (the whole tower is offloaded)

Measured here (24 GiB 3090, `_dev/probe/_paging_watch.ps1`, KV pinned 4.2e9):

    before any image        Local 21,727.7 MiB
    after  1 x 196-tok img  Local 22,535      (+808, i.e. the tower)
    after  8 x 1024 px      Local 23,007.7    (+120 more -- saturated)

and replicated offline, with no server, by `_dev/probe/_uva_offload_cache_probe.py`
(a pure-torch copy of this exact branch): `reserved` goes 0 -> 98 MiB on
forward #1 (weights 96 MiB) and +0 on every forward after that.  The copy is
*cached*, not leaked -- which is why it never drops back, and also why one
`empty_cache()` is enough to undo it.

Why that matters on a pinned-to-the-card KV pool: with `-KvBytes` at the card
limit, those 879 MiB are exactly what pushes decode over the edge, and the pages
WDDM then demotes to host include KV pages that every decode step has to read
back (measured: rx 2000-3386 MB/s of PCIe churn, 104.6 ms/step instead of 26.6).

WHAT THIS DOES: drop the local reference after `functional_call` and hand the
blocks straight back to the driver.  The tower already re-uploads over PCIe on
every forward, so no transfer is added -- the only new cost is one cudaMalloc on
the next tower forward.  Set `VLLM_OFFLOAD_RELEASE_AFTER_FORWARD=1` to enable;
`start_server.ps1 -ReleaseOffloadCopy` does it.  Unset it and the code path is
byte-identical to upstream.

Usage:
    fix_offload_release.py <vllm_package_dir> [<vllm_package_dir> ...]
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

MARKER = "VLLM_OFFLOAD_RELEASE_AFTER_FORWARD"
REL_PATH = Path("model_executor/offloader/uva.py")

# `uva.py` only imports torch/nn/functional_call today; the release hook needs os.
IMPORT_ANCHOR = """from collections.abc import Generator

import torch
import torch.nn as nn"""

IMPORT_PATCHED = """from collections.abc import Generator
import os

import torch
import torch.nn as nn"""

FORWARD_ANCHOR = """                module.forward = forward
                return output
"""

FORWARD_PATCHED = """                # [fix_offload_release] The upload above is unreachable now,
                # but it lives in the caching allocator's reserved pool, and
                # vLLM never empties that inside the inference loop -- so the
                # tower would keep its full size on the card for the rest of
                # the process, squeezing a pinned-to-the-limit KV pool into
                # host memory. Hand the blocks back to the driver instead.
                if os.environ.get("VLLM_OFFLOAD_RELEASE_AFTER_FORWARD", "0") == "1":
                    del device_state
                    torch.cuda.empty_cache()
                module.forward = forward
                return output
"""


def patch_one(pkg_dir: Path, *, check_only: bool = False) -> str:
    target = pkg_dir / REL_PATH
    if not target.is_file():
        return f"SKIP   {target} (no such file)"

    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        return f"OK     {target} (already patched)"

    for anchor, patched, what in (
        (IMPORT_ANCHOR, IMPORT_PATCHED, "import os"),
        (FORWARD_ANCHOR, FORWARD_PATCHED, "release hook"),
    ):
        if source.count(anchor) != 1:
            return (
                f"FAIL   {target}: anchor for {what} matched "
                f"{source.count(anchor)} times (expected 1) -- upstream changed, "
                "re-derive the patch"
            )
        source = source.replace(anchor, patched, 1)

    if check_only:
        return f"WOULD  {target}"

    target.write_text(source, encoding="utf-8", newline="\n")

    try:
        py_compile.compile(str(target), doraise=True)
    except py_compile.PyCompileError as exc:  # pragma: no cover - defensive
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
