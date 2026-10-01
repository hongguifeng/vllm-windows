"""Unbreak `vllm.entrypoints.openai.api_server` on Windows.

`vllm/entrypoints/launchers/api_server/entry.py::main()` imports `uvloop`
unconditionally, *before* it branches on the platform:

    def main():
        import uvloop          # <-- dies here on Windows
        import platform
        if platform.system() == "Windows":
            import winloop as uvloop_impl
            ...
        else:
            import uvloop as uvloop_impl

`uvloop` has no Windows build (it is libuv+CPython internals only), so the
stray import makes the server exit with

    ModuleNotFoundError: No module named 'uvloop'

even though `winloop` -- the Windows port -- is installed and is what the
`if` branch actually wants.  Drop the dead import; the real implementation is
still selected per-platform two lines below.

`launchers/dp_supervisor.py` already does this correctly, so `entry.py` is the
outlier.  `launchers/render/entry.py` has the same flaw but calls `uvloop.run`
directly, i.e. it needs a real port (not a one-line delete) and we do not use
it -- left alone on purpose.

Usage:
    python fix_winloop_import.py <vllm_pkg_dir> [<another_vllm_pkg_dir> ...]

    # e.g. both the installed package and the checked-out source tree
    python fix_winloop_import.py .venv/Lib/site-packages/vllm vllm
"""

import os
import sys

TARGET = os.path.join("entrypoints", "launchers", "api_server", "entry.py")

_NEEDLE_LF = b"def main():\n    import uvloop\n    import platform\n"
_REPLACEMENT_LF = b"def main():\n    import platform\n"

_MARKER = b"import winloop as uvloop_impl"


def patch(root: str) -> None:
    path = os.path.join(root, TARGET)
    if not os.path.exists(path):
        print(f"skip (not found): {path}")
        return

    with open(path, mode="rb") as file:
        raw = file.read()

    # Preserve the file's existing newline style: match against whichever
    # convention is actually on disk instead of rewriting the whole file.
    eol = b"\r\n" if b"\r\n" in raw else b"\n"
    needle = _NEEDLE_LF.replace(b"\n", eol)
    replacement = _REPLACEMENT_LF.replace(b"\n", eol)

    if needle not in raw:
        if _MARKER in raw:
            print(f"already patched: {path}")
        else:
            print(f"WARN: anchor not found, left untouched: {path}")
        return

    with open(path, mode="wb") as file:
        file.write(raw.replace(needle, replacement, 1))
    print(f"patched: {path}")


def main() -> int:
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        return 2
    for root in roots:
        patch(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
