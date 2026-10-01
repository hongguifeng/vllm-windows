"""Undo the accidental edit to _dev/out/_wsl_probe.txt (a raw diagnostic dump).

_scan_ctrlchars.py --fix replaced every NUL byte in that file with the two
characters `\\0`, because its skip-list check missed files sitting *directly* in
`_dev\\out` (the directory path has no trailing separator).  The fix is exact:
only that one substitution was applied, so reversing it restores the original
bytes.  The skip-list is fixed too, so it cannot recur.

    python _restore_wsl_probe.py [--apply]
"""
import os
import sys

P = r"D:\code\vllm-windows\_dev\out\_wsl_probe.txt"


def main():
    apply = "--apply" in sys.argv
    b = open(P, "rb").read()
    n = b.count(b"\\0")
    print(f"{P}: {len(b)} bytes, {n} x '\\0' sequences")
    if not n:
        print("nothing to undo")
        return
    if not apply:
        print("dry run -- pass --apply to rewrite")
        return
    out = b.replace(b"\\0", b"\x00")
    open(P, "wb").write(out)
    print(f"rewritten: {len(out)} bytes, {out.count(b'\\x00')} NUL bytes restored")


if __name__ == "__main__":
    main()
