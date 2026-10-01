"""Which CreateFileW flag combinations does the shim's open accept in-process?

The buffered control failed to bind, and the verification handle came back NULL,
so before trusting any throughput number we need to know what opens at all.

  python _dev/probe/_nbshim_open_matrix.py [path]
"""

import ctypes
import os
import sys

GENERIC_READ = 0x80000000
SHARE_RW = 0x00000006
NO_BUFFERING = 0x40000000
RANDOM_ACCESS = 0x10000000
WRITE_THROUGH = 0x80000000
SEQUENTIAL = 0x00000000
SHIM = r"D:\code\vllm-windows\_dev\out\nbshim\nbshim.dll"
TARGET = (
    r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
    r"\model-00001-of-00011.safetensors"
)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else TARGET
    shim = ctypes.CDLL(SHIM)
    shim.nb_open.restype = ctypes.c_void_p
    shim.nb_open.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                             ctypes.c_uint]
    shim.nb_close.argtypes = [ctypes.c_void_p]
    shim.nb_size.restype = ctypes.c_longlong
    shim.nb_size.argtypes = [ctypes.c_void_p]
    shim.nb_read.restype = ctypes.c_longlong
    shim.nb_read.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_uint,
                             ctypes.c_void_p]
    shim.nb_last_error.restype = ctypes.c_uint

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
                                 ctypes.c_uint]
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    k32.VirtualFree.restype = ctypes.c_int
    buf = k32.VirtualAlloc(None, 65536, 0x3000, 0x04)

    print(f"file: {path} ({os.path.getsize(path)/2**30:.2f} GiB)")
    cases = [
        ("RANDOM_ACCESS only", RANDOM_ACCESS),
        ("no flags", 0),
        ("SEQUENTIAL only", SEQUENTIAL),
        ("WRITE_THROUGH|RANDOM", WRITE_THROUGH | RANDOM_ACCESS),
        ("NO_BUFFERING|RANDOM", NO_BUFFERING | RANDOM_ACCESS),
        ("NO_BUFFERING only", NO_BUFFERING),
    ]
    for label, flags in cases:
        h = shim.nb_open(path, GENERIC_READ, SHARE_RW, flags)
        if not h:
            e = shim.nb_last_error()
            print(f"  {label:22s} open -> NULL err={e}")
            continue
        size = shim.nb_size(h)
        n = shim.nb_read(h, 4096, 4096, buf)
        err = shim.nb_last_error() if n < 0 else 0
        printable = ""
        if n > 0:
            chunk = ctypes.string_at(buf, 16)
            printable = "printable" if all(32 <= b < 127 for b in chunk) \
                else repr(chunk[:12])
        print(f"  {label:22s} open -> 0x{h:x} size={size} "
              f"read(4096@4096)={n} err={err} {printable}")
        shim.nb_close(h)
    if not k32.VirtualFree(buf, 0, 0x8000):
        print(f"  VirtualFree failed err={ctypes.get_last_error()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
