"""Does a compiled C ReadFile succeed when it runs inside python.exe?

If it does, the ERROR_INVALID_PARAMETER we kept seeing is purely an artifact of
how ctypes forms the call, and this shim becomes the PLE/SSD read path on
Windows. If it does not, the process itself is being blocked.

  python _dev/probe/_nbshim_test.py [path]
"""

import ctypes
import os
import sys

GENERIC_READ = 0x80000000
SHARE_RW = 0x00000006
NO_BUFFERING = 0x40000000
RANDOM_ACCESS = 0x10000000

SHIM = r"D:\code\vllm-windows\_dev\out\nbshim\nbshim.dll"
TARGET = (
    r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
    r"\model-00001-of-00011.safetensors"
)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else TARGET
    print(f"file: {path} ({os.path.getsize(path)/2**30:.2f} GiB)")
    if not os.path.exists(SHIM):
        print(f"missing {SHIM} -- run _dev/bin/_nbshim_build.ps1")
        return 1
    shim = ctypes.CDLL(SHIM)
    shim.nb_open.restype = ctypes.c_void_p
    shim.nb_open.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                             ctypes.c_uint]
    shim.nb_close.argtypes = [ctypes.c_void_p]
    shim.nb_close.restype = ctypes.c_int
    shim.nb_read.restype = ctypes.c_longlong
    shim.nb_read.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_uint,
                             ctypes.c_void_p]
    shim.nb_size.restype = ctypes.c_longlong
    shim.nb_size.argtypes = [ctypes.c_void_p]
    shim.nb_last_error.restype = ctypes.c_uint
    shim.nb_read_ov.restype = ctypes.c_longlong
    shim.nb_read_ov.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_uint,
                                ctypes.c_void_p, ctypes.c_void_p]

    h = shim.nb_open(path, GENERIC_READ, SHARE_RW, NO_BUFFERING | RANDOM_ACCESS)
    if not h:
        print(f"nb_open -> {shim.nb_last_error()}")
        return 1
    size = shim.nb_size(h)
    print(f"nb_size -> {size} B ({size/2**30:.2f} GiB)")

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
                                 ctypes.c_uint]
    buf = k32.VirtualAlloc(None, 65536, 0x3000, 0x04)
    print(f"buffer 0x{buf:x} mod4096={buf % 4096}")

    for off, count in ((4096, 4096), (0, 4096), (512, 512),
                       (319488, 4096), (3948544, 4096), (0, 65536)):
        n = shim.nb_read(h, off, count, buf)
        err = "" if n >= 0 else f" err={shim.nb_last_error()}"
        head = ""
        if n > 0:
            chunk = ctypes.string_at(buf, min(int(n), 24))
            printable = all(32 <= b < 127 or b in (9, 10, 13) for b in chunk)
            head = f" first {len(chunk)} B " + (
                "printable" if printable else repr(chunk[:12]))
        print(f"  nb_read({off}, {count}) -> {n}{err}   {head}")

    # one overlapped read straight through the shim
    class OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_ulong),
            ("InternalHigh", ctypes.c_ulong),
            ("Offset", ctypes.c_ulong),
            ("OffsetHigh", ctypes.c_ulong),
            ("hEvent", ctypes.c_void_p),
        ]

    ov = OVERLAPPED()
    r = shim.nb_read_ov(h, 4096, 4096, buf, ctypes.byref(ov))
    print(f"  nb_read_ov(4096, 4096) -> {r} err={shim.nb_last_error()}")
    shim.nb_close(h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
