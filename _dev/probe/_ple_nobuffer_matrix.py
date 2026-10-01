"""Isolate what differs between the working C call and the failing ctypes call.

Every case here uses a VirtualAlloc'd buffer (64 KiB granule, so page-aligned by
construction) and a 4096-byte read at offset 4096, so alignment is never the
variable. Only share mode, flags, entry point, and seek call change.

  python _dev/probe/_ple_nobuffer_matrix.py [path]
"""

import ctypes
import os
import sys

GENERIC_READ = 0x80000000
SHARE_RW = 0x00000006
SHARE_RWD = 0x00000007
SHARE_R = 0x00000001
OPEN_EXISTING = 3
NO_BUFFERING = 0x40000000
RANDOM_ACCESS = 0x10000000
WRITE_THROUGH = 0x80000000
INVALID = ctypes.c_void_p(-1).value

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileA.restype = ctypes.c_void_p
k32.CreateFileA.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint,
                            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                            ctypes.c_void_p]
k32.CreateFileW.restype = ctypes.c_void_p
k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                            ctypes.c_void_p]
k32.SetFilePointer.restype = ctypes.c_ulong
k32.SetFilePointer.argtypes = [ctypes.c_void_p, ctypes.c_long,
                               ctypes.POINTER(ctypes.c_long), ctypes.c_uint]
k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                 ctypes.POINTER(ctypes.c_longlong), ctypes.c_uint]
k32.SetFilePointerEx.restype = ctypes.c_int
k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                         ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
k32.ReadFile.restype = ctypes.c_int
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
                             ctypes.c_uint]
k32.CloseHandle.argtypes = [ctypes.c_void_p]

TARGET = (
    r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
    r"\model-00001-of-00011.safetensors"
)

BUF = None
TARGET_PATH = TARGET


def buffer():
    global BUF
    if BUF is None:
        BUF = k32.VirtualAlloc(None, 65536, 0x3000, 0x04)
    return BUF


def case(label, ansi, share, flags, use_old_seek, offset=4096, count=4096):
    buf = buffer()
    if ansi:
        h = k32.CreateFileA(str(TARGET_PATH).encode("mbcs"), GENERIC_READ, share,
                            None, OPEN_EXISTING, flags, None)
    else:
        h = k32.CreateFileW(TARGET_PATH, GENERIC_READ, share, None, OPEN_EXISTING,
                            flags, None)
    if h in (None, INVALID):
        print(f"  {label:38s} CreateFile err="
              f"{ctypes.get_last_error()}")
        return
    if use_old_seek:
        cur = k32.SetFilePointer(h, offset, None, 0)
        if cur == 0xFFFFFFFF:
            print(f"  {label:38s} SetFilePointer err={ctypes.get_last_error()}")
            k32.CloseHandle(h)
            return
    else:
        moved = ctypes.c_longlong(0)
        if not k32.SetFilePointerEx(h, offset, ctypes.byref(moved), 0):
            print(f"  {label:38s} SetFilePointerEx err={ctypes.get_last_error()}")
            k32.CloseHandle(h)
            return
    got = ctypes.c_uint(0)
    ok = k32.ReadFile(h, buf, count, ctypes.byref(got), None)
    if ok:
        print(f"  {label:38s} ok, {got.value} B")
    else:
        print(f"  {label:38s} err={ctypes.get_last_error()}")
    k32.CloseHandle(h)


def main() -> int:
    global TARGET_PATH
    TARGET_PATH = sys.argv[1] if len(sys.argv) > 1 else TARGET
    TARGET_PATH = sys.argv[1] if len(sys.argv) > 1 else TARGET
    print(f"file: {TARGET_PATH} "
          f"({os.path.getsize(TARGET_PATH)/2**30:.2f} GiB)")
    buf = buffer()
    print(f"buffer 0x{buf:x}: mod4096={buf % 4096} mod512={buf % 512}")
    print()
    case("A W, share RW, NB|RA, seekEx", False, SHARE_RW,
         NO_BUFFERING | RANDOM_ACCESS, False)
    case("B W, share RWD, NB|RA, seekEx", False, SHARE_RWD,
         NO_BUFFERING | RANDOM_ACCESS, False)
    case("C W, share R, NB|RA, seekEx", False, SHARE_R,
         NO_BUFFERING | RANDOM_ACCESS, False)
    case("D A, share RW, NB|RA, seekEx", True, SHARE_RW,
         NO_BUFFERING | RANDOM_ACCESS, False)
    case("E W, share RW, NB only, seekEx", False, SHARE_RW, NO_BUFFERING, False)
    case("F W, share RW, NB|RA, seek32", False, SHARE_RW,
         NO_BUFFERING | RANDOM_ACCESS, True)
    case("G W, share RW, NB|RA|WT, seekEx", False, SHARE_RW,
         NO_BUFFERING | RANDOM_ACCESS | WRITE_THROUGH, False)
    case("H W, share RW, NB|RA, seekEx @512/512", False, SHARE_RW,
         NO_BUFFERING | RANDOM_ACCESS, False, 512, 512)
    case("I W, share RW, NB|RA, seekEx @0/4096", False, SHARE_RW,
         NO_BUFFERING | RANDOM_ACCESS, False, 0, 4096)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
