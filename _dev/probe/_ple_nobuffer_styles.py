"""Why does the C program read NO_BUFFERING while ctypes gets ERROR_INVALID_PARAMETER?

Same file, same offset, same count, same alignment. This runs the C exe from
inside the same Python process and then tries several ctypes call styles, so the
difference is localized to how Python binds the call.

  python _dev/probe/_ple_nobuffer_styles.py [path]
"""

import ctypes
import os
import subprocess
import sys

GENERIC_READ = 0x80000000
SHARE_ALL = 0x00000007
OPEN_EXISTING = 3
NO_BUFFERING = 0x40000000
RANDOM_ACCESS = 0x10000000
INVALID = ctypes.c_void_p(-1).value

NBTEST = r"C:\Users\hong\AppData\Local\Temp\psprobe\nbtest.exe"

TARGET = (
    r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
    r"\model-00001-of-00011.safetensors"
)


def c_reference(path):
    if not os.path.exists(NBTEST):
        print("  nbtest.exe missing -- build it with _nbtest_build_run.ps1")
        return
    r = subprocess.run([NBTEST, path], capture_output=True, text=True, timeout=120)
    for line in r.stdout.splitlines():
        if "ReadFile(4096 @4096" in line:
            print(f"  C reference in this very process: {line.strip()}")


def style_strict(path):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.CreateFileW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
    ]
    k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                             ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    k32.ReadFile.restype = ctypes.c_int
    k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                     ctypes.POINTER(ctypes.c_longlong), ctypes.c_uint]
    k32.SetFilePointerEx.restype = ctypes.c_int
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
                                 ctypes.c_uint]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]

    h = k32.CreateFileW(path, GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING,
                        NO_BUFFERING | RANDOM_ACCESS, None)
    buf = k32.VirtualAlloc(None, 8192, 0x3000, 0x04)
    moved = ctypes.c_longlong(0)
    k32.SetFilePointerEx(h, 4096, ctypes.byref(moved), 0)
    got = ctypes.c_uint(0)
    ok = k32.ReadFile(h, buf, 4096, ctypes.byref(got), None)
    err = 0 if ok else ctypes.get_last_error()
    print(f"  strict argtypes, VirtualAlloc ptr : ok={int(bool(ok))} "
          f"bytes={got.value} err={err}")
    k32.CloseHandle(h)


def style_strict_bufferobj(path):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.ReadFile.restype = ctypes.c_int
    k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                             ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    k32.SetFilePointerEx.restype = ctypes.c_int
    h = k32.CreateFileW(path, GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING,
                        NO_BUFFERING | RANDOM_ACCESS, None)
    buf = ctypes.create_string_buffer(8192)
    addr = ctypes.addressof(buf)
    aligned = (addr + 4095) & ~4095
    k32.SetFilePointerEx(h, 4096, None, 0)
    got = ctypes.c_uint(0)
    ok = k32.ReadFile(h, aligned, 4096, ctypes.byref(got), None)
    err = 0 if ok else ctypes.get_last_error()
    print(f"  strict, page-aligned buffer addr  : ok={int(bool(ok))} "
          f"bytes={got.value} err={err}")
    k32.CloseHandle(h)


def style_no_prototype(path):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = k32.CreateFileW(path, GENERIC_READ, SHARE_ALL, 0, OPEN_EXISTING,
                        NO_BUFFERING | RANDOM_ACCESS, 0)
    buf = ctypes.create_string_buffer(8192)
    addr = ctypes.addressof(buf)
    addr = (addr + 4095) & ~4095
    k32.SetFilePointerEx(h, 4096, 0, 0)
    got = ctypes.c_uint(0)
    ok = k32.ReadFile(h, addr, 4096, ctypes.byref(got), 0)
    err = 0 if ok else ctypes.get_last_error()
    print(f"  no argtypes/restypes at all       : ok={int(bool(ok))} "
          f"bytes={got.value} err={err}")
    k32.CloseHandle(h)


def style_windll(path):
    k32 = ctypes.windll.kernel32
    h = k32.CreateFileW(path, GENERIC_READ, SHARE_ALL, 0, OPEN_EXISTING,
                        NO_BUFFERING | RANDOM_ACCESS, 0)
    buf = ctypes.create_string_buffer(8192)
    addr = (ctypes.addressof(buf) + 4095) & ~4095
    k32.SetFilePointerEx(h, 4096, 0, 0)
    got = ctypes.c_uint(0)
    ok = k32.ReadFile(h, addr, 4096, ctypes.byref(got), 0)
    print(f"  windll (stdcall view), no use_last: ok={int(bool(ok))} "
          f"bytes={got.value} err={ctypes.get_last_error()}")
    k32.CloseHandle(h)


def style_write_through(path):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    WT = 0x80000000
    h = k32.CreateFileW(path, GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING,
                        WT | RANDOM_ACCESS, None)
    buf = ctypes.create_string_buffer(8192)
    addr = (ctypes.addressof(buf) + 4095) & ~4095
    k32.SetFilePointerEx(h, 4096, None, 0)
    got = ctypes.c_uint(0)
    ok = k32.ReadFile(h, addr, 4096, ctypes.byref(got), None)
    err = 0 if ok else ctypes.get_last_error()
    print(f"  WRITE_THROUGH (no alignment req)  : ok={int(bool(ok))} "
          f"bytes={got.value} err={err}")
    k32.CloseHandle(h)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else TARGET
    print(f"file: {path} ({os.path.getsize(path)/2**30:.2f} GiB)")
    c_reference(path)
    for fn in (style_strict, style_strict_bufferobj, style_no_prototype,
               style_windll, style_write_through):
        fn(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
