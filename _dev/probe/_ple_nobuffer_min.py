"""ctypes reads NO_BUFFERING line by line, printing every step's Win32 error.

The C program (_nbtest.c) succeeds on the same file with the same flags, so the
87s in _ple_nobuffer_align.py are a binding artifact. This narrows it down:
same call, same order, but with VirtualAlloc reported and two buffer variants.

  python _dev/probe/_ple_nobuffer_min.py [path]
"""

import ctypes
import os
import sys

GENERIC_READ = 0x80000000
FILE_SHARE_ALL = 0x00000007
OPEN_EXISTING = 3
NO_BUFFERING = 0x40000000
RANDOM_ACCESS = 0x10000000
MEM_COMMIT_RESERVE = 0x00003000
PAGE_READWRITE = 0x04
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
k32.CloseHandle.restype = ctypes.c_int


def step(label, ok, err):
    print(f"  {label:44s} ok={int(bool(ok))} err={err}")


def run(path, use_ansi, use_virtualalloc):
    print(f"--- CreateFile{'A' if use_ansi else 'W'}, "
          f"buffer={'VirtualAlloc' if use_virtualalloc else 'create_string_buffer'} ---")
    flags = NO_BUFFERING | RANDOM_ACCESS
    if use_ansi:
        enc = str(path).encode("mbcs")
        h = k32.CreateFileA(enc, GENERIC_READ, FILE_SHARE_ALL, None,
                            OPEN_EXISTING, flags, None)
    else:
        h = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_ALL, None,
                            OPEN_EXISTING, flags, None)
    if h in (None, INVALID):
        step("CreateFile", False, ctypes.WinError(ctypes.get_last_error()))
        return
    step("CreateFile", True, 0)

    if use_virtualalloc:
        ptr = k32.VirtualAlloc(None, 8192, MEM_COMMIT_RESERVE, PAGE_READWRITE)
        if not ptr:
            step("VirtualAlloc", False, ctypes.WinError(ctypes.get_last_error()))
            return
        base = ptr if isinstance(ptr, int) else ptr.value
        print(f"  VirtualAlloc -> 0x{base:x} (mod 4096 = {base % 4096}, "
              f"mod 512 = {base % 512})")
        buf = base
    else:
        cbuf = ctypes.create_string_buffer(8192)
        buf = ctypes.addressof(cbuf)
        print(f"  create_string_buffer -> 0x{buf:x} (mod 4096 = {buf % 4096}, "
              f"mod 512 = {buf % 512})")

    moved = ctypes.c_longlong(0)
    ok = k32.SetFilePointerEx(h, 4096, ctypes.byref(moved), 0)
    step("SetFilePointerEx(4096)", ok, 0 if ok else ctypes.get_last_error())

    got = ctypes.c_uint(0)
    ok = k32.ReadFile(h, buf, 4096, ctypes.byref(got), None)
    step("ReadFile(4096 @4096)", ok, 0 if ok else ctypes.get_last_error())
    if ok:
        print(f"  bytes read = {got.value}")

    # 512-byte alignment, matching what the C program also proved works
    moved = ctypes.c_longlong(0)
    k32.SetFilePointerEx(h, 512, ctypes.byref(moved), 0)
    got = ctypes.c_uint(0)
    ok = k32.ReadFile(h, buf, 512, ctypes.byref(got), None)
    step("ReadFile(512 @512)", ok, 0 if ok else ctypes.get_last_error())
    k32.CloseHandle(h)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else (
        r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
        r"\model-00001-of-00011.safetensors")
    print(f"file: {path} ({os.path.getsize(path)/2**30:.2f} GiB)")
    for use_ansi in (True, False):
        for use_va in (True, False):
            run(path, use_ansi, use_va)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
