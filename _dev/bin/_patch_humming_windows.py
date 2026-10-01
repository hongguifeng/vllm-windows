"""Apply the Windows humming-kernels 0.1.15 runtime port.

The upstream wheel has Linux-only native artifact names and build paths.  This
script keeps the port reproducible while leaving .orig backups beside every
modified site-packages file.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILES = (
    Path("utils/device.py"),
    Path("utils/nvrtc.py"),
    Path("utils/cubin.py"),
    Path("ops/utils.py"),
    Path("csrc/launcher/mapped_file.h"),
    Path("csrc/nvrtc_compile.cpp"),
    Path("csrc/patch_cubin.cpp"),
)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="")


def apply_device(path: Path) -> None:
    replace_once(
        path,
        '    extension_name = "_device_info.abi3.so"\n',
        '    extension_name = (\n'
        '        "_device_info.pyd" if sys.platform == "win32"\n'
        '        else "_device_info.abi3.so"\n'
        '    )\n',
    )


def apply_nvrtc(path: Path) -> None:
    replace_once(
        path,
        '''def _select_nvrtc_lib(lib_dir):
    unversioned = os.path.join(lib_dir, "libnvrtc.so")
    if os.path.exists(unversioned):
        return unversioned
    versioned = sorted(glob.glob(os.path.join(lib_dir, "libnvrtc.so.*")))
    if versioned:
        return versioned[-1]
    return None
''',
        '''def _select_nvrtc_lib(lib_dir):
    if sys.platform == "win32":
        versioned = sorted(glob.glob(os.path.join(lib_dir, "nvrtc*.dll")))
        return versioned[-1] if versioned else None
    unversioned = os.path.join(lib_dir, "libnvrtc.so")
    if os.path.exists(unversioned):
        return unversioned
    versioned = sorted(glob.glob(os.path.join(lib_dir, "libnvrtc.so.*")))
    if versioned:
        return versioned[-1]
    return None
''',
    )
    replace_once(
        path,
        '''    candidates = [
        os.path.join(root, "lib64"),
        os.path.join(root, "lib"),
        *sorted(glob.glob(os.path.join(root, "*", "lib64"))),
        *sorted(glob.glob(os.path.join(root, "*", "lib"))),
    ]
''',
        '''    if sys.platform == "win32":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            root = cuda_path
            env["path"] = root
            env["include_paths"] = [os.path.join(root, "include")]
        candidates = [os.path.join(root, "bin"), os.path.join(root, "bin", "x64")]
    else:
        candidates = [
            os.path.join(root, "lib64"),
            os.path.join(root, "lib"),
            *sorted(glob.glob(os.path.join(root, "*", "lib64"))),
            *sorted(glob.glob(os.path.join(root, "*", "lib"))),
        ]
''',
    )
    old = '        raise RuntimeError("Could not locate libnvrtc.so in CUDA path")\n'
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 2:
        raise RuntimeError(f"{path}: expected two NVRTC error messages")
    path.write_text(
        text.replace(old, '        raise RuntimeError("Could not locate NVRTC in CUDA path")\n'),
        encoding="utf-8",
        newline="",
    )
    replace_once(
        path,
        '''    cmd = [
        compiler,
        "-O2",
        "-std=c++17",
        str(src_path),
        *[f"-I{path}" for path in include_paths],
        "-ldl",
        "-o",
        str(tmp_path),
    ]
''',
        '''    if sys.platform == "win32":
        tmp_path = output_path.with_name(output_path.stem + ".tmp.exe")
        cmd = [
            compiler,
            "/nologo",
            "/O2",
            "/std:c++17",
            "/EHsc",
            "/MD",
            *[f"/I{path}" for path in include_paths],
            str(src_path),
            f"/Fe{tmp_path}",
        ]
    else:
        cmd = [
            compiler,
            "-O2",
            "-std=c++17",
            str(src_path),
            *[f"-I{path}" for path in include_paths],
            "-ldl",
            "-o",
            str(tmp_path),
        ]
''',
    )
    replace_once(
        path,
        '    native_path = jit_utils.get_precompiled_artifact_path(src_path, "nvrtc_compile")\n',
        '    artifact_name = "nvrtc_compile.exe" if sys.platform == "win32" else "nvrtc_compile"\n'
        '    native_path = jit_utils.get_precompiled_artifact_path(src_path, artifact_name)\n',
    )
    replace_once(
        path,
        '    binary_path = build_dir / "nvrtc_compile"\n',
        '    binary_path = build_dir / ("nvrtc_compile.exe" if sys.platform == "win32" else "nvrtc_compile")\n',
    )
    old = '    compiler = os.environ.get("CXX") or "g++"\n'
    new = '    compiler = os.environ.get("CXX") or ("cl" if sys.platform == "win32" else "g++")\n'
    count = path.read_text(encoding="utf-8").count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one compiler default, found {count}")
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
        newline="",
    )


def apply_cubin(path: Path) -> None:
    replace_once(path, "import subprocess\n", "import subprocess\nimport sys\n")
    replace_once(
        path,
        '''    cmd = [
        compiler,
        "-O2",
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(src_path),
        "-o",
        str(tmp_path),
    ]
''',
        '''    if sys.platform == "win32":
        tmp_path = output_path.with_name(output_path.stem + ".tmp.dll")
        cmd = [
            compiler,
            "/nologo",
            "/O2",
            "/std:c++17",
            "/EHsc",
            "/MD",
            str(src_path),
            "/LD",
            f"/Fe{tmp_path}",
        ]
    else:
        cmd = [
            compiler,
            "-O2",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(src_path),
            "-o",
            str(tmp_path),
        ]
''',
    )
    replace_once(
        path,
        '    native_path = jit_utils.get_precompiled_artifact_path(src_path, "libcubinpatch.so")\n',
        '    artifact_name = "libcubinpatch.dll" if sys.platform == "win32" else "libcubinpatch.so"\n'
        '    native_path = jit_utils.get_precompiled_artifact_path(src_path, artifact_name)\n',
    )
    replace_once(
        path,
        '    lib_path = build_dir / "libcubinpatch.so"\n',
        '    lib_path = build_dir / ("libcubinpatch.dll" if sys.platform == "win32" else "libcubinpatch.so")\n',
    )
    replace_once(
        path,
        '    compiler = os.environ.get("CXX") or "g++"\n',
        '    compiler = os.environ.get("CXX") or ("cl" if sys.platform == "win32" else "g++")\n',
    )


def apply_ops(path: Path) -> None:
    replace_once(
        path,
        '        extra_ldflags=["-lcuda", "-lc10_cuda", "-ltorch_cuda"],\n',
        '        extra_ldflags=(\n'
        '            ["cuda.lib", "c10_cuda.lib", "torch_cuda.lib"]\n'
        '            if sys.platform == "win32"\n'
        '            else ["-lcuda", "-lc10_cuda", "-ltorch_cuda"]\n'
        '        ),\n',
    )


def apply_mapped_file(path: Path) -> None:
    path.write_text(
        '''#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

class MappedFile {
public:
  explicit MappedFile(const std::string &path) {
#ifdef _WIN32
    file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) throw_error("CreateFile", path, GetLastError());
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file_, &size) || size.QuadPart <= 0) {
      DWORD error = GetLastError();
      CloseHandle(file_);
      file_ = INVALID_HANDLE_VALUE;
      throw_error("GetFileSizeEx", path, error);
    }
    size_ = static_cast<size_t>(size.QuadPart);
    mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_ == nullptr) {
      DWORD error = GetLastError();
      CloseHandle(file_);
      file_ = INVALID_HANDLE_VALUE;
      throw_error("CreateFileMapping", path, error);
    }
    data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
    if (data_ == nullptr) {
      DWORD error = GetLastError();
      CloseHandle(mapping_);
      CloseHandle(file_);
      mapping_ = nullptr;
      file_ = INVALID_HANDLE_VALUE;
      throw_error("MapViewOfFile", path, error);
    }
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw_error("open", path, errno);
    struct stat st;
    if (fstat(fd, &st) != 0) {
      int error = errno;
      close(fd);
      throw_error("fstat", path, error);
    }
    if (st.st_size <= 0) {
      close(fd);
      throw std::runtime_error("empty file: " + path);
    }
    size_ = static_cast<size_t>(st.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    int error = errno;
    close(fd);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw_error("mmap", path, error);
    }
#endif
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  ~MappedFile() {
#ifdef _WIN32
    if (data_ != nullptr) UnmapViewOfFile(data_);
    if (mapping_ != nullptr) CloseHandle(mapping_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
    if (data_ != nullptr) munmap(data_, size_);
#endif
  }

  const void *data() const { return data_; }
  size_t size() const { return size_; }

private:
  [[noreturn]] static void throw_error(const char *operation, const std::string &path,
                                       unsigned long error) {
    throw std::runtime_error(std::string(operation) + " failed for " + path +
                             " (error " + std::to_string(error) + ")");
  }

  void *data_ = nullptr;
  size_t size_ = 0;
#ifdef _WIN32
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
#endif
};
''',
        encoding="utf-8",
        newline="",
    )


def apply_nvrtc_cpp(path: Path) -> None:
    replace_once(
        path,
        '#include <dlfcn.h>\n#include <nvrtc.h>\n',
        '''#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <nvrtc.h>
''',
    )
    replace_once(
        path,
        '''template <typename T>
void load_symbol(void *library, T &symbol, const char *name) {
  symbol = reinterpret_cast<T>(dlsym(library, name));
  if (symbol == nullptr) die(std::string("cannot load ") + name + ": " + dlerror());
}

void load_nvrtc(const std::string &configured_path) {
  void *library = dlopen(configured_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr)
    die("cannot load NVRTC from " + configured_path + ": " + dlerror());
''',
        '''#ifdef _WIN32
using NativeLibrary = HMODULE;
#else
using NativeLibrary = void *;
#endif

template <typename T>
void load_symbol(NativeLibrary library, T &symbol, const char *name) {
#ifdef _WIN32
  symbol = reinterpret_cast<T>(GetProcAddress(library, name));
  if (symbol == nullptr) die(std::string("cannot load ") + name + " from NVRTC");
#else
  symbol = reinterpret_cast<T>(dlsym(library, name));
  if (symbol == nullptr) die(std::string("cannot load ") + name + ": " + dlerror());
#endif
}

void load_nvrtc(const std::string &configured_path) {
#ifdef _WIN32
  NativeLibrary library = LoadLibraryA(configured_path.c_str());
  if (library == nullptr)
    die("cannot load NVRTC from " + configured_path + " (error " + std::to_string(GetLastError()) + ")");
#else
  NativeLibrary library = dlopen(configured_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr)
    die("cannot load NVRTC from " + configured_path + ": " + dlerror());
#endif
''',
    )


def apply_exports(path: Path) -> None:
    replace_once(
        path,
        'extern "C" int cubin_patch(const char *path, const char *mode, int dry, int backup) {\n',
        '#ifdef _WIN32\n#define HUMMING_EXPORT __declspec(dllexport)\n#else\n#define HUMMING_EXPORT\n#endif\n\n'
        'extern "C" HUMMING_EXPORT int cubin_patch(const char *path, const char *mode, int dry, int backup) {\n',
    )
    replace_once(
        path,
        'extern "C" int cubin_patch_buffer(uint8_t *data, size_t n, const char *mode, int dry) {\n',
        'extern "C" HUMMING_EXPORT int cubin_patch_buffer(uint8_t *data, size_t n, const char *mode, int dry) {\n',
    )


def apply(root: Path) -> None:
    sp = root / ".venv" / "Lib" / "site-packages" / "humming"
    paths = [sp / f for f in FILES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        orig = path.with_name(path.name + ".orig")
        if not orig.exists():
            shutil.copy2(path, orig)
    apply_device(paths[0])
    apply_nvrtc(paths[1])
    apply_cubin(paths[2])
    apply_ops(paths[3])
    apply_mapped_file(paths[4])
    apply_nvrtc_cpp(paths[5])
    apply_exports(paths[6])
    print(f"patched humming under {sp}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("pass --apply to modify the venv")
    apply(args.repo)


if __name__ == "__main__":
    main()
