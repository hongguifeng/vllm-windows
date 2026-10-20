# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the EXL3 extension vllm._exl3_C (SM 8.0, BF16) without rebuilding vLLM.

EXL3 checkpoints need it (vllm/sm80/kernels/exl3/ops.py). Requires CUDA-enabled
PyTorch, a matching CUDA toolkit, a C++17/20 compiler and Ninja. No visible GPU
is needed. The output directory is explicit; an installed precompiled engine is
retained. module.cpp registers the operators with the non-stable TORCH_LIBRARY
API, so the library works only with the PyTorch it was built against:
exl3.ops.ensure_extension compares its build_info() with the running PyTorch and
refuses another build.

Windows: MSVC 2022 + CUDA 13.x. The POSIX host reader (ple_read.cpp) is
replaced by ple_read_win.cpp, `windows.h`'s `small`/`large` macros are removed
by the forced include win_compat.h, and CUDA 13's CCCL requires the conforming
preprocessor (/Zc:preprocessor).
"""

import argparse
import os
import shutil
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSRC = HERE.parent
ROOT = CSRC.parent
NAME = "_exl3_C"

_COMMON_SOURCES = (
    "moe_prefill.cu",
    "dense.cu",
    "dense_splitk.cu",
    "gemm.cu",
    "moe.cu",
    "moe_v2.cu",
    "ple.cu",
    "dense_mma.cu",
    "reference.cu",
    "module.cpp",
    "dense_prefill_tc.cu",
)
_LINUX_SOURCES = _COMMON_SOURCES + ("ple_read.cpp",)
_WINDOWS_SOURCES = _COMMON_SOURCES + ("ple_read_win.cpp",)

# The flags the former CMake target compiled with (VLLM_GPU_FLAGS for SM 8.0
# plus its definitions), so both builds generate the same code.
_DEFINES = ["-DNDEBUG", "-DUSE_CUDA", "-DENABLE_FP8", "-DENABLE_SCALED_MM_C2X=1"]
_NVCC_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]


def _windows_flags() -> tuple[list[str], list[str], list[str]]:
    compat = HERE / "win_compat.h"
    defines = ["-DNDEBUG", "-DUSE_CUDA", "-DENABLE_FP8"]
    cuda_flags = [
        "-O3",
        "-std=c++20",
        f"--pre-include={compat}",
        *defines,
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-Xcompiler=/bigobj",
        "-Xcompiler=/utf-8",
        "-Xcompiler=/Zc:preprocessor",
    ]
    cpp_flags = [
        "/O2",
        "/std:c++20",
        "/bigobj",
        "/utf-8",
        "/Zc:preprocessor",
        f"/FI{compat}",
        *defines,
    ]
    return cuda_flags, cpp_flags, list(_WINDOWS_SOURCES)


def _linux_flags() -> tuple[list[str], list[str], list[str]]:
    maps = [
        f"-ffile-prefix-map={ROOT}=vllm",
    ]
    cuda_flags = [
        *_NVCC_FLAGS,
        "-std=c++20",
        *_DEFINES,
        *[f"-Xcompiler={flag}" for flag in maps],
    ]
    cpp_flags = ["-O3", "-std=gnu++20", *_DEFINES, *maps]
    return cuda_flags, cpp_flags, list(_LINUX_SOURCES)


def build(out: Path, build_dir: Path, verbose: bool) -> None:
    # Never infer architectures from visible devices or initialise CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get(
        "PATH", ""
    )
    windows = sys.platform == "win32"
    if windows:
        # torch's compiler probe decodes cl.exe output with the OEM code page and
        # fails on localized (e.g. Chinese) MSVC banners. The extension checks
        # the ABI it was built with itself (build_info()), so skip the probe.
        os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")
    import torch
    from torch.utils import cpp_extension

    if torch.version.cuda is None or cpp_extension.CUDA_HOME is None:
        raise RuntimeError("A CUDA-enabled PyTorch and matching CUDA toolkit are required")
    build_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cuda_flags, cpp_flags, sources = (
        _windows_flags() if windows else _linux_flags()
    )
    cpp_extension.load(
        name=NAME,
        sources=[str(HERE / source) for source in sources],
        extra_include_paths=[str(CSRC)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=cpp_flags,
        build_directory=str(build_dir),
        # Windows needs a real Python module for PyInit__exl3_C and the .pyd
        # suffix; on Linux the recipe ships a plain shared object.
        is_python_module=windows,
        verbose=verbose,
    )
    if windows:
        source = build_dir / f"{NAME}.pyd"
        destination = out / f"{NAME}.pyd"
    else:
        source = build_dir / f"{NAME}.so"
        # Match the source-build install path; tagged files outrank plain .so imports.
        destination = out / f"{NAME}.abi3.so"
    with tempfile.NamedTemporaryFile(dir=out, prefix=NAME, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if not windows:
        for suffix in EXTENSION_SUFFIXES:
            obsolete = out / (NAME + suffix)
            if obsolete != destination:
                obsolete.unlink(missing_ok=True)
    print(f"{destination} ({destination.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for the extension (the tree's vllm/)")
    parser.add_argument("--build-dir", type=Path,
                        help="persistent build directory (default: temporary)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.build_dir is None:
        with tempfile.TemporaryDirectory(prefix="exl3_") as directory:
            build(args.out.resolve(), Path(directory), args.verbose)
    else:
        build(args.out.resolve(), args.build_dir.resolve(), args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
