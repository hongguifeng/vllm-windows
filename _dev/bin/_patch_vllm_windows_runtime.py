# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deploy tracked Flash-Next Python sources, or check them without writing."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RUNTIME_FILES = (
    "compilation/breakable_cudagraph.py",
    "entrypoints/launchers/api_server/entry.py",
    "model_executor/layers/quantization/inc/config_parser.py",
    "model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py",
    "model_executor/model_loader/default_loader.py",
    "models/qwen4_exp/config.py",
    "models/qwen4_exp/nvidia/model.py",
    "models/qwen4_exp/nvidia/model_state.py",
    "models/qwen4_exp/nvidia/mtp.py",
    "models/qwen4_exp/nvidia/ngram_embedding.py",
    "models/qwen4_exp/nvidia/ple_ssd.py",
    "triton_utils/prepared.py",
    "v1/worker/gpu/cudagraph_utils.py",
)


def deploy(source: Path, target: Path) -> None:
    """Copy a source while preserving the first installed version for rollback."""
    if target.exists() and target.read_bytes() == source.read_bytes():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    original = target.with_name(target.name + ".orig")
    if target.exists() and not original.exists():
        shutil.copy2(target, original)
    shutil.copy2(source, target)


def runtime_files(repo: Path, venv_root: Path) -> list[tuple[Path, Path]]:
    """Validate the installation and resolve the complete deployment plan."""
    venv = venv_root / ".venv"
    site = venv / "Lib" / "site-packages" / "vllm"
    if not (venv / "pyvenv.cfg").is_file() or not site.is_dir():
        raise FileNotFoundError(f"Install vLLM in {venv} before deploying its runtime")
    files = [(repo / "vllm" / name, site / name) for name in RUNTIME_FILES]
    for source, _ in files:
        if not source.is_file():
            raise FileNotFoundError(source)
    return files


def check_runtime(repo: Path, venv_root: Path) -> list[Path]:
    """Return missing or changed deployment targets without modifying them."""
    mismatches = []
    for source, target in runtime_files(repo, venv_root):
        if not target.is_file() or source.read_bytes().replace(
            b"\r\n", b"\n"
        ) != target.read_bytes().replace(b"\r\n", b"\n"):
            mismatches.append(target)
    return mismatches


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--venv-root",
        type=Path,
        default=repo / ".flashnext-root",
        help="Root containing .venv; defaults to the isolated .flashnext-root.",
    )
    parser.add_argument("--check", action="store_true", help="Audit without writing")
    args = parser.parse_args()
    if not args.check:
        for source, target in runtime_files(repo, args.venv_root):
            deploy(source, target)
    mismatches = check_runtime(repo, args.venv_root)
    if mismatches:
        for path in mismatches:
            print(f"Runtime differs from tracked source: {path}")
        raise SystemExit(1)
    print(f"Flash-Next runtime matches {len(RUNTIME_FILES)} tracked source files")


if __name__ == "__main__":
    main()
