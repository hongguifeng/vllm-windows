# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deploy changed Flash-Next Python sources into the installed vLLM package.

The production runtime is a non-editable install under ``.flashnext-root/.venv``,
so a Python-only change needs to be copied into ``site-packages`` to take effect.
The tracked 13 files are owned by ``_dev/bin/_patch_vllm_windows_runtime.py``;
this script handles anything else, keeps a ``.orig`` copy of the first installed
version for rollback, and refuses to touch a live service.
"""

from __future__ import annotations

import argparse
import shutil
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO / ".flashnext-root"
SITE = RUNTIME / ".venv" / "Lib" / "site-packages" / "vllm"
TRACKED = {
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
}


def service_is_live(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def resolve(name: str) -> tuple[Path, Path]:
    """Map a repo-relative (or vllm-relative) path to source and target."""
    rel = name
    if rel.startswith("vllm/"):
        rel = rel[len("vllm/") :]
    source = REPO / "vllm" / rel
    if not source.exists():
        raise FileNotFoundError(f"no such source file: {source}")
    return source, SITE / rel


def check(names: list[str]) -> list[str]:
    out = []
    for name in names:
        source, target = resolve(name)
        if not target.exists():
            out.append(f"MISSING target: {target}")
        elif target.read_bytes() != source.read_bytes():
            out.append(f"STALE target: {target}")
        else:
            out.append(f"up to date: {target}")
    return out


def deploy(source: Path, target: Path) -> None:
    if target.exists() and target.read_bytes() == source.read_bytes():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    original = target.with_name(target.name + ".orig")
    if target.exists() and not original.exists():
        shutil.copy2(target, original)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="changed files under vllm/")
    parser.add_argument("--check", action="store_true", help="compare only")
    parser.add_argument("--port", type=int, default=9393, help="service port")
    parser.add_argument(
        "--all",
        action="store_true",
        help="deploy every file git reports as changed or untracked under vllm/",
    )
    parser.add_argument(
        "--venv-root",
        type=Path,
        default=None,
        help="Root containing .venv; defaults to .flashnext-root. The Qwen3.8-27B "
        "service runs out of the repo's own .venv, whose root is the repo "
        "itself, so deploying to it needs --venv-root .",
    )
    args = parser.parse_args()

    global SITE
    root = args.venv_root if args.venv_root else Path(".flashnext-root")
    if not root.is_absolute():
        root = REPO / root
    SITE = root / ".venv" / "Lib" / "site-packages" / "vllm"
    if not SITE.is_dir():
        parser.error(f"no vLLM package installed under {SITE}")

    names = list(args.paths)
    if args.all:
        import subprocess

        out = subprocess.run(
            ["git", "status", "--porcelain", "--", "vllm/"],
            cwd=REPO,
            capture_output=True,
            text=True,
        ).stdout
        names = sorted({line[3:] for line in out.splitlines() if line[3:].strip()})
    if not names:
        parser.error("no paths given (and nothing changed under vllm/ with --all)")

    for name in names:
        if name.startswith("vllm/") and name[len("vllm/") :] in TRACKED:
            print(f"skip {name}: tracked by _dev/bin/_patch_vllm_windows_runtime.py")

    if args.check:
        for line in check(names):
            print(line)
        return 0

    if service_is_live(args.port):
        raise RuntimeError(
            f"a Flash-Next service is listening on :{args.port}; stop it first "
            "(stop_vllm.ps1) before deploying into site-packages"
        )

    deployed = []
    for name in names:
        rel = name[len("vllm/") :] if name.startswith("vllm/") else name
        if rel in TRACKED:
            continue
        source, target = resolve(name)
        deploy(source, target)
        deployed.append(str(target))
    print(f"deployed {len(deployed)} file(s) into {SITE}")
    for path in deployed:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
