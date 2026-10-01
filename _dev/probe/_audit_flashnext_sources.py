# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit venv-only changes and replay Humming patches without touching the venv."""

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import regex as re

REPO = Path(__file__).resolve().parents[2]


def normalized(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


class WithoutDocstrings(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if (
            isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node


def python_code(path):
    return ast.dump(WithoutDocstrings().visit(ast.parse(path.read_bytes())))


def replay_humming(site):
    spec = importlib.util.spec_from_file_location(
        "humming_patch", REPO / "_dev/bin/_patch_humming_windows.py"
    )
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    with tempfile.TemporaryDirectory(prefix="flashnext-replay-") as directory:
        root = Path(directory)
        package = root / ".venv/Lib/site-packages/humming"
        for name in patch.FILES:
            original = (site / "humming" / name).with_name(name.name + ".orig")
            target = package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        patch.apply(root)
        first = {name: normalized(package / name) for name in patch.FILES}
        patch.apply(root)
        for name in patch.FILES:
            if normalized(package / name) != first[name]:
                raise RuntimeError(f"Humming patch is not idempotent: {name}")
            if normalized(site / "humming" / name) != first[name]:
                raise RuntimeError(f"Humming patch does not reproduce installed {name}")
        return [name.as_posix() for name in patch.FILES]


def check_flash_attention(site, installed_only):
    """Validate Python files installed by the tracked CMake FetchContent recipe."""
    checkout = REPO / ".deps/vllm-flash-attn-src"
    if not checkout.is_dir():
        return None
    cmake = (REPO / "cmake/external_projects/vllm_flash_attn.cmake").read_text(
        encoding="utf-8"
    )
    pinned = re.search(r"GIT_TAG ([0-9a-f]{40})", cmake).group(1)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if revision != pinned:
        raise RuntimeError(f"FlashAttention checkout {revision} != pinned {pinned}")
    reproduced = []
    prefix = "vllm/vllm_flash_attn/"
    for name in installed_only:
        if not name.startswith(prefix):
            continue
        tail = name.removeprefix(prefix)
        source = "flash_attn/" if tail.startswith("cute/") else "vllm_flash_attn/"
        source += tail
        content = subprocess.check_output(
            ["git", "show", "HEAD:" + source], cwd=checkout
        ).replace(b"\r\n", b"\n")
        if tail.startswith("cute/"):
            content = content.replace(b"flash_attn.cute", b"vllm.vllm_flash_attn.cute")
        if normalized(site / name) != content:
            raise RuntimeError(f"Installed dependency differs from pinned Git: {name}")
        reproduced.append(name)
    return {"revision": revision, "reproduced": reproduced}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, default=REPO / ".venv-flashnext")
    parser.add_argument("--replay-humming", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=REPO / "_dev/out/flashnext_source_audit.json"
    )
    args = parser.parse_args()
    site = args.venv / "Lib/site-packages"
    if not (site / "vllm").is_dir():
        parser.error(f"vLLM is not installed under {site}")
    tracked = set(
        subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    )
    report = {
        "vllm_text_differences": [],
        "vllm_code_differences": [],
        "installed_only_python": [],
        "generated_python": [],
        "vllm_backups": [],
        "humming_backups": [],
        "versions": {
            dist.metadata["Name"]: dist.version
            for dist in metadata.distributions(path=[str(site)])
        },
    }
    for path in sorted((site / "vllm").rglob("*.py")):
        relative = path.relative_to(site).as_posix()
        source = REPO / relative
        if relative in tracked and source.is_file():
            if normalized(source) != normalized(path):
                report["vllm_text_differences"].append(relative)
                if python_code(source) != python_code(path):
                    report["vllm_code_differences"].append(relative)
        elif relative == "vllm/_version.py":
            report["generated_python"].append(relative)
        else:
            report["installed_only_python"].append(relative)
    for package in ("vllm", "humming"):
        for backup in sorted((site / package).rglob("*.orig")):
            path = backup.with_name(backup.name[:-5])
            relative = path.relative_to(site).as_posix()
            source = REPO / relative
            report[package + "_backups"].append(
                {
                    "path": relative,
                    "modified_since_backup": path.read_bytes() != backup.read_bytes(),
                    "in_git": relative in tracked,
                    "matches_source": source.exists()
                    and normalized(source) == normalized(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    report["flash_attention_git_sources"] = check_flash_attention(
        site, report["installed_only_python"]
    )
    if args.replay_humming:
        report["humming_replayed_twice"] = replay_humming(site)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit: {args.output}")
    for key in ("vllm_text_differences", "vllm_code_differences"):
        print(f"{key}: {report[key]}")
    if args.replay_humming:
        print(
            f"Replayed all {len(report['humming_replayed_twice'])} "
            "Humming patches twice"
        )
    if report["vllm_code_differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
