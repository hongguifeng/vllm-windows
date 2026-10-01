# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only regression tests for replayable Flash-Next deployment patches."""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

BIN = Path(__file__).resolve().parents[1] / "bin"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, BIN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_script("_patch_vllm_windows_runtime")
humming = load_script("_patch_humming_windows")


class RuntimeDeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.runtime_root = self.root / "runtime"
        venv = self.runtime_root / ".venv"
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = fixture", encoding="utf-8")
        for name in runtime.RUNTIME_FILES:
            source = self.repo / "vllm" / name
            target = venv / "Lib" / "site-packages" / "vllm" / name
            source.parent.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("# tracked revision\n", encoding="utf-8")
            target.write_text("# installed original\n", encoding="utf-8")

    def test_redeploy_preserves_original_for_rollback(self):
        files = runtime.runtime_files(self.repo, self.runtime_root)
        for source, target in files:
            original = target.read_bytes()
            runtime.deploy(source, target)
            source.write_text("# next tracked revision\n", encoding="utf-8")
            runtime.deploy(source, target)
            runtime.deploy(source, target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(
                target.with_name(target.name + ".orig").read_bytes(),
                original,
            )
        self.assertEqual(runtime.check_runtime(self.repo, self.runtime_root), [])

    def test_audit_reports_drift_and_missing_files_without_writing(self):
        files = runtime.runtime_files(self.repo, self.runtime_root)
        for source, target in files:
            runtime.deploy(source, target)
        files[0][1].write_text("# venv-only edit\n", encoding="utf-8")
        files[1][1].unlink()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(
            set(runtime.check_runtime(self.repo, self.runtime_root)),
            {files[0][1], files[1][1]},
        )
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_source_fails_before_deployment(self):
        (self.repo / "vllm" / runtime.RUNTIME_FILES[-1]).unlink()
        with self.assertRaises(FileNotFoundError):
            runtime.runtime_files(self.repo, self.runtime_root)
        self.assertFalse(list(self.runtime_root.rglob("*.orig")))


class HummingPatchTests(unittest.TestCase):
    def test_cccl_include_patch_can_be_reapplied(self):
        original = (
            "import subprocess\n"
            '        env = filter_cuda_paths(required_headers=["cuda_runtime.h"])\n'
            '        return list(cls.include_dirs()) + list(env["include_paths"])\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compiler.py"
            path.write_text(original, encoding="utf-8")
            humming.apply_compiler(path)
            patched = path.read_bytes()
            humming.apply_compiler(path)
            self.assertEqual(path.read_bytes(), patched)
            self.assertIn(b'"include" / "cccl"', patched)
            self.assertEqual(path.read_text(encoding="utf-8").count("import sys"), 1)


@unittest.skipUnless(os.name == "nt" and shutil.which("pwsh"), "requires Windows PS7")
class SetupCommandTests(unittest.TestCase):
    def test_setup_defaults_to_plan_without_creating_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(BIN / "_setup_flashnext.ps1"),
                    "-Repo",
                    directory,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Provision", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_execute_refuses_an_occupied_port_before_provisioning(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = str(BIN / "_setup_flashnext.ps1").replace("'", "''")
            repo = directory.replace("'", "''")
            script = f"""
$ErrorActionPreference = 'Stop'
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
try {{
    $port = $listener.LocalEndpoint.Port
    try {{
        & '{setup}' -Repo '{repo}' -Execute -Port $port
        throw 'Did not refuse the occupied port'
    }} catch {{
        if ($_.Exception.Message -notlike "*Port $port is in use*") {{ throw }}
    }}
}} finally {{ $listener.Stop() }}
"""
            subprocess.run(
                ["pwsh", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
