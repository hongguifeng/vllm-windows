"""Fix humming's Windows path leak so its precompiled helpers are found again.

`humming/utils/jit.py::hash_path_content` derives the "relative" name of a file
source with `path.split("/")[-1]`. On Windows the path uses backslashes, so the
split does nothing and the *whole absolute path* becomes part of the hash. Every
`~/.humming/cache` key and every `manifest.json` entry for a file artifact
(`_device_info.pyd`, `libcubinpatch.dll`, `nvrtc_compile.exe`) therefore depends
on where the venv lives:

* a venv copied or moved from its build location misses the precompiled
  manifest AND the cache, and
* the fallback builds with a POSIX command line (`g++ -O3 -fPIC -shared`),
  which cannot work on Windows -- the engine dies at model load with
  ``FileNotFoundError`` before KV profiling.

This patches the hash to use the basename (content + file name, path-free) and
rewrites the artifact manifest with the patched hash, so the shipped artifacts
are used and nothing is compiled. Idempotent; `--check` only reports.

  .venv\\Scripts\\python.exe _dev\\probe\\_patch_humming_win_path.py --check
  .venv\\Scripts\\python.exe _dev\\probe\\_patch_humming_win_path.py --python <venv>\\Scripts\\python.exe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OLD = '        filename = path.split("/")[-1] if releative else path\n'
NEW = "        filename = os.path.basename(path) if releative else path\n"

# Windows artifact names against the source each is built from.
ARTIFACTS = (
    ("_device_info.pyd", "csrc/device_info.cpp"),
    ("libcubinpatch.dll", "csrc/patch_cubin.cpp"),
    ("libhumming_launcher.so", "csrc/launcher"),
    ("nvrtc_compile.exe", "csrc/nvrtc_compile.cpp"),
)

REWRITE = r'''
import json
from pathlib import Path
import humming.utils.jit as J

ham = Path(J.__file__).parents[1]
native = ham / "_native" / J.get_native_arch()
artifacts = {
    "_device_info.pyd": "csrc/device_info.cpp",
    "libcubinpatch.dll": "csrc/patch_cubin.cpp",
    "libhumming_launcher.so": "csrc/launcher",
    "nvrtc_compile.exe": "csrc/nvrtc_compile.cpp",
}
manifest = {}
for name, rel in artifacts.items():
    src = ham / rel
    if not (native / name).is_file():
        raise SystemExit(f"missing artifact {native / name}")
    manifest[name] = J.hash_path_content(str(src), releative=True, text_only=False)
    hit = J.get_precompiled_artifact_path(str(src), name)
    print(f"  {name:<24} hash={manifest[name]} precompiled={'HIT' if hit else 'MISS'}")
(native / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
print("  manifest:", native / "manifest.json")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter of the venv to patch")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    python = Path(args.python)
    if not python.is_file():
        print(f"ERROR: no interpreter at {python}")
        return 1

    root = python.parents[1]
    jit = root / "Lib" / "site-packages" / "humming" / "utils" / "jit.py"
    if not jit.is_file():
        print(f"ERROR: humming is not installed in {root}")
        return 1

    text = jit.read_text(encoding="utf-8")
    patched = NEW in text
    print(f"venv      : {root}")
    print(f"jit.py    : {'already patched' if patched else 'needs patch'}")
    if not patched:
        if OLD not in text:
            print("ERROR: the hash line does not match; humming changed upstream")
            return 1
        if args.check:
            print("(check only)")
            return 1
        jit.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
        print("jit.py    : patched")

    if args.check:
        return 0 if patched else 1
    print("manifest  : rewriting")
    result = subprocess.run([str(python), "-c", REWRITE], text=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
