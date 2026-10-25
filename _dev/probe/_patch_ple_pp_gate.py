"""Patch the installed Qwen4Exp model_state so PLE tolerates PP when the
operator explicitly allows it (mirrors the source tree's gate).

The installed copy gates on pipeline_parallel_size > 1 unconditionally, while
the tree also accepts VLLM_PLE_ALLOW_PP. This script restores the .bak, adds
the env bypass, and checks the file still parses.
"""
import ast
import pathlib
import shutil

P = pathlib.Path(
    r"D:\code\vllm-windows\.flashnext-root\.venv\Lib\site-packages"
    r"\vllm\models\qwen4_exp\nvidia\model_state.py"
)
BAK = P.with_suffix(".py.bak")
shutil.copyfile(BAK, P)
src = P.read_text(encoding="utf-8")

old = "    if vllm_config.parallel_config.pipeline_parallel_size > 1:\n"
new = (
    "    if (\n"
    "        vllm_config.parallel_config.pipeline_parallel_size > 1\n"
    '        and not os.environ.get("VLLM_PLE_ALLOW_PP")\n'
    "    ):\n"
)
assert src.count(old) == 1, f"expected exactly one gate, found {src.count(old)}"
src = src.replace(old, new)

if "\nimport os" not in src:
    lines = src.splitlines(keepends=True)
    last_import = max(
        i for i, line in enumerate(lines) if line.startswith(("import ", "from "))
    )
    lines.insert(last_import + 1, "import os\n")
    src = "".join(lines)

ast.parse(src)
P.write_text(src, encoding="utf-8")
print("patched ok")
start = src.index("if (")
print(src[start - 4 : start + 190])
