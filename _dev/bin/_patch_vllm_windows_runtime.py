"""Apply the small vLLM Python deployment patches required on Windows."""
from pathlib import Path
import shutil


def deploy_file(source: Path, target: Path) -> None:
    if not target.with_name(target.name + ".orig").exists():
        shutil.copy2(target, target.with_name(target.name + ".orig"))
    shutil.copy2(source, target)


def patch(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new.strip() in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f"expected one match in {path}")
    orig = path.with_name(path.name + ".orig")
    if not orig.exists():
        shutil.copy2(path, orig)
    path.write_text(text.replace(old, new), encoding="utf-8", newline="")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    site = repo / ".venv" / "Lib" / "site-packages" / "vllm"
    deploy_file(
        repo / "vllm" / "models" / "qwen4_exp" / "nvidia" / "ple_ssd.py",
        site / "models" / "qwen4_exp" / "nvidia" / "ple_ssd.py",
    )
    patch(
        site / "entrypoints" / "launchers" / "api_server" / "entry.py",
        "def main():\n    import uvloop\n    import platform\n",
        "def main():\n    import platform\n",
    )
    print("Windows vLLM runtime patches present")


if __name__ == "__main__":
    main()
