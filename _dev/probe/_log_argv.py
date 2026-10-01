#!/usr/bin/env python3
"""Reconstruct every launch from the surviving host logs.

The launcher script is not the authority. What matters is the argument set that
reached the API server, the build that answered, and how the KV cache was sized.
Each log carries three useful anchors:

* ``non-default args: {...}`` from ``api_utils`` -- the argument dict.
* ``Initializing a V1 LLM engine (<version>) with config:`` -- the build string,
  which identifies which working tree the interpreter imported from.
* ``enforce_eager=``, ``cudagraph_mode`` and ``quantization=`` on that same line.

Logs are encoded inconsistently (ANSI prefix then UTF-16LE), so decode them the
way ``_log_decode.py`` does.

Usage:
    python _log_argv.py             # group launches by identical signature
    python _log_argv.py --verbose   # list every log with its signature
"""
import ast
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

LOG_DIR = Path(r"D:\code\vllm-windows\_dev\out\logs")
SALIENT = (
    "model",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "kv_cache_memory_bytes",
    "quantization",
    "enforce_eager",
    "async_scheduling",
    "speculative_config",
    "additional_config",
)
PLE_KEYS = (
    "ple_ssd_cache_mb",
    "ple_ssd_workers",
    "ple_ssd_io_depth",
    "ple_ssd_prefetch_tokens",
)


def _decoder():
    here = Path(__file__).with_name("_log_decode.py")
    spec = importlib.util.spec_from_file_location("_log_decode", here)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch_of(path: Path) -> dict | None:
    """Return the launch signature recorded in one log, or None if it never started."""
    text = _decoder().decode(path.read_bytes())
    args = None
    for line in text.splitlines():
        if "non-default args: {" in line:
            blob = line.split("non-default args:", 1)[1].strip()
            try:
                args = ast.literal_eval(blob)
            except (ValueError, SyntaxError):
                return None
            break
    engine = None
    for line in text.splitlines():
        if "Initializing a V1 LLM engine" in line:
            engine = line
            break
    if args is None and engine is None:
        return None
    info = {"args": args or {}, "version": "?", "eager": "?", "graph": "?"}
    if engine:
        match = re.search(r"LLM engine \(([^)]*)\)", engine)
        if match:
            info["version"] = match.group(1)
        info["eager"] = (re.search(r"enforce_eager=(\w+)", engine) or [None, "?"])[1]
        info["quant"] = (re.search(r"quantization=(\w+)", engine) or [None, "?"])[1]
        graph = re.search(r"cudagraph_mode': <CUDAGraphMode\.(\w+)", engine)
        info["graph"] = graph.group(1) if graph else "none"
    return info


def summarize(info: dict) -> list:
    """Reduce a launch to the knobs that distinguish one arm from another."""
    args = info["args"]
    rows = [("version", info["version"]), ("graph", info["graph"]),
            ("eager", info["eager"]), ("quant", info.get("quant", args.get("quantization", "?")))]
    for key in SALIENT:
        if key in args:
            value = args[key]
            if key == "speculative_config" and isinstance(value, dict):
                value = f"{value.get('method')} x{value.get('num_speculative_tokens')}"
            elif key == "additional_config" and isinstance(value, dict):
                value = " ".join(
                    f"{name.split('_')[-1]}={value[name]}"
                    for name in PLE_KEYS if name in value
                ) or "ple off"
            elif key == "model":
                value = Path(str(value)).name
            rows.append((key, value))
    return sorted(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    verbose = "--verbose" in sys.argv
    paths = sorted(LOG_DIR.glob("*.log"))
    groups = Counter()
    spans = {}
    max_lens = Counter()
    async_arms = 0
    for path in paths:
        info = launch_of(path)
        if info is None:
            continue
        key = tuple(summarize(info))
        groups[key] += 1
        spans.setdefault(key, []).append(path.stem.replace("flashnext_", ""))
        max_lens[str(info["args"].get("max_model_len", "?"))] += 1
        async_arms += bool(info["args"].get("async_scheduling"))
    started = sum(groups.values())
    versions = Counter()
    for key, count in groups.items():
        versions[dict(key)["version"]] += count
    print(f"scanned {len(paths)} logs, {started} reached the engine")
    for version, count in versions.most_common():
        print(f"    {count:3d} launches from build {version}")
    print()
    print("\ncontext length across launches:")
    for length, count in max_lens.most_common():
        print(f"    {count:3d} launches with max_model_len {length}")
    print(f"    {async_arms} launches asked for async scheduling\n")
    for key, count in groups.most_common():
        stamp = sorted(spans[key])
        print(f"[{count} launches] {stamp[0]} .. {stamp[-1]}")
        for name, value in key:
            print(f"    {name:23s} {value}")
        print()
    if verbose:
        print("per log:")
        for path in paths:
            info = launch_of(path)
            tag = "never started" if info is None else info["version"]
            print(f"    {path.name}  {tag}")


if __name__ == "__main__":
    main()
