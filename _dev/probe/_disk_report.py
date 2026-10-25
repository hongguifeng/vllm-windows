#!/usr/bin/env python3
"""Report disk usage of the benchmark assets (docker images + datasets + job outputs)."""

import re
import subprocess

UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def parse_size(text: str) -> float:
    match = re.match(r"([\d.]+)\s*([KMGTP]?B)", text.strip())
    if not match:
        return 0.0
    return float(match.group(1)) * UNITS[match.group(2)]


def human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1000 or unit == "TB":
            return f"{num:.1f} {unit}"
        num /= 1000
    return f"{num:.1f} TB"


def main() -> None:
    out = subprocess.check_output(
        ["docker", "images", "--format", "{{.Repository}}\t{{.Size}}"], text=True
    )
    buckets: dict[str, float] = {}
    total = 0.0
    for line in out.splitlines():
        if "\t" not in line:
            continue
        repo, size = line.split("\t", 1)
        value = parse_size(size)
        total += value
        if "epoch-research" in repo:
            key = "swe-bench Verified 镜像 (ghcr/epoch-research)"
        elif "swe-bench_pro-v2" in repo:
            key = "SWE-Bench Pro V2 镜像 (ghcr/scaleapi)"
        elif "swe-bench-202605" in repo:
            key = "DeepSWE 镜像 (ecr public)"
        elif repo == "<none>":
            key = "dangling 镜像"
        else:
            key = f"其他镜像 ({repo.split(':')[0].split('/')[0]})"
        buckets[key] = buckets.get(key, 0.0) + value
    print("== Docker 镜像 ==")
    for key, value in sorted(buckets.items(), key=lambda item: -item[1]):
        print(f"  {key:<45} {human(value):>10}")
    print(f"  {'镜像总计':<45} {human(total):>10}")
    print()
    print("== 数据集 / 仓库 / 运行产物 ==")
    for path in (
        "/home/hong/bench",
        "/home/hong/SWE-bench_Pro-os",
        "/home/hong/deep-swe",
        "/home/hong/SWE-bench-Live",
        "/home/hong/bench/runEH",
        "/home/hong/bench/runES40",
        "/home/hong/SWE-bench_Pro-os/jobs",
        "/home/hong/deep-swe/jobs",
    ):
        try:
            size = subprocess.check_output(["du", "-sb", path], text=True).split()[0]
            print(f"  {path:<45} {human(float(size)):>10}")
        except subprocess.CalledProcessError:
            print(f"  {path:<45} {'(missing)':>10}")


if __name__ == "__main__":
    main()
