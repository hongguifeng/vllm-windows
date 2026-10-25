#!/usr/bin/env python3
"""Per-benchmark docker image usage from `docker system df -v` (unique sizes)."""

import re
import subprocess

UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def parse_size(text: str) -> float:
    text = text.strip()
    match = re.match(r"([\d.]+)\s*([KMGTP]?B)", text)
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
    out = subprocess.check_output(["docker", "system", "df", "-v"], text=True)
    in_images = False
    buckets: dict[str, dict[str, float]] = {}
    for line in out.splitlines():
        if line.startswith("Images space usage:"):
            in_images = True
            continue
        if in_images and line.strip().startswith("Containers space usage"):
            break
        if not in_images:
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] == "IMAGE":
            continue
        image = parts[0]
        # SIZE SHARED UNIQUE CONTAINERS are the last four columns
        size, shared, unique = (parse_size(parts[-4]), parse_size(parts[-3]), parse_size(parts[-2]))
        if "epoch-research" in image:
            key = "SWE-bench Verified (ghcr.io/epoch-research)"
        elif "swe-bench_pro-v2" in image:
            key = "SWE-Bench Pro V2 (ghcr.io/scaleapi)"
        elif "swe-bench-202605" in image:
            key = "DeepSWE 基础镜像 (public.ecr.aws)"
        elif re.search(r"__(main|pier-egress-proxy)$|__.*-main$", image):
            key = "Pier 临时 trial 镜像"
        else:
            key = "其他/基础镜像"
        bucket = buckets.setdefault(key, {"count": 0, "size": 0.0, "shared": 0.0, "unique": 0.0})
        bucket["count"] += 1
        bucket["size"] += size
        bucket["shared"] += shared
        bucket["unique"] += unique
    print(f"{'类别':<42}{'镜像数':>6}{'名义大小':>12}{'独占':>12}")
    total_unique = 0.0
    for key, bucket in sorted(buckets.items(), key=lambda kv: -kv[1]["size"]):
        total_unique += bucket["unique"]
        print(f"{key:<42}{int(bucket['count']):>6}{human(bucket['size']):>12}{human(bucket['unique']):>12}")
    print(f"{'合计（独占，去重后）':<42}{'':>6}{'':>12}{human(total_unique):>12}")


if __name__ == "__main__":
    main()
