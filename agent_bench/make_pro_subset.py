#!/usr/bin/env python3
"""Materialise a 20-task SWE-Bench Pro V2 subset (10 HARD-51 + 10 medium).

Clusters tasks in a few repositories so the per-task images share layers, which
keeps the pull cost down on a slow link.

    python make_pro_subset.py --tasks v2/tasks --hard v2/hard51_ids.txt --out datasets/pro20
"""

import argparse
import json
import shutil
from pathlib import Path

# repo -> (hard picks, medium picks)
PLAN = {
    "ansible": (4, 4),
    "internetarchive": (3, 4),
    "flipt-io": (2, 0),
    "gravitational": (1, 0),
    "qutebrowser": (0, 2),
}


def repo_of(instance_id: str) -> str:
    return instance_id.removeprefix("instance_").split("__", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("v2/tasks"))
    parser.add_argument("--hard", type=Path, default=Path("v2/hard51_ids.txt"))
    parser.add_argument("--out", type=Path, default=Path("datasets/pro20"))
    args = parser.parse_args()

    task_dirs = sorted(path for path in args.tasks.iterdir() if path.is_dir())
    by_repo = {}
    for path in task_dirs:
        by_repo.setdefault(repo_of(path.name), []).append(path)

    hard_ids = {
        line.strip() for line in args.hard.read_text().splitlines() if line.strip()
    }
    hard_dirs = [path for path in task_dirs if path.name in hard_ids]
    medium_dirs = [path for path in task_dirs if path.name not in hard_ids]

    picked = []
    for repo, (hard_count, medium_count) in PLAN.items():
        hard = [path for path in hard_dirs if repo_of(path.name) == repo]
        medium = [path for path in medium_dirs if repo_of(path.name) == repo]
        picked.extend(hard[:hard_count])
        picked.extend(medium[:medium_count])

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for path in picked:
        shutil.copytree(path, args.out / path.name)
    (args.out / "ids.txt").write_text("\n".join(path.name for path in picked) + "\n")

    summary = {}
    for path in picked:
        repo = repo_of(path.name)
        summary[repo] = summary.get(repo, 0) + 1
    print(f"selected {len(picked)} tasks into {args.out}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
