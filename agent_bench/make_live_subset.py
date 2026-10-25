#!/usr/bin/env python3
"""Build a stratified hard subset from the SWE-bench-Live `verified` split.

Picks tasks whose reference patch touches >=2 files and >=15 lines (the harder
end of the split), spread round-robin across repositories, and writes a json
that mini-swe-agent's swebench runner can consume directly.

    .venv/bin/python make_live_subset.py --count 30 --out subset_live30.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = "/home/hong/bench/swe_live_verified.parquet"


def image_name(instance_id: str) -> str:
    return (
        "starryzhang/sweb.eval.x86_64."
        f"{instance_id.lower().replace('__', '_1776_')}:latest"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--min-files", type=int, default=2)
    parser.add_argument("--min-lines", type=int, default=15)
    parser.add_argument("--out", default="/home/hong/bench/subset_live30.json")
    args = parser.parse_args()

    frame = pd.read_parquet(args.parquet)
    frame = frame[
        [bool(entry) and entry.get("files", 0) >= args.min_files for entry in frame.difficulty]
    ]
    frame = frame[
        [bool(entry) and entry.get("lines", 0) >= args.min_lines for entry in frame.difficulty]
    ]
    by_repo = {
        repo: group.sort_values("created_at")
        for repo, group in frame.groupby("repo")
    }
    picked = []
    index = 0
    while len(picked) < args.count:
        added = False
        for repo in sorted(by_repo):
            group = by_repo[repo]
            if index < len(group) and len(picked) < args.count:
                picked.append(group.iloc[index])
                added = True
        if not added:
            break
        index += 1

    rows = []
    for row in picked:
        instance_id = row["instance_id"]
        rows.append(
            {
                "instance_id": instance_id,
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "version": None,
                "environment_setup_commit": None,
                "created_at": str(row["created_at"]),
                "problem_statement": row["problem_statement"],
                "hints_text": row["hints_text"],
                "patch": row["patch"],
                "test_patch": row["test_patch"],
                "FAIL_TO_PASS": list(row["FAIL_TO_PASS"]),
                "PASS_TO_PASS": list(row["PASS_TO_PASS"]),
                "test_cmds": list(row["test_cmds"]),
                "log_parser": row["log_parser"],
                "difficulty": row["difficulty"],
                "docker_image": image_name(instance_id),
                "image": image_name(instance_id),
            }
        )
    Path(args.out).write_text(json.dumps(rows, indent=1))
    repos = {}
    for row in rows:
        repos[row["repo"]] = repos.get(row["repo"], 0) + 1
    print(f"wrote {len(rows)} instances to {args.out}")
    print(f"repos: {len(repos)}")
    for repo, count in sorted(repos.items(), key=lambda item: -item[1]):
        print(f"  {repo}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
