#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build an SWE-bench Verified subset for one difficulty stratum.

Three strata exist because a single number cannot answer "does model X differ
from model Y": a saturated stratum shows nothing, and an arbitrarily chosen hard
set can encode the very property you are trying to measure.

  easy       repo-quota sample across difficulty labels; expected to saturate.
  hard       instances carrying hard difficulty labels (>4h, 1-4h).
  structural selected by size of the reference patch and number of test cases,
             i.e. how much code and how many files the task forces you to touch.
             Independent of the difficulty labels, so a gap found on `hard` can
             be checked for replication here.
"""

import argparse
import collections
import json
import random

import pyarrow.parquet as pq

REGISTRY = "ghcr.io/epoch-research/swe-bench.eval.x86_64"
QUOTA = {
    "django/django": 12,
    "sympy/sympy": 5,
    "sphinx-doc/sphinx": 3,
    "matplotlib/matplotlib": 2,
    "scikit-learn/scikit-learn": 2,
    "astropy/astropy": 2,
    "pydata/xarray": 2,
    "pytest-dev/pytest": 2,
    "pylint-dev/pylint": 1,
    "psf/requests": 1,
}


def n_files(row):
    return (row.get("patch") or "").count("diff --git")


def patch_len(row):
    return len(row.get("patch") or "")


def f2p(row):
    try:
        return len(json.loads(row.get("FAIL_TO_PASS") or "[]"))
    except Exception:
        return 0


def structural_score(row):
    return (
        n_files(row) * 2 + min(patch_len(row), 6000) / 6000.0 + min(f2p(row), 10) / 3.0
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--mode", required=True, choices=["easy", "hard", "structural"])
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=0, help="cap for hard/structural modes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--exclude",
        default="",
        help="comma separated ids or path to a json subset to exclude",
    )
    p.add_argument(
        "--images-out", default="", help="write image refs here for pre-pulling"
    )
    a = p.parse_args()

    exclude = set()
    if a.exclude:
        if a.exclude.endswith(".json"):
            with open(a.exclude) as f:
                exclude |= {r["instance_id"] for r in json.load(f)}
        else:
            exclude |= {x for x in a.exclude.split(",") if x}

    rows = pq.read_table(a.parquet).to_pylist()
    if a.mode == "easy":
        rng = random.Random(a.seed)
        picked = []
        for repo, quota in QUOTA.items():
            by_diff = collections.defaultdict(list)
            for r in rows:
                if r["repo"] == repo and r["instance_id"] not in exclude:
                    by_diff[r["difficulty"]].append(r)
            for v in by_diff.values():
                rng.shuffle(v)
            easy = by_diff.get("<15 min fix", [])
            med = by_diff.get("15 min - 1 hour", [])
            hard = by_diff.get("1-4 hours", [])
            n_easy = max(1, int(quota * 0.7))
            picked.extend(
                (easy[:n_easy] + med[: max(0, quota - n_easy)] + hard[:2])[:quota]
            )
    elif a.mode == "hard":
        picked = [
            r
            for r in rows
            if r["difficulty"] in ("1-4 hours", ">4 hours")
            and r["instance_id"] not in exclude
        ]
        if a.n:
            picked = picked[: a.n]
    else:
        cand = [
            r
            for r in rows
            if r["instance_id"] not in exclude
            and r["difficulty"] not in ("1-4 hours", ">4 hours")
        ]
        cand.sort(key=lambda r: -structural_score(r))
        picked = cand[: a.n] if a.n else cand

    out = []
    for r in picked:
        r = dict(r)
        name = f"{REGISTRY}.{r['instance_id']}:latest"
        r["docker_image"] = r["image"] = name
        out.append(r)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    ids = [r["instance_id"] for r in out]
    print(f"wrote {a.out}: {len(out)} instances")
    print("difficulty:", dict(collections.Counter(r["difficulty"] for r in out)))
    print("repo:", dict(collections.Counter(r["repo"] for r in out).most_common()))
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    print(
        f"structural medians: files={med([n_files(r) for r in out]):.1f} "
        f"patch_len={med([patch_len(r) for r in out])} "
        f"f2p={med([f2p(r) for r in out]):.1f}"
    )
    overlap = exclude & set(ids)
    print("overlap with excluded set:", len(overlap))
    if a.images_out:
        with open(a.images_out, "w") as f:
            f.write("\n".join(sorted({r["image"] for r in out})) + "\n")
    print("ids written to", a.out)


if __name__ == "__main__":
    main()
