#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact McNemar comparison across SWE-bench arms, plus stratum unions.

The two-sided exact p value is 2 * P(X >= max(b, c)) with X ~ Binomial(b+c, 1/2),
b = instances only A resolved, c = instances only B resolved. Using min(b, c)
instead of max(b, c) is a common implementation bug: it returns a large
probability and then gets clamped to 1.0, so every comparison looks null.

Instances are classified from the report's id partitions:
resolved_ids, unresolved_ids, empty_patch_ids. Anything outside those three
lists is reported as unknown rather than silently counted as a failure.
"""

import argparse
import json
import math


def load(path):
    with open(path) as f:
        r = json.load(f)
    resolved = set(r.get("resolved_ids", []))
    unresolved = set(r.get("unresolved_ids", []))
    empty = set(r.get("empty_patch_ids", []))
    set(r.get("error_ids", [])) | set(r.get("incomplete_ids", []))
    if isinstance(r.get("error_instances"), int) and r["error_instances"]:
        print(
            f"  note: {path} has {r['error_instances']} error instances "
            f"excluded from pairing"
        )
    allids = resolved | unresolved | empty
    if r.get("total_instances") and r["total_instances"] != len(allids):
        print(
            f"  note: {path} total_instances={r['total_instances']} "
            f"but partition size={len(allids)}"
        )
    return resolved, allids, r


def exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    hi = max(b, c)
    tail = sum(math.comb(n, k) * 0.5**n for k in range(hi, n + 1))
    return min(1.0, 2 * tail)


def chi_sq(b, c):
    n = b + c
    if not n:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi / 2)) if chi else 1.0


def describe(label, res_a, ids_a, res_b, ids_b):
    ids = sorted(ids_a | ids_b)
    n = len(ids)
    if not n:
        print(f"{label}: no instances")
        return None
    a_only = sum(1 for i in ids if i in res_a and i not in res_b)
    b_only = sum(1 for i in ids if i in res_b and i not in res_a)
    na = sum(1 for i in ids if i in res_a)
    nb = sum(1 for i in ids if i in res_b)
    print(
        f"{label:<18} n={n:3d}  A={na:3d} ({100.0 * na / n:5.1f}%)  "
        f"B={nb:3d} ({100.0 * nb / n:5.1f}%)  gap={100.0 * (na - nb) / n:+6.1f} pp  "
        f"discordant {a_only}/{b_only}  exact p={exact_p(a_only, b_only):.4f}  "
        f"chi2 p={chi_sq(a_only, b_only):.4f}"
    )
    if ids_a & ids_b and len(ids_a & ids_b) != len(ids):
        print(
            "  note: pair overlap differs; only paired instances contribute "
            "to discordant counts"
        )
    return set(ids), res_a, res_b, a_only, b_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar=("LABEL", "A", "B"),
        nargs=3,
        help="label, report A path, report B path",
    )
    p.add_argument(
        "--union",
        action="append",
        metavar="LABEL",
        help="labels to combine; strata must be disjoint",
    )
    a = p.parse_args()

    store = {}
    for label, pa, pb in a.pair:
        res_a, ids_a, _ = load(pa)
        res_b, ids_b, _ = load(pb)
        store[label] = describe(label, res_a, ids_a, res_b, ids_b)

    for group in a.union or []:
        labels = group.split(",")
        ids = set()
        res_a = set()
        res_b = set()
        for lab in labels:
            rec = store.get(lab.strip())
            if not rec:
                print(f"union: unknown label {lab}")
                continue
            ids |= rec[0]
            res_a |= rec[1]
            res_b |= rec[2]
        n = len(ids)
        only_a = sum(1 for i in ids if i in res_a and i not in res_b)
        only_b = sum(1 for i in ids if i in res_b and i not in res_a)
        na = sum(1 for i in ids if i in res_a)
        nb = sum(1 for i in ids if i in res_b)
        gap = 100.0 * (na - nb) / n if n else 0.0
        print(
            f"\nUNION {','.join(labels)}: n={n} A={na} ({100.0 * na / n:.1f}%) "
            f"B={nb} ({100.0 * nb / n:.1f}%) gap={gap:+.1f} pp"
        )
        p_exact = exact_p(only_a, only_b)
        p_chi = chi_sq(only_a, only_b)
        print(
            f"discordant A-only={only_a} B-only={only_b} | "
            f"exact McNemar p={p_exact:.4f} | chi2 p={p_chi:.4f}"
        )
        # Sample size still needed to confirm the observed gap.
        if n and only_a + only_b:
            delta = abs(na - nb) / n
            p_disc = (only_a + only_b) / n
            need = (
                1.96 * math.sqrt(p_disc)
                + 0.84 * math.sqrt(max(p_disc - delta**2, 1e-9))
            ) ** 2 / max(delta**2, 1e-9)
            print(
                f"paired instances needed for 80% power at this gap: ~{need:.0f} "
                f"(have {n})"
            )


if __name__ == "__main__":
    main()
