#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stamp an arm label into preds.json so evaluation output stays separable.

The evaluation harness builds both its log directory and its report filename from
each prediction record's `model_name_or_path`. Two endpoints that advertise the
same served model name therefore produce identical prefixes, and only `--run_id`
keeps them apart. Stamping a label here makes the log path and the report file
carry the arm identity as well.
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="path to preds.json")
    p.add_argument(
        "--label", required=True, help="arm label, e.g. armA/FlashNext-censored"
    )
    p.add_argument("--in-place", action="store_true")
    p.add_argument("--out", default="")
    a = p.parse_args()

    with open(a.preds) as f:
        data = json.load(f)
    for value in data.values():
        value["model_name_or_path"] = a.label
    dest = a.preds if a.in_place else (a.out or a.preds + ".tagged")
    with open(dest, "w") as f:
        json.dump(data, f, indent=1)
    print(f"stamped {len(data)} records with model_name_or_path={a.label} -> {dest}")
    print(
        "evaluation will write logs under .../<run_id>/"
        + a.label.replace("/", "__")
        + "/ and a report named "
        + a.label.replace("/", "__")
        + ".<run_id>.json"
    )


if __name__ == "__main__":
    main()
