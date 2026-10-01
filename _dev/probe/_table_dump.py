#!/usr/bin/env python3
"""Dump the request table to JSON so a rewrite can be proved content-preserving."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _battle_table import table  # noqa: E402

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump([list(r) for r in table()], fh, ensure_ascii=False, indent=0)
print(f"wrote {sys.argv[1]}: {len(table())} requests")
