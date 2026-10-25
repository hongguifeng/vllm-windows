#!/usr/bin/env bash
# Cross-check ModelScope sha256 against the HF LFS oid for the pinned revision.
set -uo pipefail
python3 - <<'PY'
import json, subprocess
hf = json.load(open('/tmp/pp_tree.json'))
hf_oid = {x['path']: x.get('lfs', {}).get('oid') for x in hf if x.get('type') == 'file'}
ms = json.load(open('/tmp/ms_files.json'))['Data']['Files']
ms_sha = {f['Path']: (f.get('Sha256') or '') for f in ms}
checked = matched = 0
mismatch = []
for p, oid in hf_oid.items():
    if not oid:
        continue
    s = ms_sha.get(p)
    if not s:
        continue
    checked += 1
    if s.lower() == oid.lower():
        matched += 1
    else:
        mismatch.append(p)
print(f"LFS files comparable: {checked}, sha256 identical: {matched}")
print("mismatched:", mismatch or "none")
PY
