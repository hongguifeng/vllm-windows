#!/bin/bash
cd /home/hong/SWE-bench_Pro-os || exit 1
missing=0
while IFS= read -r id; do
  [ -z "$id" ] && continue
  if ! docker image inspect "ghcr.io/scaleapi/swe-bench_pro-v2:$id" >/dev/null 2>&1; then
    echo "MISSING $id"
    missing=$((missing + 1))
  fi
done < datasets/pro20/ids.txt
echo "missing=$missing"
