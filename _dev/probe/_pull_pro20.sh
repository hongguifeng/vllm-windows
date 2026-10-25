#!/bin/bash
# Pull every SWE-Bench Pro V2 image needed by datasets/pro20, one at a time.
set -u
cd /home/hong/SWE-bench_Pro-os || exit 1
while read -r id; do
  [ -z "$id" ] && continue
  echo "=== $(date +%H:%M:%S) pulling $id"
  docker pull "ghcr.io/scaleapi/swe-bench_pro-v2:$id" >/dev/null 2>&1
  echo "=== $(date +%H:%M:%S) done $id rc=$?"
done < datasets/pro20/ids.txt
echo "ALL_DONE $(date +%H:%M:%S)"
