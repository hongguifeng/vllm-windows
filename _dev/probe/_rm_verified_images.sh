#!/bin/bash
# Remove the SWE-bench Verified eval images (ghcr.io/epoch-research/swe-bench.eval.*).
set -u
mapfile -t images < <(
  docker images --format '{{.Repository}}:{{.Tag}}' | grep '^ghcr.io/epoch-research/swe-bench.eval' | sort -u
)
echo "verified images found: ${#images[@]}"
if [ "${#images[@]}" -gt 0 ]; then
  docker image rm -f "${images[@]}" >/dev/null 2>&1
fi
echo "remaining: $(docker images --format '{{.Repository}}' | grep -c 'epoch-research')"
df -h / | tail -1
docker system df | head -3
