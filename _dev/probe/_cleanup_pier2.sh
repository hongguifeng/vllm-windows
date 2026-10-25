#!/bin/bash
# Second pass: remove Pier per-trial images (<task>__<suffix>-main / -pier-egress-proxy).
set -u

mapfile -t candidates < <(
  docker image ls --format '{{.Repository}}:{{.Tag}}' \
    | grep -E '__[A-Za-z0-9]+-(main|pier-egress-proxy):latest$' | sort -u
)
mapfile -t in_use < <(docker ps --format '{{.Image}}' | sort -u)

echo "candidates: ${#candidates[@]}"
removed=0
for image in "${candidates[@]}"; do
  skip=0
  for used in "${in_use[@]}"; do
    if [ "$image" = "$used" ]; then skip=1; fi
  done
  if [ "$skip" = "1" ]; then
    echo "skip (in use) $image"
    continue
  fi
  if docker image rm -f "$image" >/dev/null 2>&1; then
    removed=$((removed + 1))
  else
    echo "failed $image"
  fi
done
echo "removed: $removed"
echo "== after =="
df -h / | tail -1
docker system df | head -4
