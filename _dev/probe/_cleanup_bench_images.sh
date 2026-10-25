#!/bin/bash
# Reclaim disk: drop Pier per-trial images, dangling images and build cache.
# Keeps benchmark base images and any image used by a running container.
set -u

echo "== before =="
df -h / | tail -1
docker system df | head -4

echo
echo "== running container images (kept) =="
docker ps --format '{{.Image}}' | sort -u

mapfile -t candidates < <(
  docker image ls --format '{{.Repository}}:{{.Tag}}' \
    | grep -E '__(main|pier-egress-proxy):latest$' | sort -u
)
mapfile -t in_use < <(docker ps --format '{{.Image}}' | sort -u)

echo
echo "== pier trial candidates: ${#candidates[@]} =="
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
  docker image rm -f "$image" >/dev/null 2>&1 && removed=$((removed + 1))
done
echo "removed pier trial images: $removed"

echo
echo "== dangling images =="
docker image prune -f | tail -2

echo
echo "== build cache =="
docker builder prune -f | tail -2

echo
echo "== after =="
df -h / | tail -1
docker system df | head -4
