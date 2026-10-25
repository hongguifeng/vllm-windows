#!/bin/bash
set -u
for id in \
  instance_gravitational__teleport-1316e6728a3ee2fc124e2ea0cc6a02044c87a144-v626ec2a48416b10a88641359a169d99e935ff037 \
  instance_qutebrowser__qutebrowser-01d1d1494411380d97cac14614a829d3a69cecaf-v2ef375ac784985212b1805e1d0431dc8f1b3c171 \
  instance_qutebrowser__qutebrowser-0833b5f6f140d04200ec91605f88704dd18e2970-v059c6fdc75567943479b23ebca7c07b5e9a7f34c
do
  echo "pulling $id"
  docker pull "ghcr.io/scaleapi/swe-bench_pro-v2:$id" 2>&1 | tail -1
done
