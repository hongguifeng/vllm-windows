#!/usr/bin/env bash
# Probe the host-tune.sh steps under WSL for the AlexBrandes PP recipe.
echo "uid=$(id -u)"
echo '--- power limit ---'
nvidia-smi -i 1 -pl 170 2>&1 | tail -3
echo '--- limits ---'
nvidia-smi --query-gpu=index,power.limit,power.default_limit,power.min_limit,power.max_limit --format=csv,noheader
echo '--- proc knobs ---'
[ -w /proc/sys/vm/drop_caches ] && echo 'drop_caches: writable' || echo 'drop_caches: RO'
[ -w /proc/sys/vm/compact_memory ] && echo 'compact_memory: writable' || echo 'compact_memory: RO'
echo '--- disk ---'
src=$(df --output=source /home | tail -1)
echo "model fs src: $src"
echo "pkname: $(lsblk -no pkname "$src" 2>/dev/null | head -1)"
if blockdev --setra 32 "$src" 2>/dev/null; then echo "setra: $(blockdev --getra "$src")"; else echo 'setra: failed'; fi
echo '--- mem ---'
free -g | head -2
nproc
