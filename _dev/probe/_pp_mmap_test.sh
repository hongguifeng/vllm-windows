#!/usr/bin/env bash
# Reproduce the mmap failure and dump the memory policy that causes it.
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
MODEL=/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound
cat > /tmp/mmaptest.py <<'PY'
import mmap, os, resource
p = '/models/Qwen3.8-Flash-Next-W4A16-AutoRound/model-00016-of-00017.safetensors'
st = os.stat(p)
print('file size GiB: %.2f' % (st.st_size / 2**30))
fd = os.open(p, os.O_RDONLY)
try:
    m = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    print('ACCESS_READ ok:', len(m))
    m.close()
except Exception as e:
    print('ACCESS_READ failed:', type(e).__name__, e)
print('RLIMIT_AS:', resource.getrlimit(resource.RLIMIT_AS))
print('overcommit_memory:', open('/proc/sys/vm/overcommit_memory').read().strip())
for line in open('/proc/meminfo'):
    if line.startswith(('MemTotal', 'CommitLimit', 'Committed_AS', 'SwapTotal')):
        print(line.strip())
for p2 in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.swap.max'):
    try:
        print(p2, open(p2).read().strip())
    except Exception as e:
        print(p2, 'n/a')
PY
docker run --rm \
  -v "$MODEL:/models/Qwen3.8-Flash-Next-W4A16-AutoRound:ro" \
  -v /tmp/mmaptest.py:/tmp/mmaptest.py:ro \
  --entrypoint python3 "$IMG" /tmp/mmaptest.py 2>&1 | tail -20
