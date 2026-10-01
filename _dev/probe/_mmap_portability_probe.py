"""Check whether vLLM's SharedOffloadRegion (v1/kv_offload/cpu/shared_offload_region.py)
can run on Windows unchanged.  It is the piece behind `--kv-offloading-size`, which the
WSL container sets to 16 GiB and Windows does not.

Three Linux-only assumptions are exercised here, in isolation, without booting vLLM:
  1. mmap.mmap.madvise()        -- Unix-only in CPython
  2. os.unlink() on a mapped file -- Windows refuses while a mapping is open
  3. torch.frombuffer(memoryview(mmap)) -- zero-copy view over an mmap buffer

Run:  .venv\\Scripts\\python.exe _dev\\probe\\_mmap_portability_probe.py
"""

import mmap
import os
import tempfile

print(f"platform      : {os.name} / mmap.PAGESIZE={mmap.PAGESIZE}")

path = os.path.join(tempfile.gettempdir(), "vllm_offload_probe.mmap")
for stale in (path,):
    try:
        os.unlink(stale)
    except OSError:
        pass

# --- 1. open + ftruncate + mmap (the creator path, :108-145) -------------------
print(f"hasattr MAP_SHARED  : {hasattr(mmap, 'MAP_SHARED')}"
      f"   PROT_READ={hasattr(mmap, 'PROT_READ')}  PROT_WRITE={hasattr(mmap, 'PROT_WRITE')}")
fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
os.ftruncate(fd, 1 << 20)
try:
    # exactly what shared_offload_region.py:140-145 does
    m = mmap.mmap(fd, 1 << 20, flags=mmap.MAP_SHARED,
                  prot=mmap.PROT_READ | mmap.PROT_WRITE)
    print("verbatim mmap()     : OK")
except AttributeError as e:
    print(f"verbatim mmap()     : AttributeError: {e}  -> does NOT start")
    # Python's Windows mmap takes access= instead of flags=/prot=
    m = mmap.mmap(fd, 1 << 20, access=mmap.ACCESS_WRITE)
    print("Windows mmap(access=): OK  -> portable with a branch")

# --- 2. madvise (the populate path, :37-63) -----------------------------------
# The code catches only OSError/EINVAL and falls back; anything else propagates.
print(f"hasattr(madvise)    : {hasattr(m, 'madvise')}")
try:
    m.madvise(getattr(mmap, "MADV_POPULATE_WRITE", 23), 0, mmap.PAGESIZE)
    print("madvise             : OK")
except OSError as e:
    print(f"madvise             : OSError errno={e.errno} -> falls back (handled)")
except Exception as e:
    print(f"madvise             : {type(e).__name__}: {e}  -> NOT handled, raises")

# --- 3. zero-copy torch view over the mmap (the _base, :208) -------------------
try:
    import torch
    base = torch.frombuffer(memoryview(m), dtype=torch.int8)
    print(f"torch.frombuffer    : OK  shape={tuple(base.shape)}")
except Exception as e:
    print(f"torch.frombuffer    : {type(e).__name__}: {e}")

# --- 4. unlink semantics (:178 the kernel-reclaims-on-exit trick) -------------
try:
    os.unlink(path)
    print("unlink while mapped : OK")
except OSError as e:
    print(f"unlink while mapped : {type(e).__name__} winerror={getattr(e, 'winerror', None)}"
          " -> file would leak")

# Windows requires every exported buffer pointer to be gone before close();
# vLLM's cleanup() does drop _views/_base first (:326-328), so this order is the
# load-bearing part of the port, not the unlink.
base = None
import gc
gc.collect()
m.close()
os.close(fd)
try:
    os.unlink(path)
    print("unlink after close  : OK")
except OSError as e:
    print(f"unlink after close  : {type(e).__name__}: {e}")
print("done")
