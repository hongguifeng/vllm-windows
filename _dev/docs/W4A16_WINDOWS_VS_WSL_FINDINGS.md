# Qwen3.8-Flash-Next W4A16 on two CMP 170HX — what works, what does not

Consolidated findings, 2026-10-24. Covers both routes tried for the 4-bit
AutoRound checkpoint: the Linux container under WSL2 (works, 82 tok/s) and the
native Windows fork (loads, then fails inside the MoE path under PP).

Related: [PP_RECIPE_WSL_STATUS.md](PP_RECIPE_WSL_STATUS.md) — the WSL engine's
config, patches and benchmark sweep.

---

## 1. TL;DR

| | WSL2 + AlexBrandes recipe (`~/pp-recipe`) | Windows fork (`D:\code\vllm-windows`) |
|---|---|---|
| Status | **runs**, PP=2, k=3 MTP, 262k ctx | **loads weights**, dies at MoE init under PP=2 |
| Decode c1 (prose) | **82 tok/s** (median 77.9 over 5 runs) | — |
| c2 / c4 | 128.3 / 202.9 tok/s | — |
| Cold prefill | 3,616 tok/s (13.8k prompt) | — |
| KV pool | 1,245,920 tokens @ 262k | 160,861 tokens @ 2k (load test) |
| Blocker | none (PLE table read from NVMe, ~29 % of step time) | group-wise WNA16 MoE has no PP-safe backend |

The Windows route is blocked by quantisation-format × backend coverage, not by
pipeline parallelism itself — see §5.

---

## 2. Checkpoint facts (both routes use the same files)

- `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`: **168.8 GiB, 31 files**, pinned at
  revision `4c67bf68…`. All files sha256-verified against the ModelScope
  manifest; the 19 LFS blobs match the pinned HF revision bit for bit.
- Composition: 16 weight shards (**73.4 GiB**, `group_size: 128`, bits 4,
  `quant_method: auto-round`, `packing_format: auto_round:auto_gptq`) plus one
  **95.37 GiB** shard holding the 128 PLE n-gram tensors (BF16, 320,001,536 rows
  × 160 dims).
- 222,720 expert tensors, **zero expert bias** — the fork's
  `may_have_bias=True` default is a false positive for this checkpoint (§5.5).
- `text_config.ple_layer_ids = [2]`, 48 layers, `model_type: qwen4_exp`.

### 2.1 The 95.37 GiB PLE shard can be shared with the existing 3bpw model

The 3bpw checkout already has the same table as `model-00001-of-00011.safetensors`
(102,400,512,288 B). Ours is 102,400,512,256 B — 32 bytes shorter because the
tensor map is serialised in numeric order (`shard_0, shard_1, …`) instead of
lexicographic (`shard_0, shard_1, shard_10, …`). Same 128 tensors, same dtype
and shape, same total payload.

Verified by hashing three tensors by name from both files: `shard_0`,
`shard_64`, `shard_127` all **identical**. Because safetensors is
self-describing (tensor lookup by name, offsets read from the header), an NTFS
**hardlink** is enough — no 95 GiB copy, no double disk usage. That is what the
Windows staging folder used (`model-00016-of-00017.safetensors` → the 3bpw
file), and the PLE SSD layer read it happily:

```
PLE SSD: 95.37 GiB on disk, 95.37 GiB in rows [0, 320001536) of 320001536,
         16 I/O workers, 8192 MiB row cache
PLE SSD uses native Windows overlapped reads, queue depth 512 (asked 512)
```

---

## 3. WSL2 route — the one that works

Full details in [PP_RECIPE_WSL_STATUS.md](PP_RECIPE_WSL_STATUS.md). Summary:

- Recipe at `~/pp-recipe`; engine image `ghcr.io/alexbrandes/…@sha256:d6fc872d…`;
  checkpoint in WSL ext4; container `hong-pc` on `127.0.0.1:8000`.
- Four WSL adaptations were required (device selection, overcommit, pinned-memory
  opt-in, `HOST_TUNE=0`).
- MTP depth sweep (median of 5 passes): k=3 beats k=5 on every level —
  c1 77.9 vs 72.0, c2 128.3 vs 114.0, c4 202.9 vs 200.3, prefill 3,616 vs 3,484 —
  consistent with fewer draft positions costing fewer NVMe lookups.
- Per-step budget at c1: 30 ms/step, 2.45 tokens/step. Disk reads during decode:
  1,854 IOPS at 4 KiB ≈ 22 % page-cache hit rate against a 95.37 GiB table with
  37 GB of cache. A queue-depth sweep on the same drive shows **31,861 IOPS at
  qd=8**, i.e. the engine uses ~6 % of what the SSD can deliver — the cost is
  latency, not device throughput.
- Gap to the reference host decomposes as ≈1.4× per-step time (I/O wait) × 1.35×
  tokens/step (acceptance, prompt-dependent), which multiplies to the observed
  ~1.9×.

---

## 4. Windows fork — what was proven to work

Staging on NTFS: 15 weight shards copied (73.4 GiB, ~145 MB/s through 9p) plus
14 small files; PLE shard hardlinked (§2.1). Total new disk: 73.4 GiB.

The engine then did, repeatedly:

```
quantization=inc                     # auto-detected, no code changes
world_size=2 rank=0/1                # PP=2, VLLM_PP_LAYER_PARTITION=26,22
Model loading took 38.86 GiB (PP0) / 33.35 GiB (PP1)
GPU KV cache size: 160,861 tokens    # max-model-len 2048
kernel warmups: QSA, mamba, M-RoPE, vision
```

So: quantisation format, PP layer split, dual-card placement, the PLE SSD DLL
with a hardlinked table, and the mamba/QSA warmups all work. What fails is the
MoE path (§5).

Hard constraint discovered on the way: the model needs **two cards** — 73.4 GiB
of weights does not fit one 64 GiB CMP 170HX, so PP=2 (or TP=2) is mandatory.
TP=2 is not viable on this fork because **NCCL is unavailable on Windows**
(`Failed to load NCCL library from libnccl.so.2`), leaving gloo for collectives.

---

## 5. Six blockers found, in the order they appeared

### 5.1 PP + PLE gate (config level) — solved

`vllm/model_executor/models/config.py` rejects PLE models with PP>1 unless
`VLLM_PLE_ALLOW_PP` is set. The working 3bpw PP runs had that variable in their
environment (their logs contain `Unknown vLLM environment variable detected:
VLLM_PLE_ALLOW_PP`), which is why PP worked for them. Not a real limitation for
this model: the PLE layer (id 2) lives on PP rank 0.

### 5.2 PP + PLE gate (runtime level) — solved once the stale copy was noticed

`vllm/models/qwen4_exp/nvidia/model_state.py` raises
`N-gram PLE embedding currently requires pipeline_parallel_size=1 …` again at
model-state construction. **The installed copy in
`.flashnext-root/.venv/Lib/site-packages` did not have the `VLLM_PLE_ALLOW_PP`
escape hatch that the source tree has** — a stale install. A one-line patch
(mirroring the tree) gets past it. Patch applied for testing, then reverted;
backup at `model_state.py.bak`.

### 5.3 Two API servers, one handshake port — solved

Symptom: `zmq.error.ZMQError: Address in use (tcp://127.0.0.1:29550)`, and the
port follows whatever `--data-parallel-rpc-port` is passed (29550 → 29994), so
it is not a leftover-process problem.

Cause: vLLM sets the multiprocessing start method to **forkserver**, and the
launcher wrapper was re-imported by children, each starting a full API server;
both bind the same handshake port. Fix: put the launch behind
`if __name__ == "__main__":` and pass an explicit `--data-parallel-rpc-port`.

### 5.4 Expert-bias false positive — solved

With `--moe-backend triton`:
`ValueError: WNA16 MoE backend 'TRITON' does not support the deployment
configuration since expert bias is not supported.` But this checkpoint has
**no expert bias at all** (0 of 222,720 expert tensors). The flag is hardcoded
`may_have_bias=True` at `auto_gptq.py:479`; flipping it lets the Triton backend
through. Patched for testing, then reverted; backup at `auto_gptq.py.bak`.

### 5.5 Triton WNA16 MoE does not support group-wise scales — **open**

With the bias flag flipped, the Triton backend is selected
(`Using 'TRITON' WNA16 MoE backend`) and PP initialisation proceeds to the first
(dummy) forward, then:

```
invoke_fused_moe_wna16_triton_kernel(...)
  assert block_shape is not None and block_shape[0] == 0      # fused_moe.py:630
AssertionError
```

`block_shape[0] == 0` means "per-channel only"; this checkpoint is
`group_size: 128`, so `block_shape[0] = 128`. **The Triton WNA16 MoE has no
group-wise path**, independent of PP.

### 5.6 Marlin WNA16 MoE under PP — **open, and silent**

`--moe-backend auto` selects Marlin (`Using MarlinExperts`). Then, with real
weights:

- attempt 1: weights loaded (152 s), KV cache and warmups completed, then the
  API server hung with frozen CPU (no listener);
- attempts 2–3: PP0 disappeared right after `Using MarlinExperts` — **no Python
  traceback, no Windows Application Error record, no faulthandler dump**;
- attempt 4: both ranks stalled with `read +0 MB/s, cpu +0 s`.

Three different failure shapes with no stack is the signature of something
native (kernel/allocator), not a Python-level guard.

### 5.7 Load-time host memory (orthogonal, measured)

During weight load the commit charge jumps from 34 GB to **121 GB** on an 88 GB
machine (both PP ranks load their shards concurrently; pagefile absorbs it).
This is enough to explain a silently killed worker on its own, and it will have
to be addressed (serialise PP loading, or enlarge the pagefile) even after the
MoE question is solved.

---

## 6. Why this checkpoint and not the 3bpw one

| | 3bpw (`Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`) | W4A16 (this one) |
|---|---|---|
| Weights on GPU | 47 GiB (23.7 per rank) | 73.4 GiB (38.9 / 33.4) |
| MoE backend | **humming** low-bit path (`MoEPrepareAndFinalizeNoDPEPModular`, needs `VLLM_PROBE_INC_HUMMING_LOWBIT`) | `int_wna16` → **Marlin** or **Triton** |
| PP=2 | works (KV 963,925 @ 8k) | fails (§5.5, §5.6) |

The 3bpw checkpoint's MoE runs on a bespoke low-bit kernel that has PP support;
the 4-bit AutoGPTQ checkpoint routes into vLLM's `int_wna16` oracle, where the
group-wise format has exactly one supporting backend (Marlin) and that backend
does not work under PP here, while the PP-tested backend (Triton) rejects
group-wise scales. The two models were never on the same code path.

---

## 7. Effort estimate to actually add support

| Route | Work | Estimate |
|---|---|---|
| A. Group-wise scales in the Triton WNA16 MoE | kernel scale indexing (one scale per 128 rows), `invoke_…` wrapper, `convert_to_wna16_moe_kernel_format` layout | **2–5 days**, mostly numerical validation |
| B. Debug Marlin under PP | no stack, no dump → needs a minimal repro and a native debugger; may be a workspace/EP-group sizing assumption, may be deeper | **1 week+, unbounded** |
| C. Re-quantise to per-channel | avoids A and B; group 128 → per-channel costs quality; 169 GB regenerated | **1 day machine time + quality eval** |
| D. Load-time memory | serialise PP rank loading or enlarge the pagefile | **half a day** |

Shortest workable path is **A + D ≈ 3–6 days**, with a real risk of
"runs but numerics drift" — which is why A would need logit-level comparison
against the WSL engine, not just a green health check.

Recommended instead: ask the quantisation side for a per-channel or
humming-compatible 4-bit build. Changing the quantisation is safer than
changing a MoE kernel.

---

## 8. Reproducing / restoring

### WSL engine (the working one)

```bash
# once per WSL boot - the PLE shard needs overcommit, pinned memory needs the opt-in
docker run --rm --privileged --entrypoint bash \
  ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d… \
  -c 'echo 1 > /proc/sys/vm/overcommit_memory'

cd ~/pp-recipe && ./start.sh          # ~7 min to /health, port 127.0.0.1:8000
```

`start.sh` in that checkout already carries: `CUDA_VISIBLE_DEVICES` instead of
`--gpus device=`, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, `VLLM_PLE_HOST_IDS=1`,
MTP k=3, and `.env` sets `HOST_TUNE=0`.

### Windows 3bpw services (the fast ones)

Exact command lines are saved in `D:\code\_scratch\windows_vllm_restore.txt`
(both were stopped on 2026-10-24 08:01; ports 9393 and 9402).

### Windows W4A16 experiment

- Launcher: `_dev/probe/_win_w4a16_launch.py` (PP=2, 26/22, eager, PLE SSD via
  `ple_ssd_io_win.dll`, `--data-parallel-rpc-port 29994`; `--with-spec` and
  `--with-graphs` flags exist; it currently also carries `--moe-backend triton`
  and `--load-format dummy` from the last experiments).
- venv patches were **reverted**; backups remain at
  `auto_gptq.py.bak` and `model_state.py.bak` under `.flashnext-root/.venv/…`.
- The staged checkpoint on `D:\models\Qwen3.8-Flash-Next-W4A16-AutoRound` was
  **deleted** on 2026-10-24 (freeing 73.4 GiB; the 95.37 GiB table was a
  hardlink and was never duplicated). The WSL copy still exists.

---

## 9. Machine state after this investigation

- GPU 0 (3090) idle; GPU 1/2 (CMP 170HX) free, no python processes left.
- WSL container `hong-pc` exited, image and checkpoint intact (`~/pp-recipe`).
- `D:\models` holds only 3bpw (142.6 GiB) and the DFlash draft models; D: free
  space 314.9 GB.
- NVMe health: one controller dropout on 2026-10-23 21:41 (18,353 `disk 51`
  events, one `stornvme` reset) during a heavy docker pull; the drive
  re-enumerated and has been clean since. `chkdsk D:` still pending.
