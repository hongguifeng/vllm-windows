"""Deploy the sm_80 thin-M BF16 GEMM into the venv the service imports.

The service runs from `.flashnext-root\\.venv` (a junction to `.venv-flashnext`),
not from this checkout, so editing the repo tree changes nothing there
(`_dev/bin/_sync_snapshot_overlay.ps1` exists for the same reason). This deploys
only what thin GEMM needs, additively, and backs every touched file up first:

  package         vllm/sm80/{__init__,envs}.py, vllm/sm80/kernels/__init__.py,
                  vllm/sm80/kernels/thin_gemm/{__init__,thin_gemm_v74,warmup}.py
  inserts         utils.py (dispatch hook), kernel_warmup.py (pre-capture
                  warmup), envs.py (register the switches), shared_experts.py
                  (workspace lane for the aux stream)

Idempotent: a file that already carries its marker is left alone. `-Restore`
puts the backups back.

  .venv\\Scripts\\python.exe _dev\\probe\\_deploy_thin_gemm_overlay.py --dry-run
  .venv\\Scripts\\python.exe _dev\\probe\\_deploy_thin_gemm_overlay.py
  .venv\\Scripts\\python.exe _dev\\probe\\_deploy_thin_gemm_overlay.py --restore
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import shutil
from pathlib import Path

REPO = Path(r"D:\code\vllm-windows")
LIVE = REPO / ".venv-flashnext" / "Lib" / "site-packages" / "vllm"

COPY = [
    "sm80/__init__.py",
    "sm80/envs.py",
    "sm80/kernels/__init__.py",
    "sm80/kernels/thin_gemm/__init__.py",
    "sm80/kernels/thin_gemm/thin_gemm_v74.py",
    "sm80/kernels/thin_gemm/warmup.py",
    "third_party/flash_linear_attention/ops/pinned_autotune.py",
]

DISPATCH_OLD = "    backend_spec = _FLASHINFER_BF16_BACKENDS.get(linear_backend)\n"
DISPATCH_NEW = '''    # sm_80 thin-M BF16 GEMM (vllm/sm80/kernels/thin_gemm/), for the layers a
    # 2-bit checkpoint leaves unquantized. The env check comes first so that
    # with the flag unset this costs one bool read and nothing in that package
    # -- triton included -- is imported.
    if envs.VLLM_THIN_GEMM:
        from vllm.sm80.kernels.thin_gemm import (
            ampere_thin_gemm,
            use_ampere_thin_gemm,
        )

        if use_ampere_thin_gemm():
            return ampere_thin_gemm

''' + DISPATCH_OLD

WARMUP_OLD = "    minimax_m3_msa_warmup(worker)\n"
WARMUP_NEW = '''    # The sm_80 thin-M BF16 GEMM (vllm/sm80/kernels/thin_gemm/). MUST run before
    # capture_model(): thin_gemm keeps its FP32 split-K partials and arrival
    # counters in an allocate-once module-level cache, and a first launch cannot
    # happen inside a capture at all.
    if envs.VLLM_THIN_GEMM:
        from vllm.sm80.kernels.thin_gemm.warmup import warmup_ampere_thin_gemm

        warmup_ampere_thin_gemm(worker, cudagraph_capture_sizes)

''' + WARMUP_OLD

ENVS_TAIL = '''

# vllm/sm80: the sm_80 thin-M BF16 GEMM's switches (VLLM_THIN_GEMM,
# VLLM_THIN_GEMM_MAX_TOKENS), declared last so a name upstream adds wins.
from vllm.sm80.envs import register_env_vars as _register_sm80_env_vars

_register_sm80_env_vars(environment_variables)
'''

LANE_HELPER = '''def _thin_gemm_lane(lane: int):
    """Workspace lane for the thin GEMMs the aux stream enqueues.

    The sm_80 thin-M BF16 GEMM keeps its split-K partials per lane; the shared
    expert runs on a second stream while the main stream is busy, so it needs a
    lane of its own. A no-op when ``VLLM_THIN_GEMM`` is off.
    """
    if not envs.VLLM_THIN_GEMM:
        return nullcontext()
    from vllm.sm80.kernels.thin_gemm import use_lane

    return use_lane(lane)


logger = init_logger(__name__)
'''

SE_OLD = (
    "        with torch.cuda.stream(self._stream):\n"
    "            self._input_ready_event[idx].wait(self._stream)\n"
    "            self._output[idx] = self._layer(shared_experts_input)\n"
    "            self._output_ready_event[idx].record(self._stream)\n"
)
SE_NEW = SE_OLD.replace(
    "with torch.cuda.stream(self._stream):",
    "with torch.cuda.stream(self._stream), _thin_gemm_lane(1):",
)

KV_PLAN_OLD_0 = (
    "    num_blocks_per_request = 0\n"
    "    host_blocks_per_request = 0\n"
    "    scratch_blocks_per_request = 0\n"
    "    host_scratch_blocks_per_request = 0\n"
    "    for group in kv_cache_config.kv_cache_groups:\n"
)
KV_PLAN_NEW_0 = (
    "    num_blocks_per_request = 0\n"
    "    host_blocks_per_request = 0\n"
    "    scratch_blocks_per_request = 0\n"
    "    host_scratch_blocks_per_request = 0\n"
    "    plan: list[str] = []\n"
    "    for group in kv_cache_config.kv_cache_groups:\n"
)

KV_PLAN_OLD_1 = (
    "        scratch = min(scratch, required)\n"
    "        if group.host_resident:\n"
)
KV_PLAN_NEW_1 = (
    "        scratch = min(scratch, required)\n"
    "        if envs.VLLM_KV_PLAN_LOG:\n"
    "            per_layer = spec\n"
    "            if isinstance(per_layer, UniformTypeKVCacheSpecs):\n"
    "                per_layer = per_layer.first_spec\n"
    "            plan.append(\n"
    '                f"[{len(plan)}] {type(per_layer).__name__} "\n'
    "                f\"layers={len(group.layer_names)} \"\n"
    '                f"block={spec.block_size} page={spec.page_size_bytes} "\n'
    '                f"charged={required} scratch={scratch}"\n'
    '                + (" eagle" if group.is_eagle_group else "")\n'
    '                + (" host" if group.host_resident else "")\n'
    "            )\n"
    "        if group.host_resident:\n"
)
KV_PLAN_OLD_2 = "    return min(limits)\n\n\ndef _log_mamba_inflight_states("
KV_PLAN_NEW_2 = (
    "    concurrency = min(limits)\n"
    "    if envs.VLLM_KV_PLAN_LOG:\n"
    "        logger.info_once(\n"
    '            "KV cache plan (VLLM_KV_PLAN_LOG): num_blocks=%d, %d blocks per "\n'
    '            "request of which %d scratch, max_num_seqs=%d, concurrency %.4f; %s",\n'
    "            kv_cache_config.num_blocks,\n"
    "            num_blocks_per_request,\n"
    "            scratch_blocks_per_request,\n"
    "            max_num_seqs,\n"
    "            concurrency,\n"
    '            "; ".join(plan),\n'
    "        )\n"
    "    return concurrency\n"
    "\n\n"
    "def _log_mamba_inflight_states("
)

# (relative path, old text, new text, marker that means "already applied")
PATCHES = [
    ("model_executor/layers/utils.py", DISPATCH_OLD, DISPATCH_NEW, "VLLM_THIN_GEMM"),
    (
        "model_executor/warmup/kernel_warmup.py",
        WARMUP_OLD,
        WARMUP_NEW,
        "warmup_ampere_thin_gemm",
    ),
    (
        "v1/core/kv_cache_utils.py",
        KV_PLAN_OLD_0,
        KV_PLAN_NEW_0,
        "plan: list[str] = []",
    ),
    (
        "v1/core/kv_cache_utils.py",
        KV_PLAN_OLD_1,
        KV_PLAN_NEW_1,
        "VLLM_KV_PLAN_LOG",
    ),
    (
        "v1/core/kv_cache_utils.py",
        KV_PLAN_OLD_2,
        KV_PLAN_NEW_2,
        "KV cache plan (VLLM_KV_PLAN_LOG)",
    ),
    (
        "model_executor/layers/fused_moe/runner/shared_experts.py",
        "from enum import IntEnum\n",
        "from contextlib import nullcontext\nfrom enum import IntEnum\n",
        "from contextlib import nullcontext",
    ),
    (
        "model_executor/layers/fused_moe/runner/shared_experts.py",
        "logger = init_logger(__name__)\n",
        LANE_HELPER,
        "_thin_gemm_lane",
    ),
    (
        "model_executor/layers/fused_moe/runner/shared_experts.py",
        SE_OLD,
        SE_NEW,
        "_thin_gemm_lane(1)",
    ),
]

APPEND = ("envs.py", ENVS_TAIL, "_register_sm80_env_vars")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--backup-root", default=str(REPO / "_dev" / "out" / "overlay_backup"))
    args = ap.parse_args()

    if not LIVE.is_dir():
        print(f"ERROR: no deployed tree at {LIVE}")
        return 1
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = Path(args.backup_root) / f"thin_gemm_{stamp}"

    def save(path: Path, rel: str) -> None:
        if args.dry_run:
            return
        b = backup / rel
        b.parent.mkdir(parents=True, exist_ok=True)
        if not b.exists():
            shutil.copy2(path, b)

    if args.restore:
        backups = sorted(Path(args.backup_root).glob("thin_gemm_*"))
        if not backups:
            print("ERROR: no backup to restore")
            return 1
        src = backups[-1]
        restored = 0
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(src)
            if args.dry_run:
                print(f"would restore {rel}")
                continue
            (LIVE / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, LIVE / rel)
            restored += 1
        print(f"restored {restored} files from {src}")
        return 0

    for rel in COPY:
        src, dst = REPO / "vllm" / rel, LIVE / rel
        if not src.is_file():
            print(f"ERROR: missing source {src}")
            return 1
        same = dst.is_file() and digest(dst) == digest(src)
        action = "same" if same else ("update" if dst.exists() else "create")
        print(f"{action:<6} {rel}")
        if args.dry_run or same:
            continue
        if dst.is_file():
            save(dst, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for rel, old, new, marker in PATCHES:
        path = LIVE / rel
        text = path.read_text(encoding="utf-8")
        if marker in text:
            print(f"skip   {rel} ({marker})")
            continue
        if text.count(old) != 1:
            print(f"ERROR: anchor appears {text.count(old)}x in {rel}: {old!r}")
            return 1
        print(f"patch  {rel}")
        save(path, rel)
        if not args.dry_run:
            path.write_text(text.replace(old, new, 1), encoding="utf-8")

    rel, tail, marker = APPEND
    path = LIVE / rel
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"skip   {rel} ({marker})")
    else:
        print(f"append {rel}")
        save(path, rel)
        if not args.dry_run:
            path.write_text(text + tail, encoding="utf-8")

    print(f"\nbackup: {backup}")
    print("(dry run: nothing written)" if args.dry_run else "deployed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
