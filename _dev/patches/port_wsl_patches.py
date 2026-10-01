"""Port the HyperQwen patch subset that pays off on a CMP 170HX into this tree.

Source: WSL ``~/testcode/qwen38-27b-rtx3090`` (upstream ``syv-ai/HyperQwen``),
41 patches cut against v0.28.0. Only six of them are worth carrying here, and
the reasoning is in ``WSL_3090_PATCH_PORT_REPORT.md``. Short version: that repo
solves "fit a 27B on a 24 GB 3090", this one solves "make vLLM work on
Windows/sm_80". The overlap is the Marlin-on-sm80 group plus three patches that
are hardware-independent engineering fixes.

Batch 1 (in ``patches/series`` order -- the order is load-bearing, see below):
hardware-independent or sm80-specific.

  marlin-int8-layer-select      per-layer include/exclude regexes for the W4A8
                                activation path
  marlin-int8-negative-scales   AutoRound exports negative group scales and the
                                int8 Marlin kernel reads them unsigned -> garbage
  marlin-repack-staged-sm80     one grow-only staging buffer for the sm80 repack;
                                defaults ON at compute capability 8.0 exactly
  engine-completion-log         one ungated INFO line per finished request
  engine-stall-sentinel         warn when the core stops stepping
  no-output-token-reservation   stop rejecting long prompts with a large max_tokens

Batch 2 (added for the RTX 3090 profile; still in ``patches/series`` order).
These were written and tuned against a 3090 -- several say so in their own
headers -- and were only left out because the deploy target used to be a 64 GB
CMP 170HX. That hardware excuse no longer holds on a 24 GB sm_86 card:

  sampler-small-topk-fast-softmax  sort-free top-k/top-p + multi-block row
                                softmax for speculative-decode steps; the model's
                                vocab is 248 320, exactly what its header targets
  spec-decode-attn              split-KV Triton verify attention. The header
                                names the 3090's 82 SMs and MTP's k+1=5 queries;
                                it is on the FLASH_ATTN backend with a bf16 KV
                                cache, both of which we run
  prefill-attn-int8             int8-QK Triton prefill for head_dim 256. Its
                                gate is `num_heads==24 and num_kv_heads==4 and
                                head_size==256`, which this model matches
                                exactly, and its header measures FA2-hd256 at
                                54-57 TFLOPS on a 250 W 3090 -- our prefill is
                                the weakest number we have
  mamba-chunked-prefill-align   state loss and NaN during chunked prefill in
                                Mamba/GDN models; we run chunked prefill on a
                                hybrid GDN model
  vllm-pr50021-gdn-spec-bounds  GDN/KDA spec-decode out-of-bounds guard (an
                                upstream PR, unmerged); we run MTP on GDN layers

Batch 3. ``hybrid-kv-groups-v2-cudagraph`` is carried for its *first* hunk only.
That patch has two halves -- hybrid group sizing and an explicit CUDA-graph
memory reserve for the V2 runner -- and only the first one is still live at this
pin:

  hybrid-kv-groups-v2-cudagraph  _get_kv_cache_groups_uniform_page_size picks
                                group_size = the smallest bucket of same-type
                                layers, and with the DFlash2 drafter that bucket
                                is its 5 sliding-window layers: the target's 16
                                full-attention layers are then padded to 20 and
                                the 48 GDN layers to 50, and the padding is
                                charged per token of context like real KV. The
                                patch prefers padding the sliding-window bucket
                                instead (5 -> 8, window-sized blocks only).
                                0.29 logs exactly that tax ("Add 4 padding
                                layers, may waste at most 25.00%", "Add 2 padding
                                layers, ... 4.17%") and has no knob to avoid it.
                                The SECOND half is obsolete: 0.29 implements
                                ``profile_cudagraph_memory`` for real
                                (``v1/worker/gpu/model_runner.py`` delegates to
                                the imported ``_profile_cudagraph_memory``), so
                                both its ``envs.py`` and its ``model_runner.py``
                                sections are dropped and its
                                ``VLLM_V2_CUDAGRAPH_MEM_MIB`` knob is not
                                registered. Inert without a speculative drafter:
                                with MTP only, ``min_num_layers == 16`` and the
                                early return keeps the pool bit-identical.

Dependency note: in ``patches/series`` the order is
sampler -> spec-decode-attn -> prefill-attn-int8, and it is not cosmetic --
`spec-decode-attn`'s hunk context contains lines `sampler` inserts, and
`prefill-attn-int8`'s contains lines `spec-decode-attn` inserts. Applied out of
order they conflict; applied in order they are clean.

Still not carried, and correctly so on any card: DFlash2 (needs a per-model
drafter checkpoint) and the int4-KV chain (it hinges on hybrid-sw-block-promote,
whose promotion branch only fires when a layer's page fails to divide the
maximum -- impossible with a bf16 cache, where the target and the drafter both
land on 4096 B/token/layer). Also skipped: the WSL2-only device-pointer patch,
the sm89+ fp8 KV patch (sm_86 has no FP8 either), the vision-tower offload (0.29
upstreamed it as --cpu-offload-gb/--cpu-offload-params, so no patch is needed),
and the patches 0.29 already ships.

Usage::

    python port_wsl_patches.py            # apply to vllm/ and sync site-packages
    python port_wsl_patches.py --check    # dry run, report only
    python port_wsl_patches.py --repo-only

Applying order matters for one pair: ``marlin-repack-staged-sm80`` registers
its knob in ``envs.py`` inside a hunk whose context already contains the two
knobs ``marlin-int8-layer-select`` adds, so it only applies after it. Checking
it standalone always fails; that is expected, not a broken patch.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = os.path.join(REPO, "_wslpatch")
SITE_PACKAGES_VLLM = os.path.join(REPO, ".venv", "Lib", "site-packages", "vllm")

# (basename, why it is carried) -- ordered exactly as patches/series has them.
CARRIED = [
    (
        "marlin-int8-layer-select",
        "per-layer W4A8 selection (VLLM_MARLIN_INT8_INCLUDE_RE/_EXCLUDE_RE)",
    ),
    (
        "marlin-int8-negative-scales",
        "AutoRound negative group scales are read unsigned by the int8 Marlin kernel",
    ),
    (
        "marlin-repack-staged-sm80",
        "grow-only staging buffer for the sm80 repack; on by default at CC 8.0",
    ),
    ("engine-completion-log", "one INFO line per request that leaves the scheduler"),
    ("engine-stall-sentinel", "warn once per episode when the core stops stepping"),
    (
        "no-output-token-reservation",
        "long prompt + large max_tokens no longer rejected with a 400",
    ),
]

# Batch 2 -- see the module docstring. Order is load-bearing: each one's hunk
# context contains lines the previous one inserts.
CARRIED_B = [
    (
        "sampler-small-topk-fast-softmax",
        "sort-free top-k/top-p and multi-block row softmax for spec-decode steps",
    ),
    (
        "spec-decode-attn",
        "split-KV Triton verify attention; needs the FLASH_ATTN backend + bf16 KV",
    ),
    (
        "prefill-attn-int8",
        "int8-QK prefill attention for head_dim 256 (gate: 24 heads / 4 kv / 256)",
    ),
    (
        "mamba-chunked-prefill-align",
        "no state loss / NaN when chunked prefill crosses a Mamba block boundary",
    ),
    (
        "vllm-pr50021-gdn-spec-bounds",
        "out-of-bounds guard in the GDN/KDA spec-decode path",
    ),
]

# Batch 3 -- only needed once a speculative drafter adds its own sliding-window
# layers to the model. See the module docstring; its second half is obsolete at
# this pin, so this patch is carried through PARTIAL with two files dropped.
CARRIED_C = [
    (
        "hybrid-kv-groups-v2-cudagraph",
        "pad the drafter's 5 sliding-window layers instead of the target's 16 "
        "full + 48 GDN layers (0.29's 'Add 4 padding layers ... 25.00%')",
    ),
]

# Applied first if present, because the tree may not have it yet and several
# hunks above carry its lines as context. It is already in this tree.
PREREQ = "qwen3_5-embed-quant"

# Patches that cannot be applied whole. `marlin-repack-staged-sm80` registers
# its knob in envs.py inside a hunk whose context includes
# VLLM_V2_CUDAGRAPH_MEM_MIB -- a knob belonging to
# hybrid-kv-groups-v2-cudagraph, which is deliberately not carried here (0.29
# profiles V2 CUDA graphs itself, so that patch is obsolete at this pin). The
# envs.py hunks are therefore dropped from the git-apply pass and the two
# registrations are inserted by hand, anchored on the knob the previous patch
# left behind.
ENVS_ANNOTATION_ANCHOR = '    VLLM_MARLIN_INT8_EXCLUDE_RE: str = "lm_head|mtp"\n'
ENVS_ANNOTATION_INSERT = (
    "    # syv patch (marlin-repack-staged-sm80): \"1\"/\"0\" override of the"
    " staging buffer, unset = on for sm80 only\n"
    "    VLLM_MARLIN_REPACK_STAGED: str | None = None\n"
)

ENVS_REGISTRY_ANCHOR = (
    '    "VLLM_MARLIN_INT8_EXCLUDE_RE": lambda: os.environ.get(\n'
    '        "VLLM_MARLIN_INT8_EXCLUDE_RE", "lm_head|mtp"\n'
    "    ),\n"
)
ENVS_REGISTRY_INSERT = (
    "    # syv patch (marlin-repack-staged-sm80)\n"
    '    "VLLM_MARLIN_REPACK_STAGED": lambda: os.environ.get('
    '"VLLM_MARLIN_REPACK_STAGED"),\n'
)

# Batch 2's three patches register their knobs in the same spot -- right after
# VLLM_MARLIN_INT8_EXCLUDE_RE, which is also where batch 1's
# marlin-repack-staged-sm80 landed. Only their envs.py hunks are dropped (they
# are the whole reason a bare git-apply conflicts); the inserts go *after* the
# anchor line so the anchor stays findable for the next patch in the queue.
B2_ANNOTATION_ANCHOR = '    VLLM_MARLIN_INT8_EXCLUDE_RE: str = "lm_head|mtp"\n'
B2_REGISTRY_ANCHOR = (
    '    "VLLM_MARLIN_INT8_EXCLUDE_RE": lambda: os.environ.get(\n'
    '        "VLLM_MARLIN_INT8_EXCLUDE_RE", "lm_head|mtp"\n'
    "    ),\n"
)

B2_SAMPLER_ANNOTATION = (
    "    # syv patch (sampler-small-topk-fast-softmax): draft-side top-k/top-p"
    " truncation and temperature scale,\n"
    "    # read once at import; registered so they take part in the"
    " torch.compile cache key\n"
    "    VLLM_DRAFT_TOPK_TOPP: bool = True\n"
    "    VLLM_DRAFT_TEMP_SCALE: float = 1.0\n"
)
B2_SAMPLER_REGISTRY = (
    "    # syv patch (sampler-small-topk-fast-softmax)\n"
    '    "VLLM_DRAFT_TOPK_TOPP": lambda: os.environ.get('
    '"VLLM_DRAFT_TOPK_TOPP", "1") == "1",\n'
    '    "VLLM_DRAFT_TEMP_SCALE": lambda: float(os.environ.get('
    '"VLLM_DRAFT_TEMP_SCALE", "1.0")),\n'
)

B2_SPECDEC_ANNOTATION = (
    "    # syv patch (spec-decode-attn): split-KV verify kernel switch,"
    " query-token cap for its partial buffers\n"
    "    # (0 = 1 + num_speculative_tokens) and a forced query-row tile"
    " (0 = pick by row count); registered so\n"
    "    # they take part in the torch.compile cache key"
    " (both change the kernel launch)\n"
    "    VLLM_SPEC_DECODE_ATTN: bool = False\n"
    "    VLLM_SPEC_DECODE_ATTN_QMAX: int = 0\n"
    "    VLLM_SPEC_ATTN_BLOCK_M: int = 0\n"
)
B2_SPECDEC_REGISTRY = (
    "    # syv patch (spec-decode-attn)\n"
    '    "VLLM_SPEC_DECODE_ATTN": lambda: os.environ.get('
    '"VLLM_SPEC_DECODE_ATTN", "0") == "1",\n'
    '    "VLLM_SPEC_DECODE_ATTN_QMAX": lambda: int(os.environ.get('
    '"VLLM_SPEC_DECODE_ATTN_QMAX") or 0),\n'
    '    "VLLM_SPEC_ATTN_BLOCK_M": lambda: int(os.environ.get('
    '"VLLM_SPEC_ATTN_BLOCK_M") or 0),\n'
)

B2_PREFILL_ANNOTATION = (
    '    # syv patch (prefill-attn-int8): "int8" / "fp16" selects the'
    " Triton prefill kernel on the hybrid model\n"
    '    VLLM_PREFILL_ATTN: str = ""\n'
)
B2_PREFILL_REGISTRY = (
    "    # syv patch (prefill-attn-int8)\n"
    '    "VLLM_PREFILL_ATTN": lambda: os.environ.get("VLLM_PREFILL_ATTN", ""),\n'
)

PARTIAL = {
    "marlin-repack-staged-sm80": {
        "skip_files": ["envs.py"],
        "manual_edits": [
            ("envs.py", ENVS_ANNOTATION_ANCHOR, ENVS_ANNOTATION_INSERT),
            ("envs.py", ENVS_REGISTRY_ANCHOR, ENVS_REGISTRY_INSERT),
        ],
    },
    "sampler-small-topk-fast-softmax": {
        "skip_files": ["envs.py"],
        "manual_edits": [
            ("envs.py", B2_ANNOTATION_ANCHOR, B2_SAMPLER_ANNOTATION),
            ("envs.py", B2_REGISTRY_ANCHOR, B2_SAMPLER_REGISTRY),
        ],
    },
    "spec-decode-attn": {
        "skip_files": ["envs.py"],
        "manual_edits": [
            ("envs.py", B2_ANNOTATION_ANCHOR, B2_SPECDEC_ANNOTATION),
            ("envs.py", B2_REGISTRY_ANCHOR, B2_SPECDEC_REGISTRY),
        ],
    },
    "prefill-attn-int8": {
        "skip_files": ["envs.py"],
        "manual_edits": [
            ("envs.py", B2_ANNOTATION_ANCHOR, B2_PREFILL_ANNOTATION),
            ("envs.py", B2_REGISTRY_ANCHOR, B2_PREFILL_REGISTRY),
        ],
    },
    # Batch 3. Both dropped files belong to the patch's CUDA-graph half: 0.29
    # profiles CUDA graphs itself, so the knob would be dead config, and its
    # model_runner.py hunk anchors on 0.28's `return 0` body that 0.29 replaced
    # with a delegation call. What is left is one hunk of kv_cache_utils.py and
    # no manual insert at all.
    "hybrid-kv-groups-v2-cudagraph": {
        "skip_files": ["envs.py", "v1/worker/gpu/model_runner.py"],
        "manual_edits": [],
    },
}

# Re-scoped patches. 0.29 grew ten lines (`num_splits = attn_metadata.
# max_num_splits` plus the FA4 hd256 page alignment) between
# `causal = not has_window` and the `flash_attn_varlen_func(` call -- which is
# exactly the context both 0.28-era attention hunks quote, hence the conflict
# that no amount of `-C1` fixes. Both hunks are pure additions at this spot, so
# the added lines are lifted out of the patch verbatim (added_block) and
# re-anchored on the blank line that follows `causal = not has_window`.
FLASH_ATTN = "v1/attention/backends/flash_attn.py"
FLASH_ATTN_ANCHOR = "                    causal = not has_window\n\n"
REBASED = {
    name: {"file": FLASH_ATTN, "hunk": 0, "anchor": FLASH_ATTN_ANCHOR}
    for name in ("spec-decode-attn", "prefill-attn-int8")
}


def rebased_partial(name):
    """Entry for a patch whose ``flash_attn.py`` dispatch hunk needs re-anchoring.

    The envs.py knobs still come from PARTIAL. The conflicting hunk is dropped
    from the patch and its added lines are inserted at `anchor` instead; any
    other hunks for the same file stay in the patch and apply normally, so the
    returned ``patch`` replaces the skip-based split. spec-decode-attn has a
    second flash_attn.py hunk (its kernel helpers, further down the file) that
    takes this path unchanged.
    """
    spec = REBASED.get(name)
    if spec is None:
        return None

    patch_path = os.path.join(PATCH_DIR, f"{name}.patch")
    block = added_block(patch_path, spec["file"], spec["hunk"])
    if not block:
        raise RuntimeError(
            f"{name}: no added lines in hunk {spec['hunk']} of {spec['file']}"
        )

    stripped = split_sections(
        patch_path,
        PARTIAL[name]["skip_files"],
        out=os.path.join(PATCH_DIR, f"_{name}.stripped.patch"),
    )[0]
    rebased = drop_hunk(
        stripped,
        spec["file"],
        spec["hunk"],
        out=os.path.join(PATCH_DIR, f"_{name}.rebased.patch"),
    )

    edits = list(PARTIAL[name]["manual_edits"])
    edits.append((spec["file"], spec["anchor"], block))
    return {"patch": rebased, "manual_edits": edits}


def run(cmd, cwd=REPO, check=True):
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return proc


def touched_paths(patch_path):
    """Relative paths the patch writes, as they appear after --directory=vllm.

    Files are named on the `+++ b/<path>` line. Some patches in this series
    (the "local, not exported from the fork" ones) have no `diff --git` header
    at all, so the +++ line is the only reliable source.
    """
    paths = []
    with open(patch_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("+++ b/"):
                rel = line[len("+++ b/"):].strip()
                if rel and rel != "/dev/null":
                    paths.append(rel)
    return paths


def split_sections(patch_path, skip_files, out=None):
    """Return a patch whose per-file sections exclude `skip_files`.

    Returns (tmp_path, kept_paths). `git apply --exclude` matches on the path
    after --directory is prepended, which makes it awkward to use for a patch
    whose files live at different depths; splitting the text is unambiguous.
    """
    with open(patch_path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    parts = re.split(r"(?m)^(?=diff --git )", text)
    header = parts[0] if not parts[0].startswith("diff --git ") else ""
    sections = parts[1:] if header else parts

    kept, kept_paths = [], []
    for section in sections:
        first = section.split("\n", 1)[0]
        name = first.split()[-1][2:] if len(first.split()) >= 4 else ""
        if any(name == skip or name.endswith("/" + skip) for skip in skip_files):
            continue
        kept.append(section)
        kept_paths.append(name)

    tmp = out or os.path.join(PATCH_DIR, "_partial.patch")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(header + "".join(kept))
    return tmp, kept_paths


def added_block(patch_path, target, hunk_index=0):
    """The lines a patch adds to hunk ``hunk_index`` of ``target``, as one block.

    Used by the rebased patches: their first ``flash_attn.py`` hunk is pure
    addition whose surrounding context is 0.28-era code, so instead of
    rewriting it (and risking a hand-transcription error) we lift the added
    lines verbatim out of the patch and re-anchor them. Any later hunks for the
    same file stay in the patch and apply normally.
    """
    lines = []
    current = None
    index = -1
    with open(patch_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("+++ b/"):
                current = line[len("+++ b/"):].strip()
                index = -1
                continue
            if line.startswith("diff --git "):
                current = None
                continue
            if current == target and line.startswith("@@"):
                index += 1
                continue
            if current == target and index == hunk_index and line.startswith("+"):
                lines.append(line[1:])
    return "".join(lines)


def drop_hunk(patch_path, target, hunk_index, out):
    """Rewrite the patch with hunk ``hunk_index`` of ``target`` removed."""
    with open(patch_path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    parts = re.split(r"(?m)^(?=diff --git )", text)
    header = parts[0] if not parts[0].startswith("diff --git ") else ""
    sections = parts[1:] if header else parts

    kept_sections = []
    for section in sections:
        head = section.split("\n", 1)[0]
        name = head.split()[-1][2:] if len(head.split()) >= 4 else ""
        if name != target:
            kept_sections.append(section)
            continue
        pieces = re.split(r"(?m)^(?=@@ )", section)
        preamble, hunks = pieces[0], pieces[1:]
        if hunk_index >= len(hunks):
            raise RuntimeError(
                f"{os.path.basename(patch_path)}: hunk {hunk_index} of {target}"
                " not found"
            )
        kept = [h for i, h in enumerate(hunks) if i != hunk_index]
        if not kept:
            # Every hunk of this file was dropped, so drop the file section
            # with them -- a headers-only section makes git apply unhappy.
            continue
        kept_sections.append(preamble + "".join(kept))

    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(header + "".join(kept_sections))
    return out


def apply_manual_edits(edits, dry_run):
    """Insert each (rel_path, anchor, insert) into vllm/<rel_path>.

    Idempotent: an anchor that is gone but whose insert is present counts as
    already done. Returns True when something changed.
    """
    changed = False
    for rel, anchor, insert in edits:
        path = os.path.join(REPO, "vllm", rel)
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        # "Is this insert already in the file?" must be answered with *all* its
        # non-blank lines, not one of them: the registry insert's first line is
        # `# syv patch (<name>)`, which is a substring of the annotation
        # insert's banner (`# syv patch (<name>): ...`), so a single-line
        # substring test reports the registry knob as present right after the
        # annotation went in -- and silently skips it.
        wanted = [ln.strip() for ln in insert.split("\n") if ln.strip()]
        present = {ln.strip() for ln in content.split("\n")}
        if wanted and all(ln in present for ln in wanted):
            print(f"           already present: {rel}")
            continue
        if anchor not in content:
            raise RuntimeError(
                f"manual edit anchor not found in {rel}; the tree has drifted:\n"
                f"{anchor!r}"
            )
        print(f"           inserting knob into {rel}")
        if not dry_run:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content.replace(anchor, anchor + insert, 1))
        changed = True
    return changed


def textually_applied(patch_path, skip_files=()):
    """Fallback: every added line already present in its target file.

    Needed because a later patch can insert a line in the middle of an earlier
    patch's context, which breaks both the forward and the reverse git-apply
    check while the change itself is plainly there. `marlin-repack-staged-sm80`
    does exactly that to `marlin-int8-layer-select`'s envs.py hunk.
    """
    current = None
    pending_by_file = {}
    with open(patch_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("+++ b/"):
                current = line[len("+++ b/"):].strip()
                continue
            if line.startswith("diff --git "):
                current = None
                continue
            if current is None or not line.startswith("+"):
                continue
            if any(current == skip or current.endswith("/" + skip)
                   for skip in skip_files):
                continue
            body = line[1:].rstrip("\n")
            if body.strip():
                pending_by_file.setdefault(current, []).append(body)

    if not pending_by_file:
        return False

    for rel, lines in pending_by_file.items():
        path = os.path.join(REPO, "vllm", rel)
        if not os.path.exists(path):
            return False
        with open(path, encoding="utf-8", errors="replace") as handle:
            have = {cand.strip() for cand in handle.read().split("\n")}
        if not all(ln.strip() in have for ln in lines):
            return False
    return True


def state(patch_path, skip_files=()):
    """'applied', 'pending', or 'conflict'."""
    if run(["git", "apply", "--check", "-R", "-p1", "--directory=vllm", patch_path],
           check=False).returncode == 0:
        return "applied"
    if textually_applied(patch_path, skip_files):
        return "applied"
    if run(["git", "apply", "--check", "-p1", "--directory=vllm", patch_path],
           check=False).returncode == 0:
        return "pending"
    return "conflict"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="dry run")
    parser.add_argument("--repo-only", action="store_true",
                        help="patch vllm/ but do not mirror into site-packages")
    parser.add_argument("--batch", choices=["1", "2", "3", "all"], default="all",
                        help="which batch to process; 'all' keeps series order")
    args = parser.parse_args()

    prereq = os.path.join(PATCH_DIR, f"{PREREQ}.patch")
    if os.path.exists(prereq):
        print(f"prereq {PREREQ}: {state(prereq)}")

    synced_files = set()
    queue = {
        "1": CARRIED,
        "2": CARRIED_B,
        "3": CARRIED_C,
        "all": CARRIED + CARRIED_B + CARRIED_C,
    }[args.batch]

    for name, why in queue:
        patch_path = os.path.join(PATCH_DIR, f"{name}.patch")
        if not os.path.exists(patch_path):
            print(f"  MISSING  {name}")
            continue

        partial = rebased_partial(name) or PARTIAL.get(name)
        if partial and "patch" in partial:
            target = partial["patch"]
            kept = touched_paths(target)
            st = state(target)
            applied = st == "applied"
            print(f"  {'applied ' if applied else 'APPLYING'} {name}"
                  f"  (rebased: {kept})")
        elif partial:
            target, kept = split_sections(patch_path, partial["skip_files"])
            st = state(target)
            applied = st == "applied"
            print(f"  {'applied ' if applied else 'APPLYING'} {name}  (partial: {kept})")
        else:
            target, kept = patch_path, touched_paths(patch_path)
            st = state(patch_path)
            applied = st == "applied"
            print(f"  {'applied ' if applied else 'APPLYING'} {name}")
        print(f"           {why}")

        if st == "conflict" and not partial:
            print("           CONFLICT -- a conflict here usually means a missing"
                  " predecessor; the series applies in order")
            continue

        if not applied:
            if st == "conflict" and partial:
                # A conflict on the remaining hunks usually just means an
                # earlier patch in the series has not landed yet (batch 2's
                # hunks quote each other's lines). Applying is still the right
                # move -- if the conflict is real, git apply raises and we stop
                # instead of silently skipping the hunks.
                print("           (conflict on remaining hunks; applying anyway)")
            if not args.check:
                run(["git", "apply", "-p1", "--directory=vllm", target])
            st = "pending"

        if partial:
            apply_manual_edits(partial["manual_edits"], args.check)
            # Files touched only by a manual insert (envs.py, and flash_attn.py
            # for the rebased pair) are not in `kept` but do need mirroring.
            synced_files.update(rel for rel, _anchor, _ins in partial["manual_edits"])

        synced_files.update(kept)

    if args.check:
        print("\n(dry run -- nothing written)")
        return

    if not synced_files:
        print("\nno files to mirror")
        return

    if args.repo_only:
        print(f"\nrepo only; {len(synced_files)} file(s) touched")
        return

    if not os.path.isdir(SITE_PACKAGES_VLLM):
        print(f"\nno installed package at {SITE_PACKAGES_VLLM}; skipping mirror")
        return

    print(f"\nmirroring into {SITE_PACKAGES_VLLM}")
    for rel in sorted(synced_files):
        src = os.path.join(REPO, "vllm", rel)
        dst = os.path.join(SITE_PACKAGES_VLLM, rel)
        if not os.path.exists(src):
            print(f"  skip (absent in tree): {rel}")
            continue
        if os.path.exists(dst):
            with open(src, "rb") as a, open(dst, "rb") as b:
                if a.read() == b.read():
                    print(f"  same: {rel}")
                    continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"  copied: {rel}")

    # Stale bytecode would shadow the edit on some import paths.
    for root, _dirs, files in os.walk(SITE_PACKAGES_VLLM):
        if "__pycache__" not in root:
            continue
        for fname in files:
            if fname.endswith(".pyc"):
                try:
                    os.unlink(os.path.join(root, fname))
                except OSError:
                    pass

    print("\ndone")


if __name__ == "__main__":
    sys.exit(main())
