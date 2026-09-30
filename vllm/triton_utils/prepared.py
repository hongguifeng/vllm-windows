# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bypass Triton's Python dispatch front end for steady-state launches.

A ``kernel[grid](...)`` call spends roughly ten microseconds before the CUDA
launcher ever sees it: Triton binds the arguments, computes a cache key, checks
that referenced globals have not changed and builds launch metadata. Kernels
that vLLM launches on every decoding step repeat that work with an unchanging
call shape, so the front end can be cached next to the compiled kernel.

``install_prepared_launch()`` puts a fast path in front of ``JITFunction.run``.
A call is served from the cache only when its cheap key reproduces the
specialization recorded when the kernel was compiled: tensor dtype and 16 byte
pointer alignment, integer value classes, and the exact value of every
constexpr. Any specialization axis we cannot interpret, changed globals, or an
installed launch hook falls back to Triton's own dispatch, so correctness never
depends on the fast path. Enable with ``VLLM_TRITON_PREPARED_LAUNCH=1``.
"""

from __future__ import annotations

from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)

if HAS_TRITON:
    import torch
    from triton import knobs
    from triton.runtime import driver
    from triton.runtime.jit import JITFunction

# The only specialization attributes Triton is allowed to have used: divisible
# by sixteen, and equal to one. Anything else keeps the normal dispatch.
_KNOWN_ATTRS = frozenset({"D", "1"})

# Bound the number of prepared call shapes remembered per kernel.
_MAX_ENTRIES = 32

_ORIG_RUN = None
_stats = {"fast": 0, "full": 0, "checked": 0, "mismatch": 0}
_strict = False


def _tag(arg: Any) -> tuple | None:
    """Everything Triton specializes a plain argument on, or None if unknown."""
    if isinstance(arg, torch.Tensor):
        return ("t", str(arg.dtype), arg.data_ptr() % 16 == 0)
    if isinstance(arg, bool):
        return ("b",)
    if isinstance(arg, int):
        return ("i", arg == 1, arg % 16 == 0, arg >= 2**31)
    if isinstance(arg, float):
        return ("f",)
    return None


def _constexpr_names(jit) -> frozenset:
    names = getattr(jit, "_vllm_constexpr_names", None)
    if names is None:
        names = frozenset(p.name for p in jit.params if p.is_constexpr)
        jit._vllm_constexpr_names = names
    return names


def _param_names(jit) -> frozenset:
    names = getattr(jit, "_vllm_param_names", None)
    if names is None:
        names = frozenset(p.name for p in jit.params)
        jit._vllm_param_names = names
    return names


def _key(jit, args: tuple, kwargs: dict) -> tuple | None:
    """Cache key that refines every axis Triton specializes on."""
    constexprs = _constexpr_names(jit)
    params = _param_names(jit)
    parts: list[Any] = []
    for index, arg in enumerate(args):
        if index < len(jit.params) and jit.params[index].is_constexpr:
            if not isinstance(arg, (int, float, str, bool)):
                return None
            parts.append(("c", arg))
        else:
            tag = _tag(arg)
            if tag is None:
                return None
            parts.append(tag)
    for name in sorted(kwargs):
        value = kwargs[name]
        if name in constexprs:
            if not isinstance(value, (int, float, str, bool)):
                return None
            parts.append((name, value))
        elif name in params:
            tag = _tag(value)
            if tag is None:
                return None
            parts.append((name, tag))
        else:
            # Compile options are part of Triton's own cache key.
            parts.append(("o", name, value))
    return tuple(parts)


def _hook_active(hook: Any) -> bool:
    """Triton keeps an empty hook chain by default; only registered calls count."""
    calls = getattr(hook, "calls", None)
    if calls is None:
        return hook is not None
    return bool(calls)


def _strict_check(jit, entry, args: tuple, kwargs: dict) -> bool:
    """Ask Triton's own dispatch which compiled kernel these args select.

    ``warmup=True`` runs the whole front end without launching anything, so the
    kernel it returns is exactly what the slow path would have used. This is
    verification only: it pays the dispatch cost the fast path avoids, so it
    belongs in a measurement run rather than in production.
    """
    try:
        reference = _ORIG_RUN(jit, *args, grid=None, warmup=True, **kwargs)
    except Exception:  # pragma: no cover - defensive
        return True
    if reference is None:
        return True
    matches = reference is entry.compiled
    if not matches:
        _stats["mismatch"] += 1
        logger.warning(
            "prepared launch mismatch for %s: Triton's dispatch selects kernel "
            "%s, the cache holds %s",
            entry.name,
            id(reference),
            id(entry.compiled),
        )
    return matches


class _Prepared:
    """A compiled kernel plus how to rebuild its launch arguments."""

    def __init__(self, jit, compiled, slots: list[tuple[str, Any]]):
        self.jit = jit
        self.compiled = compiled
        self.slots = slots
        self.name = compiled.name
        self.hits = 0

    def usable(self) -> bool:
        """The fast path may only skip work Triton would have skipped anyway."""
        return (
            not _hook_active(knobs.runtime.launch_enter_hook)
            and not _hook_active(knobs.runtime.launch_exit_hook)
            and not self.jit.used_global_vals
            and not getattr(self.jit, "pre_run_hooks", ())
        )

    def launch(self, grid: tuple, args: tuple, kwargs: dict):
        values = [
            args[payload]
            if kind == "a"
            else kwargs[payload]
            if kind == "k"
            else payload
            for kind, payload in self.slots
        ]
        stream = torch.cuda.current_stream().cuda_stream
        grid_x = grid[0]
        grid_y = grid[1] if len(grid) > 1 else 1
        grid_z = grid[2] if len(grid) > 2 else 1
        self.compiled.run(
            grid_x,
            grid_y,
            grid_z,
            stream,
            self.compiled.function,
            self.compiled.packed_metadata,
            None,
            None,
            None,
            *values,
        )
        self.hits += 1
        _stats["fast"] += 1
        if self.hits % 10000 == 0:
            logger.debug(
                "prepared launch %s: %d hits (fast %d, full %d)",
                self.name,
                self.hits,
                _stats["fast"],
                _stats["full"],
            )
        return self.compiled


def _build(jit, compiled, args: tuple, kwargs: dict) -> _Prepared | None:
    """Record a prepared launch, or decline if any part is not understood."""
    if jit.used_global_vals or getattr(jit, "pre_run_hooks", ()):
        return None
    if _hook_active(knobs.runtime.launch_enter_hook):
        return None
    if _hook_active(knobs.runtime.launch_exit_hook):
        return None
    if hasattr(compiled, "result"):
        # Still asynchronously compiling; nothing safe to cache yet.
        return None
    try:
        device = driver.active.get_current_device()
        binder = jit.device_caches[device][4]
        bound_args, specialization, _ = binder(*args, **kwargs)
    except Exception:  # pragma: no cover - defensive
        return None

    slots: list[tuple[str, Any]] = []
    values = list(bound_args.values())
    for index, param in enumerate(jit.params):
        name = param.name
        value = values[index]
        if param.is_constexpr:
            slots.append(("c", value))
        elif index < len(args) and name not in kwargs:
            slots.append(("a", index))
        elif name in kwargs:
            slots.append(("k", name))
        else:
            # Filled in from the parameter default, which cannot vary.
            slots.append(("c", value))

    for index, param in enumerate(jit.params):
        if param.is_constexpr:
            continue
        entry = specialization[index]
        value = bound_args[param.name]
        for token in entry[1:]:
            if token is None:
                continue
            if token not in _KNOWN_ATTRS:
                return None
            if token == "1" and value != 1:
                return None
            if token == "D":
                if isinstance(value, torch.Tensor):
                    if value.data_ptr() % 16 != 0:
                        return None
                elif not isinstance(value, int) or value % 16 != 0:
                    return None

    # Touches the property that loads the module and builds the launcher.
    compiled.run  # noqa: B018
    return _Prepared(jit, compiled, slots)


def install_prepared_launch(strict: bool | None = None) -> bool:
    """Route repeat Triton launches through a cached compiled kernel.

    ``strict`` re-checks every cache hit against Triton's own dispatch, which
    is slow but proves the cheap key never picks the wrong kernel. Defaults to
    ``VLLM_TRITON_PREPARED_LAUNCH_STRICT``. Returns True when the fast path is
    live. Safe to call more than once.
    """
    global _ORIG_RUN, _strict
    if not HAS_TRITON or not envs.VLLM_TRITON_PREPARED_LAUNCH:
        return False
    if _ORIG_RUN is not None:
        return True

    if strict is None:
        strict = envs.VLLM_TRITON_PREPARED_LAUNCH_STRICT
    _strict = strict

    _ORIG_RUN = JITFunction.run

    def run(self, *args, grid=None, warmup=False, **kwargs):
        cache = None
        key = None
        if not warmup and isinstance(grid, (tuple, list)) and grid:
            key = _key(self, args, kwargs)
            if key is not None:
                cache = getattr(self, "_vllm_prepared", None)
                if cache is None:
                    cache = self._vllm_prepared = {}
                entry = cache.get(key)
                if entry is not None and entry.usable():
                    if _strict:
                        _strict_check(self, entry, args, kwargs)
                        _stats["checked"] += 1
                        if _stats["checked"] % 2000 == 0:
                            logger.info(
                                "prepared launch strict check: %d hits verified, "
                                "%d mismatch",
                                _stats["checked"],
                                _stats["mismatch"],
                            )
                    return entry.launch(grid, args, kwargs)
        compiled = _ORIG_RUN(self, *args, grid=grid, warmup=warmup, **kwargs)
        _stats["full"] += 1
        if (
            not warmup
            and key is not None
            and compiled is not None
            and len(cache) < _MAX_ENTRIES
            and key not in cache
        ):
            entry = _build(self, compiled, args, kwargs)
            if entry is not None:
                cache[key] = entry
        return compiled

    JITFunction.run = run
    logger.info(
        "Enabled prepared Triton launches; %d call shapes cached per kernel%s",
        _MAX_ENTRIES,
        " (strict verification on)" if _strict else "",
    )
    return True


def prepared_launch_stats() -> dict[str, int]:
    """Launch counts by path, plus strict-verification results if enabled."""
    return dict(_stats)
