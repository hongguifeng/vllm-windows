# SPDX-License-Identifier: Apache-2.0
"""Device-side phase spans for steps that replay a CUDA graph.

Set ``VLLM_PHASE_EVENTS=1`` before the model is built. Each ``begin``/``end``
pair then falls into the captured graph as an event-record node, so every
replay re-records the same pair and the span is measured on the device. The
worker harvests once per step by querying readiness rather than
synchronizing, which keeps the observer off the critical path.

Names carry the layer index they belong to, so the printed table can be
scaled across layers of the same type; phases nest, and the residue between a
layer span and its inner spans is the hyper-connection work.
"""

from __future__ import annotations

import os
import statistics
from collections import defaultdict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Enough samples to settle a median without growing without bound.
MAX_SAMPLES = 50_000


def phase_events_enabled() -> bool:
    """Whether in-graph phase timing was requested for this process."""
    return os.environ.get("VLLM_PHASE_EVENTS", "0") == "1"


# The plan runs where the model is built. The worker finds the same object here
# instead of walking a module tree whose shape depends on the architecture.
_ACTIVE: PhaseEvents | None = None


def activate(events: PhaseEvents) -> None:
    """Make this the object the worker harvests."""
    global _ACTIVE
    _ACTIVE = events


def get_active() -> PhaseEvents | None:
    """The object the worker should harvest, if one was planned."""
    return _ACTIVE


class PhaseEvents:
    """Named start/end event pairs recorded on the forward stream.

    Args:
        log_every: Emit the phase table once this many harvests accumulate;
            zero takes ``VLLM_PHASE_EVENTS_LOG_EVERY``, defaulting to 2000.
    """

    def __init__(self, log_every: int = 0) -> None:
        if log_every <= 0:
            log_every = int(os.environ.get("VLLM_PHASE_EVENTS_LOG_EVERY", "2000"))
        self._pairs: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._recorded: dict[str, bool] = {}
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._log_every = max(1, log_every)
        self._harvests = 0

    def register(self, name: str) -> None:
        """Create the event pair that begin/end will time."""
        if name not in self._pairs:
            self._pairs[name] = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )

    def begin(self, name: str) -> None:
        pair = self._pairs.get(name)
        if pair is not None:
            pair[0].record()

    def end(self, name: str) -> None:
        pair = self._pairs.get(name)
        if pair is not None:
            pair[1].record()
            # Querying an event that was never recorded reports it as complete,
            # so elapsed time needs its own guard, set where python does run:
            # at capture time, or eagerly. Replay re-records without running.
            self._recorded[name] = True

    def harvest(self) -> None:
        """Read every pair whose latest records have completed.

        Called once per engine step from python that also runs when the step
        replays a graph. Pairs still in flight stay unread until a later
        harvest finds both of them complete, so no step waits on the device.
        """
        if torch.cuda.is_current_stream_capturing():
            # Records made while capturing are deferred into the graph, so a
            # query here would report them complete and elapsed time would be
            # meaningless. Wait for a replay instead.
            return
        for name, (start, end) in self._pairs.items():
            samples = self._samples[name]
            if len(samples) >= MAX_SAMPLES:
                continue
            if not self._recorded.get(name):
                continue
            if start.query() and end.query():
                samples.append(end.elapsed_time(start))
        self._harvests += 1
        if self._harvests % self._log_every == 0:
            self.log()

    def log(self) -> None:
        """Log the median, 90th percentile and maximum of every phase, in ms."""
        parts = []
        for name in sorted(self._pairs):
            values = sorted(self._samples[name])
            if not values:
                continue
            p50 = statistics.median(values)
            p90 = values[min(len(values) - 1, int(len(values) * 0.9))]
            parts.append(
                f"{name}={p50:.3f}/{p90:.3f}/{values[-1]:.3f} n={len(values)}"
            )
        logger.info("PHASES harvest=%d %s", self._harvests, " ".join(parts))
