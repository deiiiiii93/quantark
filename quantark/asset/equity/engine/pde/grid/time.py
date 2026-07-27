"""The ONE time builder (spec §4.3).

Every ``event_times`` entry becomes a grid node exactly; fill between nodes is
sized solely by ``steps_per_day`` (no per-interval floor beyond 1 — the ≥10
floor caused the historical ~10x grid inflation); ``max_steps`` caps fill via
extras-scaling and can never move or drop a node. Damping schedules for the
θ-loop and the 2D ADI stepper are derived here, once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.engine.pde.grid.request import GridRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class TimeLayout:
    """Per-product time geometry; compared by identity (eq=False)."""

    t: np.ndarray
    dt: np.ndarray
    step_of: Mapping[float, int]
    event_damping_steps: frozenset
    terminal_damping_steps: frozenset
    requested_steps: int
    actual_steps: int
    cap_exceeded: bool


def build_time(request: GridRequest, config: GridConfig) -> TimeLayout:
    """Build the event-aligned time grid for one product.

    Step-index convention: step ``k`` advances the backward solve from node
    ``k+1`` to node ``k``; a step is "after" an event node (backward time)
    when its index is ``node_index - j`` for ``j = 1..count``.
    """
    tau, events = request.tau, request.event_times
    boundaries = np.array([0.0, *events, tau], dtype=float)
    lengths = np.diff(boundaries)
    days = np.maximum(1.0, lengths * float(config.day_count))
    fill = np.maximum(1, np.round(days * float(config.steps_per_day)).astype(int))
    requested = int(fill.sum())

    n_int, cap = len(lengths), int(config.max_steps)
    cap_exceeded = False
    if n_int > cap:
        # Mandatory nodes are inviolable: keep every node, minimal fill,
        # exceed the cap and say so (spec §4.3).
        fill = np.ones_like(fill)
        cap_exceeded = True
        logger.warning(
            "time grid: %d event intervals exceed max_steps=%d; keeping all "
            "nodes and exceeding the cap",
            n_int,
            cap,
        )
    elif requested > cap:
        # Scale only the extras into the remaining budget: baselines (1 per
        # interval) + scaled extras <= baselines + budget = cap.
        extras = fill - 1
        budget = cap - n_int
        total_extras = int(extras.sum())
        if total_extras > 0:
            fill = 1 + np.floor(
                extras.astype(float) * (budget / total_extras)
            ).astype(int)

    points = [np.array([0.0])]
    for start, end, n in zip(boundaries[:-1], boundaries[1:], fill):
        points.append(np.linspace(start, end, int(n) + 1)[1:])
    t = np.concatenate(points)
    dt = np.diff(t)

    step_of = {}
    for e in events:
        k = int(np.searchsorted(t, e))
        # linspace endpoints reproduce the boundary values exactly, so the
        # verbatim request float IS the node value.
        assert t[k] == e
        step_of[e] = k

    event_damp = frozenset(
        k - j
        for k in step_of.values()
        for j in range(1, int(config.event_damping_steps) + 1)
        if k - j >= 0
    )
    n_steps = len(dt)
    term_damp = frozenset(
        n_steps - j
        for j in range(1, int(config.terminal_damping_steps) + 1)
        if n_steps - j >= 0
    )

    t.setflags(write=False)
    dt.setflags(write=False)
    return TimeLayout(
        t=t,
        dt=dt,
        step_of=MappingProxyType(dict(step_of)),
        event_damping_steps=event_damp,
        terminal_damping_steps=term_damp,
        requested_steps=requested,
        actual_steps=n_steps,
        cap_exceeded=cap_exceeded,
    )
