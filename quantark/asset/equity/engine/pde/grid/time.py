"""The ONE time builder (spec §4.3).

Every ``event_times`` entry becomes a grid node exactly (indexed and damped),
and every ``extra_times`` entry a node only (indexed, never damped, no event
keyed on it); fill between nodes is
sized by the time basis in force — the ``steps_per_day`` density or the
``time_steps`` budget (no per-interval floor beyond 1 — the ≥10
floor caused the historical ~10x grid inflation); ``max_steps`` caps fill via
extras-scaling and can never move or drop a node. Damping schedules for the
θ-loop and the 2D ADI stepper are derived here, once.
"""

from __future__ import annotations

import logging
import dataclasses
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
    #: True whenever fewer steps were delivered than ``steps_per_day`` asked
    #: for: the extras were scaled into ``max_steps`` (``cap_exceeded`` stays
    #: False) or the mandatory nodes alone overflowed it (``cap_exceeded``
    #: True).  A refinement ladder that does not check this refines space
    #: while time stays put.
    fill_scaled: bool = False

    def step_at(self, obs_time: float) -> int:
        """Node index for an event-bearing time.

        Exact float hit first (the common case: the verbatim request float).
        Falls back to an ``is_close`` scan for times that came from a
        DIFFERENT schedule resolution of the same date (e.g. KO 1/12 vs KI
        4/48 rounded) — well-defined because construction deduplicates
        event nodes within the same tolerance, so at most one key matches.
        """
        hit = self.step_of.get(obs_time)
        if hit is not None:
            return hit
        from quantark.util.numerical import is_close

        for key, idx in self.step_of.items():
            if is_close(key, obs_time):
                return idx
        raise KeyError(obs_time)

    def __reduce__(self):
        # mappingproxy does not pickle or deep-copy; rebuild from plain values.
        state = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        state["step_of"] = dict(self.step_of)
        return (_rebuild_time_layout, (state,))


def _rebuild_time_layout(state: dict) -> TimeLayout:
    """``TimeLayout.__reduce__``'s constructor: read-only arrays, a read-only step map."""
    state = dict(state)
    for name in ("t", "dt"):
        arr = np.array(state[name], dtype=float)
        arr.setflags(write=False)
        state[name] = arr
    state["step_of"] = MappingProxyType(dict(state["step_of"]))
    return TimeLayout(**state)


def build_time(request: GridRequest, config: GridConfig) -> TimeLayout:
    """Build the event-aligned time grid for one product.

    Step-index convention: step ``k`` advances the backward solve from node
    ``k+1`` to node ``k``; a step is "after" an event node (backward time)
    when its index is ``node_index - j`` for ``j = 1..count``.
    """
    tau, events = request.tau, request.event_times
    nodes = tuple(sorted(set(events) | set(request.extra_times)))
    boundaries = np.array([0.0, *nodes, tau], dtype=float)
    lengths = np.diff(boundaries)
    if config.time_steps is not None:
        # Budget basis (spec §4.2, decision 16): the total spread over the
        # mandatory intervals by length, at least one step each. One
        # interval gives exactly linspace(0, tau, time_steps + 1).
        fill = np.maximum(
            1, np.round(float(config.time_steps) * lengths / float(tau)).astype(int)
        )
        requested = int(config.time_steps)
    else:
        days = np.maximum(1.0, lengths * float(config.day_count))
        fill = np.maximum(1, np.round(days * float(config.steps_per_day)).astype(int))
        requested = int(fill.sum())
    # The fill before the cap: "scaled" means the cap cut it, never rounding.
    unscaled = int(fill.sum())

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
    elif unscaled > cap:
        # Scale only the extras into the remaining budget: baselines (1 per
        # interval) + scaled extras <= baselines + budget = cap.
        extras = fill - 1
        budget = cap - n_int
        total_extras = int(extras.sum())
        if total_extras > 0:
            fill = 1 + np.floor(
                extras.astype(float) * (budget / total_extras)
            ).astype(int)
        logger.warning(
            "time grid: %d requested steps scaled to %d by max_steps=%d; "
            "raise GridConfig(max_steps=...) for the requested fill",
            unscaled,
            int(fill.sum()),
            cap,
        )

    # Nodes via linspace (exact endpoints); dt as the CONSTRUCTED constant
    # per interval, not np.diff(t): interior linspace spacings wobble by an
    # ULP, and downstream operator caches key on the dt float — one exact
    # value per interval keeps cache-on and cache-off arithmetic identical.
    # (diff(t) and dt may disagree by an ULP at interval ends; t is the
    # authority for calendar times, dt for step sizes — same convention as
    # the certified ADI _advance_tau endpoint pin.)
    points = [np.array([0.0])]
    dts = []
    for start, end, n in zip(boundaries[:-1], boundaries[1:], fill):
        n = int(n)
        points.append(np.linspace(start, end, n + 1)[1:])
        dts.append(np.full(n, (end - start) / n))
    t = np.concatenate(points)
    dt = np.concatenate(dts)

    step_of = {}
    for e in nodes:
        k = int(np.searchsorted(t, e))
        # linspace endpoints reproduce the boundary values exactly, so the
        # verbatim request float IS the node value.
        assert t[k] == e
        step_of[e] = k

    # Damping restarts follow EVENT nodes only; an extra node is a readout
    # point and must not degrade the scheme around it.
    event_damp = frozenset(
        step_of[e] - j
        for e in events
        for j in range(1, int(config.event_damping_steps) + 1)
        if step_of[e] - j >= 0
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
        fill_scaled=n_steps < unscaled,
    )
