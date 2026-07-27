"""Event application: cell-average projection operators + the four-stage
EventSchedule (spec §4.2).

The projection math is the certified conservative dual-cell projection,
MOVED verbatim from ``event_projection.py`` (which now re-exports from here
until its Phase-4 deletion): each node's dual cell receives the exact cell
average of the complete composite event function, with branches read as the
grid's piecewise-linear interpolants. Away from the threshold this reduces to
the untouched branch values; only the straddling cell is split, by exact
sub-cell integration. Properties (tier-1 tested): weights in [0, 1]; constant
preservation (P @ 1 = 1); envelope containment; affine exactness in the
straddling cell; FP-scale threshold continuity; second-order convergence.

``EventSchedule`` carries per-solve event semantics as PURE transforms over
named state blocks (S always axis 0; trailing axes broadcast). Stages:
``terminal`` (t = tau, once, after terminal-payoff construction), ``apply``
(interior event nodes), ``continuous`` (every step — continuous/BGK KI
coupling, 2D KO injection), ``valuation_readout`` (t = 0 pointwise inclusive
trigger semantics at spot).
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np

from quantark.asset.equity.engine.pde.grid.space import SpatialLayout
from quantark.util.numerical import safe_log

# ---------------------------------------------------------------------------
# Certified projection core — moved verbatim from event_projection.py
# ---------------------------------------------------------------------------


def _dual_cell_edges(x_vec: np.ndarray) -> np.ndarray:
    """Edges of each node's dual cell, clipped to the grid domain."""
    edges = np.empty(x_vec.shape[0] + 1, dtype=float)
    edges[0] = x_vec[0]
    edges[-1] = x_vec[-1]
    edges[1:-1] = 0.5 * (x_vec[1:] + x_vec[:-1])
    return edges


def breach_fractions(
    x_vec: np.ndarray, b_x: float, breach_up: bool
) -> np.ndarray:
    """Fraction of each node's dual cell lying inside the breach region."""
    x_vec = np.asarray(x_vec, dtype=float)
    edges = _dual_cell_edges(x_vec)
    width = np.diff(edges)
    overlap_up = np.clip(edges[1:] - np.maximum(edges[:-1], b_x), 0.0, width)
    frac_up = overlap_up / width
    return frac_up if breach_up else 1.0 - frac_up


def _interp_columns(
    x_vec: np.ndarray, values: np.ndarray, point: float
) -> np.ndarray:
    """Piecewise-linear interpolation of column data at a single point."""
    k = int(np.searchsorted(x_vec, point, side="right")) - 1
    k = min(max(k, 0), x_vec.shape[0] - 2)
    x0, x1 = x_vec[k], x_vec[k + 1]
    w = 0.0 if x1 == x0 else (point - x0) / (x1 - x0)
    return (1.0 - w) * values[k] + w * values[k + 1]


def _straddle_cell(x_vec: np.ndarray, edges: np.ndarray, b_x: float):
    """Index and edges of the dual cell strictly containing ``b_x``."""
    if not (edges[0] < b_x < edges[-1]):
        return None
    i = int(np.searchsorted(edges, b_x, side="right")) - 1
    i = min(max(i, 0), x_vec.shape[0] - 1)
    e_lo, e_hi = float(edges[i]), float(edges[i + 1])
    if not (e_lo < b_x < e_hi):
        return None
    return i, e_lo, e_hi


def _pl_integral(
    x_vec: np.ndarray, values: np.ndarray, a: float, b: float
) -> np.ndarray:
    """Exact integral of the piecewise-linear interpolant over ``[a, b]``."""
    empty = np.zeros(values.shape[1:] if values.ndim > 1 else (), dtype=float)
    if b <= a:
        return empty
    lo = int(np.searchsorted(x_vec, a, side="right"))
    hi = int(np.searchsorted(x_vec, b, side="left"))
    points = [a] + [float(t) for t in x_vec[lo:hi]] + [b]
    integral = empty
    for p, q in zip(points[:-1], points[1:]):
        vp = _interp_columns(x_vec, values, p)
        vq = _interp_columns(x_vec, values, q)
        integral = integral + 0.5 * (q - p) * (vp + vq)
    return integral


def project_breach_jump(
    x_vec: np.ndarray,
    b_x: float,
    jump: np.ndarray,
    breach_up: bool,
) -> np.ndarray:
    """Dual-cell average of ``1_breach(x) * jump_PL(x)`` per node."""
    x_vec = np.asarray(x_vec, dtype=float)
    jump = np.asarray(jump, dtype=float)
    frac = breach_fractions(x_vec, b_x, breach_up)
    if jump.ndim == 1:
        out = jump * frac
    else:
        out = jump * frac[:, None]

    edges = _dual_cell_edges(x_vec)
    straddle = _straddle_cell(x_vec, edges, b_x)
    if straddle is None:
        return out
    i, e_lo, e_hi = straddle
    lo, hi = (b_x, e_hi) if breach_up else (e_lo, b_x)
    out[i] = _pl_integral(x_vec, jump, lo, hi) / (e_hi - e_lo)
    return out


def project_piecewise_event(
    x_vec: np.ndarray,
    breaks,
    branches,
) -> np.ndarray:
    """Dual-cell average of a piecewise event function with K thresholds.

    Region j (0-based) is ``{b_j <= x < b_(j+1)}`` with b_0 = -inf and
    b_(K+1) = +inf; ``branches[j]`` is active on region j. A dual cell crossed
    by one or more thresholds receives the exact cell average of the complete
    composite function — the one-pass generalization coincident triggers
    (phoenix coupon + KO sharing a cell) require.
    """
    x_vec = np.asarray(x_vec, dtype=float)
    n = x_vec.shape[0]
    breaks = [float(b) for b in breaks]
    if any(b2 < b1 for b1, b2 in zip(breaks, breaks[1:])):
        raise ValueError("breaks must be sorted ascending")
    if len(branches) != len(breaks) + 1:
        raise ValueError("need exactly len(breaks) + 1 branches")

    arrs = [np.asarray(v, dtype=float) for v in branches]
    if all(a.ndim == 0 for a in arrs):
        arrs = [np.full(n, float(a)) for a in arrs]
    else:
        arrs = [np.array(a, dtype=float) for a in np.broadcast_arrays(*arrs)]

    region_of_node = np.searchsorted(breaks, x_vec, side="right")
    stacked = np.stack(arrs)
    out = np.array(stacked[region_of_node, np.arange(n)], dtype=float)

    edges = _dual_cell_edges(x_vec)
    for i in sorted(
        {
            s[0]
            for s in (_straddle_cell(x_vec, edges, b) for b in breaks)
            if s is not None
        }
    ):
        e_lo, e_hi = float(edges[i]), float(edges[i + 1])
        cuts = [e_lo] + sorted({b for b in breaks if e_lo < b < e_hi}) + [e_hi]
        total = np.zeros(arrs[0].shape[1:], dtype=float)
        for a, c in zip(cuts[:-1], cuts[1:]):
            mid = 0.5 * (a + c)
            r = int(np.searchsorted(breaks, mid, side="right"))
            total = total + _pl_integral(x_vec, arrs[r], a, c)
        out[i] = total / (e_hi - e_lo)
    return out


def project_event_values(
    x_vec: np.ndarray,
    b_x: float,
    v_survive,
    v_breach,
    breach_up: bool,
) -> np.ndarray:
    """Post-event nodal values: dual-cell average of the complete event
    function (survive branch on one side of the threshold, breach on the
    other; the straddling cell integrated exactly)."""
    x_vec = np.asarray(x_vec, dtype=float)
    n = x_vec.shape[0]
    v_s = np.asarray(v_survive, dtype=float)
    v_b = np.asarray(v_breach, dtype=float)
    if v_s.ndim == 0 and v_b.ndim == 0:
        v_s = np.full(n, float(v_s))
        v_b = np.full(n, float(v_b))
    else:
        v_s, v_b = np.broadcast_arrays(v_s, v_b)
        v_s = np.array(v_s, dtype=float)
        v_b = np.array(v_b, dtype=float)

    frac = breach_fractions(x_vec, b_x, breach_up)
    f = frac if v_s.ndim == 1 else frac[:, None]
    out = v_s + (v_b - v_s) * f

    edges = _dual_cell_edges(x_vec)
    straddle = _straddle_cell(x_vec, edges, b_x)
    if straddle is None:
        return out
    i, e_lo, e_hi = straddle
    if breach_up:
        total = _pl_integral(x_vec, v_s, e_lo, b_x) + _pl_integral(
            x_vec, v_b, b_x, e_hi
        )
    else:
        total = _pl_integral(x_vec, v_b, e_lo, b_x) + _pl_integral(
            x_vec, v_s, b_x, e_hi
        )
    out[i] = total / (e_hi - e_lo)
    return out


# ---------------------------------------------------------------------------
# Layout-aware wrappers (barriers in PRICE space; S on axis 0)
# ---------------------------------------------------------------------------


def breach_weights(
    layout: SpatialLayout, barrier: float, breach_up: bool
) -> np.ndarray:
    """Dual-cell breach fractions for a price-space barrier."""
    return breach_fractions(layout.x, float(safe_log(barrier)), breach_up)


def project_between(
    layout: SpatialLayout,
    barrier: float,
    breach_up: bool,
    v_breach: np.ndarray,
    v_survive: np.ndarray,
) -> np.ndarray:
    """Cell-average projection of a single-threshold event on axis 0.

    Accepts ``(n_s,)`` or ``(n_s, k)`` blocks; a 2D ``(n_s, n_v)`` surface
    projects identically column-wise (the operator acts on the S-axis only).
    """
    return project_event_values(
        layout.x, float(safe_log(barrier)), v_survive, v_breach, breach_up
    )


def project_piecewise(
    layout: SpatialLayout, barriers, branches
) -> np.ndarray:
    """One-pass cell-average projection of a K-threshold piecewise event."""
    breaks = [float(safe_log(b)) for b in barriers]
    return project_piecewise_event(layout.x, breaks, branches)


# ---------------------------------------------------------------------------
# EventSchedule — the four stages (spec §4.2)
# ---------------------------------------------------------------------------

States = Mapping[str, np.ndarray]
_Transform = Callable[[States], States]
_StepTransform = Callable[[int, States], States]
_Readout = Callable[[float, States], float]


class EventSchedule:
    """Per-solve event semantics as pure transforms over named state blocks.

    Every transform receives a mapping of named blocks (S always axis 0) and
    returns a NEW mapping — inputs are never mutated. Missing hooks are
    identity (or, for ``valuation_readout``, must be supplied by the solver
    that prices).
    """

    def __init__(
        self,
        interior: Optional[Mapping[int, _Transform]] = None,
        continuous: Optional[_StepTransform] = None,
        terminal: Optional[_Transform] = None,
        readout: Optional[_Readout] = None,
    ):
        self._interior = dict(interior or {})
        self._continuous = continuous
        self._terminal = terminal
        self._readout = readout

    @property
    def interior_steps(self) -> frozenset:
        return frozenset(self._interior)

    def terminal(self, states: States) -> States:
        """t = tau stage: once, after terminal-payoff construction."""
        if self._terminal is None:
            return states
        return self._terminal(states)

    def apply(self, step: int, states: States) -> States:
        """Interior stage: when the backward loop lands on an event node."""
        fn = self._interior.get(step)
        if fn is None:
            return states
        return fn(states)

    def continuous(self, step: int, states: States) -> States:
        """Every-step stage: continuous-monitoring regime coupling."""
        if self._continuous is None:
            return states
        return self._continuous(step, states)

    def valuation_readout(self, spot: float, states: States) -> float:
        """t = 0 stage: pointwise inclusive trigger semantics at spot."""
        if self._readout is None:
            raise NotImplementedError(
                "this EventSchedule was built without a valuation readout"
            )
        return self._readout(spot, states)
