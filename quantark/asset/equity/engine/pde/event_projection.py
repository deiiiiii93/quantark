"""Conservative dual-cell projection of discrete event operators.

A discretely observed trigger (coupon / KO / KI threshold) applies a Heaviside
jump to the value surface at an event date:

    V+(x) = 1_breach(x) * V_breach(x) + (1 - 1_breach(x)) * V_survive(x)

Sampling that discontinuity with a Boolean nodal mask assigns the entire cell
containing the threshold to one branch, displacing the effective trigger by
up to half a cell (an O(dx) one-sided error; see
``quantark/asset/equity/engine/docs/pde_auto_grid_investigation.md``).

This module represents the jump in the finite-volume sense instead: each
node's dual cell (the interval between the midpoints to its neighbours,
clipped to the domain) receives the exact cell average of the jump function,
with the two branches read as the grid's piecewise-linear interpolants.
Away from the threshold this reduces to the untouched branch values; only the
cell straddling the threshold is split, by exact sub-cell integration.

``project_event_values`` is the event operator used by the solvers: away
from the threshold each node keeps the pointwise value of its own branch;
the straddling node receives the exact cell average of the COMPLETE
composite function (survive branch on one side of the threshold, breach
branch on the other). Averaging only the jump and adding it to the
pointwise survive value mixes two inconsistent discretizations in that one
cell and can leave the branch envelope when the survive branch is steep
there (2026-07-23 review, blocking finding 1).

Properties (exercised by ``test/test_pde_event_projection.py``):

- linear in the branch data, weights in [0, 1];
- ``project_event_values`` stays inside the [min, max] envelope of the two
  branches — non-negative branches project to non-negative values;
- constant jumps are reproduced exactly and their total dual-cell mass equals
  the exact measure of the breach region;
- affine functions are integrated exactly inside the straddling cell;
- a floating-point-scale perturbation of the threshold moves the result by
  the same infinitesimal amount (nodal masks flip a whole cell instead).
"""

from __future__ import annotations

import numpy as np


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
    """Fraction of each node's dual cell lying inside the breach region.

    Args:
        x_vec: Strictly increasing grid coordinates (typically log-price).
        b_x: Threshold in the same coordinates. ``-inf``/``+inf`` are allowed
            and yield all-breached / all-survived for ``breach_up=True``.
        breach_up: True when the breach region is ``{x >= b_x}``.

    Returns:
        Array of fractions in [0, 1], one per node.
    """
    x_vec = np.asarray(x_vec, dtype=float)
    edges = _dual_cell_edges(x_vec)
    width = np.diff(edges)
    overlap_up = np.clip(edges[1:] - np.maximum(edges[:-1], b_x), 0.0, width)
    frac_up = overlap_up / width
    return frac_up if breach_up else 1.0 - frac_up


def _interp_columns(
    x_vec: np.ndarray, values: np.ndarray, point: float
) -> np.ndarray:
    """Piecewise-linear interpolation of column data at a single point.

    ``values`` has shape (n_nodes,) or (n_nodes, n_cols); the result has the
    trailing shape. The point must lie within [x_vec[0], x_vec[-1]].
    """
    k = int(np.searchsorted(x_vec, point, side="right")) - 1
    k = min(max(k, 0), x_vec.shape[0] - 2)
    x0, x1 = x_vec[k], x_vec[k + 1]
    w = 0.0 if x1 == x0 else (point - x0) / (x1 - x0)
    return (1.0 - w) * values[k] + w * values[k + 1]


def _straddle_cell(
    x_vec: np.ndarray, edges: np.ndarray, b_x: float
):
    """Index and edges of the dual cell strictly containing ``b_x``.

    Returns ``None`` when the threshold falls outside the domain or exactly
    on a cell face (the breach fractions are already exact there).
    """
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
    """Exact integral of the piecewise-linear interpolant over ``[a, b]``.

    Split at every grid node strictly inside the interval so each trapezoid
    spans a single linear segment of the interpolant.
    """
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
    """Dual-cell average of ``1_breach(x) * jump_PL(x)`` per node.

    Args:
        x_vec: Strictly increasing grid coordinates.
        b_x: Threshold coordinate (``+-inf`` allowed).
        jump: Nodal jump data ``V_breach - V_survive``, shape (n,) or (n, k).
        breach_up: True when the breach region is ``{x >= b_x}``.

    Returns:
        Projected jump with the same shape as ``jump``: ``jump * fraction``
        away from the threshold, and the exact piecewise-linear sub-cell
        integral divided by the cell width in the straddling cell.
    """
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

    ``breaks`` are sorted ascending thresholds b_1 <= ... <= b_K; region j
    (0-based) is ``{b_j <= x < b_(j+1)}`` with b_0 = -inf and b_(K+1) = +inf,
    and ``branches[j]`` (scalar, (n,) or (n, k)) is the function active on
    region j. Away from the thresholds each node keeps the pointwise value of
    the region containing it; a dual cell crossed by one or more thresholds
    receives the exact cell average of the complete composite piecewise-linear
    function — the one-pass generalization of ``project_event_values`` that
    composite triggers (e.g. a phoenix coupon and KO sharing a dual cell)
    require: projecting sequentially double-averages the shared cell
    [2026-07-24 review, finding 3]. Equal thresholds simply produce an empty
    region. For K=1 this matches ``project_event_values`` with
    ``breach_up=True`` up to FP arithmetic (branch values are assigned
    directly instead of via ``v_s + (v_b - v_s) * frac``).
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
        cuts = (
            [e_lo]
            + sorted({b for b in breaks if e_lo < b < e_hi})
            + [e_hi]
        )
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
    function.

    Away from the threshold each node keeps the pointwise value of its own
    branch (breach fraction 0 or 1). The straddling node receives the exact
    cell average of the composite piecewise-linear function — survive branch
    on the survive side of the threshold, breach branch on the breach side —
    which keeps the result inside the [min, max] envelope of the two
    branches (non-negative branches project to non-negative values).

    Args:
        x_vec: Strictly increasing grid coordinates.
        b_x: Threshold coordinate (``+-inf`` allowed).
        v_survive: Survive-branch values — scalar, (n,) or (n, k).
        v_breach: Breach-branch values — scalar, (n,) or (n, k).
        breach_up: True when the breach region is ``{x >= b_x}``.

    Returns:
        Projected nodal values, shape (n,) or (n, k).
    """
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
