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

Properties (exercised by ``test/test_pde_event_projection.py``):

- linear in the jump data, weights in [0, 1], no overshoot;
- constant jumps are reproduced exactly and their total dual-cell mass equals
  the exact measure of the breach region;
- affine jump functions are integrated exactly inside the straddling cell;
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
    # The straddling cell is the one containing b_x strictly inside.
    if not (edges[0] < b_x < edges[-1]):
        return out
    i = int(np.searchsorted(edges, b_x, side="right")) - 1
    i = min(max(i, 0), x_vec.shape[0] - 1)
    e_lo, e_hi = float(edges[i]), float(edges[i + 1])
    if not (e_lo < b_x < e_hi):
        # threshold exactly on a cell face: fractions are already exact
        return out

    # Breach part of the straddling cell.
    lo, hi = (b_x, e_hi) if breach_up else (e_lo, b_x)
    # Split at the node itself when it lies inside the sub-interval, so each
    # trapezoid spans a single linear segment of the interpolant.
    points = [lo]
    xi = float(x_vec[i])
    if lo < xi < hi:
        points.append(xi)
    points.append(hi)

    integral = np.zeros(jump.shape[1:] if jump.ndim > 1 else (), dtype=float)
    for a, b in zip(points[:-1], points[1:]):
        va = _interp_columns(x_vec, jump, a)
        vb = _interp_columns(x_vec, jump, b)
        integral = integral + 0.5 * (b - a) * (va + vb)

    out[i] = integral / (e_hi - e_lo)
    return out
