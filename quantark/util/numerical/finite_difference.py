"""Second-order finite-difference stencils on non-uniform 1D grids (last-axis batched).

Promoted from the Dupire local-vol builder (quantark.volmodels.localvol.dupire) so any
module differencing smooth data on a non-uniform grid can reuse them.
"""

from __future__ import annotations

import numpy as np


def fd1_nonuniform(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """First derivative along last axis.

    Non-uniform central differences on the interior and 3-point Lagrange one-sided
    formulas at both ends (second-order; exact for quadratics).
    """
    d = np.zeros_like(values)
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    d[..., 1:-1] = (
        -values[..., :-2] * dxp / (dxm * (dxm + dxp))
        + values[..., 1:-1] * (dxp - dxm) / (dxm * dxp)
        + values[..., 2:] * dxm / (dxp * (dxm + dxp))
    )
    # Left end: Lagrange derivative through (x0, x1, x2) evaluated at x0.
    a, b, c = x[0], x[1], x[2]
    d[..., 0] = (
        values[..., 0] * ((a - b) + (a - c)) / ((a - b) * (a - c))
        + values[..., 1] * (a - c) / ((b - a) * (b - c))
        + values[..., 2] * (a - b) / ((c - a) * (c - b))
    )
    # Right end: Lagrange derivative through (x[-3], x[-2], x[-1]) evaluated at x[-1].
    a, b, c = x[-3], x[-2], x[-1]
    d[..., -1] = (
        values[..., -3] * (c - b) / ((a - b) * (a - c))
        + values[..., -2] * (c - a) / ((b - a) * (b - c))
        + values[..., -1] * ((c - a) + (c - b)) / ((c - a) * (c - b))
    )
    return d


def fd2_nonuniform(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Second derivative along last axis.

    Non-uniform central differences on the interior; 3-point Lagrange second
    derivative (constant across the stencil, exact for quadratics) at both ends.
    """
    d2 = np.zeros_like(values)
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    denom = 0.5 * dxm * dxp * (dxm + dxp)
    d2[..., 1:-1] = (
        values[..., :-2] * dxp - values[..., 1:-1] * (dxm + dxp) + values[..., 2:] * dxm
    ) / denom
    a, b, c = x[0], x[1], x[2]
    d2[..., 0] = (
        values[..., 0] * 2.0 / ((a - b) * (a - c))
        + values[..., 1] * 2.0 / ((b - a) * (b - c))
        + values[..., 2] * 2.0 / ((c - a) * (c - b))
    )
    a, b, c = x[-3], x[-2], x[-1]
    d2[..., -1] = (
        values[..., -3] * 2.0 / ((a - b) * (a - c))
        + values[..., -2] * 2.0 / ((b - a) * (b - c))
        + values[..., -1] * 2.0 / ((c - a) * (c - b))
    )
    return d2


def fd1_interior_coeffs(x: np.ndarray):
    """Interior non-uniform first-derivative stencil weights (wm, w0, wp), each (N-2,).

    ``d f/dx |_j ≈ wm[j-1] f[j-1] + w0[j-1] f[j] + wp[j-1] f[j+1]`` for j=1..N-2.
    Reduces to the uniform (-1/2h, 0, 1/2h) at equal spacing; exact for quadratics.
    """
    x = np.asarray(x, dtype=float)
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    wm = -dxp / (dxm * (dxm + dxp))
    w0 = (dxp - dxm) / (dxm * dxp)
    wp = dxm / (dxp * (dxm + dxp))
    return wm, w0, wp


def fd2_interior_coeffs(x: np.ndarray):
    """Interior non-uniform second-derivative stencil weights (wm, w0, wp), each (N-2,).

    Reduces to the uniform (1/h^2, -2/h^2, 1/h^2) at equal spacing; exact for quadratics.
    """
    x = np.asarray(x, dtype=float)
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    denom = 0.5 * dxm * dxp * (dxm + dxp)
    wm = dxp / denom
    w0 = -(dxm + dxp) / denom
    wp = dxm / denom
    return wm, w0, wp
