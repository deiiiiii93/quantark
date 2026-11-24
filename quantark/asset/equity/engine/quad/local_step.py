"""The QUAD V1 local step: a diffusion step shorter than the lattice resolves (sub-resolution run, spec C1).

V1 samples each step's Gaussian kernel on a lattice sized for the whole maturity. A step whose standard deviation
``s`` spans less than a cell aliases that sampled kernel (``SnowballQuadEngine._kernel_mass_error_bound``). Such a
step is applied instead as a three-point stencil on the same lattice::

    V'_i = disc * (w_- V_{i-1} + w_0 V_i + w_+ V_{i+1}),
    w_-+ = s^2 / (2 h^2) -+ m / (2 h),   w_0 = 1 - s^2 / h^2,   m = 2 alpha tau,  s^2 = 2 tau,
    disc = exp(-(beta - alpha^2) tau)

which carries the step's mass, mean and variance exactly (``I + dt L`` with central differences). Beyond the lattice
the values are extended by the edge constants, as V1's tail term does. Its error per step is fourth-moment:
``(s^2 h^2 - 3 s^4) V'''' / 24``. The S1 demos measured an increment of at most 2.3e-8 on isolated steps; chains of
them accumulate, which the engine bounds (``SnowballQuadEngine._require_resolved_kernels``).

Rows of a stacked array are transformed independently, so stacked statistics columns equal the 1-D price bit for bit.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from quantark.util.exceptions import NumericalError

OFFSETS = (-1, 0, 1)


class UnderResolvedStepError(NumericalError):
    """QUAD V1 refuses a diffusion step its lattice does not resolve and its local step cannot price
    (``SnowballQuadEngine._require_resolved_kernels``). A finer lattice may resolve it: ``auto_converge`` moves to its
    next rung instead of failing."""


def local_step_weights(h: float, alpha: float, beta: float, tau: float) -> Tuple[float, float, float]:
    """``(w_-, w_0, w_+)`` of the step, discount included.

    Raises:
        UnderResolvedStepError: a weight is negative -- the step is not shorter than a cell, or its drift dominates
            its spread on this lattice; the stencil would not be a probability kernel.
    """
    a = 2.0 * tau / (h * h)
    b = 2.0 * alpha * tau / h
    disc = math.exp(-(beta - alpha * alpha) * tau)
    weights = (disc * (0.5 * a - 0.5 * b), disc * (1.0 - a), disc * (0.5 * a + 0.5 * b))
    if min(weights) < 0.0:
        raise UnderResolvedStepError(
            f"QUAD V1: the local step (s/h {math.sqrt(a):.3g}, m/h {b:.3g}) has a negative stencil weight: it is not "
            "a sub-resolution step on this lattice."
        )
    return weights


def _neighbours(n: int, offset: int) -> np.ndarray:
    return np.clip(np.arange(n) + offset, 0, n - 1)


def local_step_apply(values: np.ndarray, weights: Sequence[float],
                     coefficients: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """``sum_k w_k c_k(x_i) V(x_{i+k})`` along the last axis (``c_k`` per target node, default 1)."""
    values = np.asarray(values, dtype=float)
    n = values.shape[-1]
    out = np.zeros_like(values)
    for j, (offset, w) in enumerate(zip(OFFSETS, weights)):
        coeff = w if coefficients is None else w * coefficients[j]
        out = out + coeff * values[..., _neighbours(n, offset)]
    return out


def local_step_transpose(dual: np.ndarray, weights: Sequence[float],
                         coefficients: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """The transpose of :func:`local_step_apply` (the adjoint statistics pass)."""
    dual = np.asarray(dual, dtype=float)
    n = dual.shape[-1]
    out = np.zeros_like(dual)
    for j, (offset, w) in enumerate(zip(OFFSETS, weights)):
        coeff = w if coefficients is None else w * coefficients[j]
        index = _neighbours(n, offset)
        if dual.ndim == 1:
            np.add.at(out, index, coeff * dual)
        else:
            np.add.at(out, (slice(None), index), coeff * dual)
    return out


def local_bridge_hit(grid: np.ndarray, log_barrier: float, denom: float, is_reverse: bool) -> Tuple[np.ndarray, ...]:
    """Brownian-bridge touch probability between each target node and its stencil neighbour, as
    ``SnowballQuadEngine._bridge_kernels`` computes it (``denom = vol^2 dt``)."""
    grid = np.asarray(grid, dtype=float)
    n = grid.size
    out = []
    for offset in OFFSETS:
        source = grid[_neighbours(n, offset)]
        if is_reverse:
            d0, d1 = log_barrier - grid, log_barrier - source
        else:
            d0, d1 = grid - log_barrier, source - log_barrier
        safe = (d0 > 0.0) & (d1 > 0.0)
        exponent = np.clip(np.where(safe, -2.0 * d0 * d1 / denom, 0.0), -745.0, 0.0)
        out.append(np.where(safe, np.exp(exponent), 1.0))
    return tuple(out)
