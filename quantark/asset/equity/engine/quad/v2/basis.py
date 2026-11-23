"""Uniform Gauss cells and local polynomial evaluation."""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from quantark.util.exceptions import NumericalError


class ResourceLimitError(NumericalError):
    """The mesh the shortest event interval needs exceeds ``max_nodes`` (the caller names the interval)."""


@lru_cache(maxsize=32)
def gauss(order):
    x, w = np.polynomial.legendre.leggauss(order)
    x.flags.writeable = w.flags.writeable = False
    return x, w


def lagrange(theta, t):
    t = np.asarray(t)
    out = np.ones((len(theta),) + t.shape)
    for q in range(len(theta)):
        for p in range(len(theta)):
            if p != q:
                out[q] *= (t - theta[p]) / (theta[q] - theta[p])
    return out


@lru_cache(maxsize=16)
def _barycentric_weights(theta):
    theta = np.array(theta)
    return np.array(
        [1.0 / np.prod(theta[q] - np.delete(theta, q)) for q in range(len(theta))]
    )


def interpolation_weights(theta, t):
    """The identical cell polynomial in O(order) barycentric form."""
    t = np.asarray(t)
    shape = (len(theta),) + (1,) * t.ndim
    difference = t[None] - theta.reshape(shape)
    exact = np.abs(difference) < 1e-14
    weights = _barycentric_weights(tuple(theta)).reshape(shape) / np.where(
        exact, 1.0, difference
    )
    normalized = weights / weights.sum(axis=0)
    return np.where(exact.any(axis=0)[None], exact.astype(float), normalized)


@dataclass(frozen=True)
class Mesh:
    lo: float
    h: float
    cells: int
    order: int

    @property
    def theta(self):
        return (gauss(self.order)[0] + 1) / 2

    @property
    def hi(self):
        return self.lo + self.cells * self.h

    @property
    def nodes(self):
        return self.lo + self.h * (np.arange(self.cells)[:, None] + self.theta)

    @property
    def edges(self):
        return self.lo + self.h * np.arange(self.cells + 1)

    def interpolate(self, values, y, derivative=0):
        """Values have shape (channels, cells, order); y has any shape.

        Only smooth branches are interpolated. Edge-cell extension is used
        outside the computational domain; the domain-expansion ladder must
        qualify its effect on queries within the declared range.
        """
        y = np.asarray(y)
        u = (y - self.lo) / self.h
        k = np.clip(np.floor(u).astype(int), 0, self.cells - 1)
        t = np.clip(u - k, 0, 1)
        if derivative:
            # Polynomial derivatives are only used by deterministic readout.
            basis = []
            for q, x in enumerate(self.theta):
                p = np.polynomial.Polynomial.fromroots(np.delete(self.theta, q))
                p /= p(x)
                basis.append(p.deriv(derivative)(t) / self.h**derivative)
            weights = np.asarray(basis)
        else:
            weights = interpolation_weights(self.theta, t)
        return np.einsum("c...q,q...->c...", values[:, k, :], weights)


def make_mesh(query_logs, means, variances, params, align=None):
    positive = np.asarray(variances[1:])
    positive = positive[positive > 0]
    if positive.size == 0:
        return None
    h = np.sqrt(positive.min()) / params.cells_per_sd
    drift = np.r_[0.0, np.cumsum(means)]
    # Include the full drift path, not just volatility around today's spot.
    reach = params.domain_sd * np.sqrt(np.sum(variances))
    lo = float(np.min(query_logs) + np.min(drift) - reach - 2 * h)
    hi = float(np.max(query_logs) + np.max(drift) + reach + 2 * h)
    if align is not None and lo < align < hi:
        # The continuous surviving branch has a boundary kink where it joins
        # the already-KI branch. Keep that kink on a cell edge.
        lo = align - np.ceil((align - lo) / h) * h
        hi = lo + np.ceil((hi - lo) / h) * h
    required = max(8, int(np.ceil((hi - lo) / h)))
    if required * params.order > params.max_nodes:
        raise ResourceLimitError(
            f"QUAD V2 resource limit: required {required * params.order} nodes "
            f"for minimum interior variance {positive.min():.6g}; max_nodes={params.max_nodes}"
        )
    if not np.isfinite([lo, hi]).all() or lo < -650 or hi > 650:
        raise NumericalError("QUAD V2 log-domain exceeds finite exponential range")
    return Mesh(
        lo, h if align is not None else (hi - lo) / required, required, params.order
    )


#: the most value-jump breaks a mesh can carry on its cell edges at the resolved cell width (one sets the origin, a
#: second the width); a third would need a common divisor of two arbitrary distances
MAX_ALIGNED_BREAKS = 2


def local_mesh(query_logs, means, variances, params, variance, breaks):
    """``make_mesh``'s domain with cells sized from ``variance`` (the smallest interval the mesh resolves) and edges on
    ``breaks``, the value-jump breaks (log spot) of the events that end the intervals priced below it.

    The cells resolve every interval of variance >= ``variance``; a shorter one is integrated by the exact local kernel
    (``Operator.kernel``), and its end event's jumps sit on cell edges, so the cell polynomials represent that event's
    pieces exactly (sub-resolution run S2, "Ea": unaligned, the projection of a jump smeared over a width s << h is
    what fails, up to 1.1e-4 rel). One break inside the domain moves the origin onto it; two also shrink the width to
    ``span / ceil(span / h)``. Returns ``None`` when the cells exceed ``max_nodes``.

    Raises:
        ResourceLimitError: more than ``MAX_ALIGNED_BREAKS`` breaks inside the domain.
    """
    h = float(np.sqrt(variance) / params.cells_per_sd)
    drift = np.r_[0.0, np.cumsum(means)]
    reach = params.domain_sd * np.sqrt(np.sum(variances))
    lo = float(np.min(query_logs) + np.min(drift) - reach - 2 * h)
    hi = float(np.max(query_logs) + np.max(drift) + reach + 2 * h)
    inside = sorted({float(b) for b in breaks if lo < b < hi})
    if len(inside) > MAX_ALIGNED_BREAKS:
        raise ResourceLimitError(
            f"the events ending them carry {len(inside)} distinct value-jump breaks "
            f"(log spot {', '.join(f'{b:.6g}' for b in inside)}); a mesh aligns at most {MAX_ALIGNED_BREAKS} "
            "jump breaks"
        )
    if len(inside) == 2:
        span = inside[1] - inside[0]
        h = span / math.ceil(span / h)
    if inside:
        lo = inside[0] - math.ceil((inside[0] - lo) / h) * h
    cells = max(8, int(math.ceil((hi - lo) / h)))
    if cells * params.order > params.max_nodes:
        return None
    if not np.isfinite([lo, hi]).all() or lo < -650 or lo + cells * h > 650:
        raise NumericalError("QUAD V2 log-domain exceeds finite exponential range")
    return Mesh(lo, h, cells, params.order)
