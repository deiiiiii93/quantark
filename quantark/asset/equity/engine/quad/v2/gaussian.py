"""Gaussian cash/asset moments and the authoritative point readout."""
import numpy as np
from scipy.special import ndtr
from quantark.util.exceptions import NumericalError

from .basis import gauss, interpolation_weights

_INV_SQRT_2PI = 1 / np.sqrt(2 * np.pi)


def _cdf_difference(lo, hi):
    return np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))


def affine_moments(x, mean, variance, lo, hi, power=0):
    """Integral of exp(power*y), and two x derivatives, over fixed limits."""
    x = np.asarray(x, dtype=float)
    sd = np.sqrt(variance)
    a = (lo - x - mean - power * variance) / sd
    b = (hi - x - mean - power * variance) / sd
    pa = _INV_SQRT_2PI * np.exp(-a * a / 2)
    pb = _INV_SQRT_2PI * np.exp(-b * b / 2)
    mass = _cdf_difference(a, b)
    first = (pa - pb) / sd
    aa = np.where(np.isfinite(a), a, 0.0)
    bb = np.where(np.isfinite(b), b, 0.0)
    second = (aa * pa - bb * pb) / variance
    factor = np.exp(power * (x + mean) + 0.5 * power * power * variance)
    return np.array(
        (
            factor * mass,
            factor * (power * mass + first),
            factor * (power * power * mass + 2 * power * first + second),
        )
    )


def _source_readout(
    source, x, mean, variance, lo, hi, order, tail_sd, max_work_bytes=None
):
    """Integrate one smooth cell field. Cell and event edges are fixed limits.

    Integrate in standardized Gaussian coordinates. Local cash/asset removal
    limits cancellation in the density's second derivative at tiny variance.
    """
    grid, shift = source.grid, source.shift
    mesh, values = grid.mesh, grid.values
    gx, gw = gauss(order)
    sd = np.sqrt(variance)
    panels = min(mesh.cells, int(np.ceil(2 * tail_sd * sd / mesh.h))) + int(tail_sd) + 5
    if (
        max_work_bytes is not None
        and panels * order * mesh.order * 8 * (len(values) + 5) > max_work_bytes
    ):
        raise NumericalError("QUAD V2 point integration exceeds workspace budget")
    result = np.zeros((3, len(values), len(x)))
    for j, xx in enumerate(x):
        center = xx + mean
        left, right = max(lo, center - tail_sd * sd), min(hi, center + tail_sd * sd)
        if left >= right:
            continue
        k = int(
            np.clip(np.floor((center + shift - mesh.lo) / mesh.h), 0, mesh.cells - 1)
        )
        # Fit the local affine part in spot from two nodes in the same cell.
        spots = np.exp(mesh.lo + (k + mesh.theta) * mesh.h - shift)
        slope = (values[:, k, -1] - values[:, k, 0]) / (spots[-1] - spots[0])
        cash = values[:, k, 0] - slope * spots[0]
        base0 = affine_moments(np.array([xx]), mean, variance, lo, hi, 0)[:, 0]
        base1 = affine_moments(np.array([xx]), mean, variance, lo, hi, 1)[:, 0]
        result[:, :, j] = base0[:, None] * cash + base1[:, None] * slope

        i0 = max(0, int(np.floor((left + shift - mesh.lo) / mesh.h)) + 1)
        i1 = min(mesh.cells, int(np.ceil((right + shift - mesh.lo) / mesh.h)))
        internal = mesh.lo + np.arange(i0, i1) * mesh.h - shift
        internal = internal[(internal > left) & (internal < right)]
        # A very short first interval can live in a coarse cell. Subdivide
        # wide Gaussian panels as well, independently of that coarse mesh.
        zcuts = np.arange(-tail_sd, tail_sd + 1, 2.0) * sd + center
        cuts = np.unique(
            np.r_[left, internal, zcuts[(zcuts > left) & (zcuts < right)], right]
        )
        a, b = (cuts[:-1] - center) / sd, (cuts[1:] - center) / sd
        half, mid = (b - a) / 2, (b + a) / 2
        z = mid[:, None] + half[:, None] * gx
        y = center + sd * z
        u = (y + shift - mesh.lo) / mesh.h
        cells = np.clip(np.floor(u).astype(int), 0, mesh.cells - 1)
        t = np.clip(u - cells, 0, 1)
        weights = interpolation_weights(mesh.theta, t)
        node_spots = np.exp(mesh.lo + (cells[..., None] + mesh.theta) * mesh.h - shift)
        residual_nodes = (
            values[:, cells, :]
            - cash[:, None, None, None]
            - slope[:, None, None, None] * node_spots
        )
        residual = np.einsum("c...q,q...->c...", residual_nodes, weights)
        # The source is a polynomial in log spot, so subtract the interpolant
        # of the affine component and restore its exact exponential residual.
        interpolated_spots = np.einsum("...q,q...->...", node_spots, weights)
        residual += slope[:, None, None] * (interpolated_spots - np.exp(y))
        density_weights = half[:, None] * gw * _INV_SQRT_2PI * np.exp(-z * z / 2)
        for d, score in enumerate((np.ones_like(z), z / sd, (z * z - 1) / variance)):
            result[d, :, j] += np.einsum("cpq,pq->c", residual, density_weights * score)
    return result * source.weight


def _source_readout_batch(source, x, mean, variance, lo, hi, params):
    """Batch fixed-cell Gaussian kernels; narrow first intervals use z panels.

    Interpolate each cell's values once per integration node, then apply all
    three density scores together. No price/Greek grid gradients are used.
    """
    mesh, values, shift = source.grid.mesh, source.grid.values, source.shift
    if source.weight == 0:
        return np.zeros((3, len(values), len(x)))
    sd = np.sqrt(variance)
    if mesh.h > sd * 0.75:
        return _source_readout(
            source,
            x,
            mean,
            variance,
            lo,
            hi,
            params.readout_order,
            params.tail_sd,
            params.max_work_bytes,
        )
    band = int(np.ceil(params.tail_sd * sd / mesh.h)) + 2
    order = 2 * mesh.order + 8
    bytes_per_query = max(
        1, (2 * band + 1) * order * mesh.order * 8 * (len(values) + 5)
    )
    if bytes_per_query > params.max_work_bytes:
        raise NumericalError("QUAD V2 vector integration exceeds workspace budget")
    block = max(1, min(64, params.max_work_bytes // bytes_per_query))
    result = np.zeros((3, len(values), len(x)))
    gx, gw = gauss(order)
    for begin in range(0, len(x), block):
        indices = np.arange(begin, min(begin + block, len(x)))
        center = x[indices] + mean
        cell0 = np.floor((center + shift - mesh.lo) / mesh.h).astype(int)
        inside = (cell0 - band >= 0) & (cell0 + band < mesh.cells)
        if np.any(~inside):
            result[:, :, indices[~inside]] = (
                _source_readout(
                    source,
                    x[indices[~inside]],
                    mean,
                    variance,
                    lo,
                    hi,
                    params.readout_order,
                    params.tail_sd,
                    params.max_work_bytes,
                )
                / source.weight
            )
        indices, center, cell0 = indices[inside], center[inside], cell0[inside]
        if not len(indices):
            continue
        k = cell0
        local_spots = np.exp(mesh.lo + (k[:, None] + mesh.theta) * mesh.h - shift)
        slope = (values[:, k, -1] - values[:, k, 0]) / (
            local_spots[:, -1] - local_spots[:, 0]
        )
        cash = values[:, k, 0] - slope * local_spots[:, 0]
        base0 = affine_moments(x[indices], mean, variance, lo, hi, 0)
        base1 = affine_moments(x[indices], mean, variance, lo, hi, 1)
        total = base0[:, None, :] * cash[None] + base1[:, None, :] * slope[None]
        cells = cell0[:, None] + np.arange(-band, band + 1)
        edges = mesh.lo + cells * mesh.h - shift
        a = np.clip((lo - edges) / mesh.h, 0, 1)
        b = np.clip((hi - edges) / mesh.h, 0, 1)
        half, mid = (b - a) / 2, (b + a) / 2
        t = mid[:, :, None] + half[:, :, None] * gx
        y = edges[:, :, None] + mesh.h * t
        z = (y - center[:, None, None]) / sd
        weights = interpolation_weights(mesh.theta, t)
        node_spots = np.exp(mesh.lo + (cells[..., None] + mesh.theta) * mesh.h - shift)
        residual_nodes = (
            values[:, cells, :]
            - cash[:, :, None, None]
            - slope[:, :, None, None] * node_spots
        )
        residual = np.einsum("cbdj,jbdq->cbdq", residual_nodes, weights)
        interpolated_spots = np.einsum("bdj,jbdq->bdq", node_spots, weights)
        residual += slope[:, :, None, None] * (interpolated_spots - np.exp(y))
        dw = (mesh.h / sd) * half[:, :, None] * gw * _INV_SQRT_2PI * np.exp(-z * z / 2)
        for d, score in enumerate((np.ones_like(z), z / sd, (z * z - 1) / variance)):
            total[d] += np.einsum("cbpq,bpq->cb", residual, dw * score)
        result[:, :, indices] = total
    return result * source.weight


def readout(function, x, mean, variance, discount, params):
    """Return (PV, d/dlogS, d²/dlogS²), shape (3, channels, queries)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if variance == 0:
        y = x + mean
        result = np.array([function.evaluate(y, d) for d in range(3)]) * discount
        for b in function.breaks:
            at = np.abs(y - b) <= 8 * np.spacing(max(1.0, abs(b)))
            if not np.any(at):
                continue
            left = next((p for p in function.pieces if p.lo < b <= p.hi), None)
            right = next((p for p in function.pieces if p.lo <= b < p.hi), None)
            if left is None or right is None:
                result[1:, :, at] = np.nan
                continue
            l = np.array([left.value(np.array([b]), d)[:, 0] for d in range(3)])
            r = np.array([right.value(np.array([b]), d)[:, 0] for d in range(3)])
            # Preserve PV equality ownership, but use the common derivative
            # only when the one-sided traces agree. Irrelevant breakpoints do
            # not turn a smooth constant/asset payoff into an undefined Greek.
            for d in (1, 2):
                defined = np.all(
                    np.isclose(l[: d + 1], r[: d + 1], rtol=1e-11, atol=1e-11), axis=0
                )
                result[d][:, at] = np.where(defined, l[d] * discount, np.nan)[:, None]
        return result
    result = np.zeros((3, function.channels, len(x)))
    for p in function.pieces:
        c = affine_moments(x, mean, variance, p.lo, p.hi, 0)
        a = affine_moments(x, mean, variance, p.lo, p.hi, 1)
        result += (
            c[:, None, :] * p.cash[None, :, None]
            + a[:, None, :] * p.asset[None, :, None]
        )
        for source in p.sources:
            if len(x) > 1:
                result += _source_readout_batch(
                    source, x, mean, variance, p.lo, p.hi, params
                )
            else:
                result += _source_readout(
                    source,
                    x,
                    mean,
                    variance,
                    p.lo,
                    p.hi,
                    params.readout_order,
                    params.tail_sd,
                    params.max_work_bytes,
                )
    return discount * result
