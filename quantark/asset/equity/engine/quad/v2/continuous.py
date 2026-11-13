"""Constant-coefficient continuous KI with full survival-kernel derivatives."""
import numpy as np
from quantark.util.exceptions import NumericalError
from .basis import gauss
from .contract import Function, Piece, Source
from .gaussian import readout


# All native engines use the same coefficient-model classification and knots.
from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid


def _difference(alive, hit, barrier, reverse):
    cuts = sorted({-np.inf, np.inf, barrier, *alive.breaks, *hit.breaks})
    pieces = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if (reverse and lo >= barrier) or (not reverse and hi <= barrier):
            continue
        a = next(p for p in alive.pieces if p.lo <= lo and p.hi >= hi)
        b = next(p for p in hit.pieces if p.lo <= lo and p.hi >= hi)
        pieces.append(
            Piece(
                lo,
                hi,
                a.cash - b.cash,
                a.asset - b.asset,
                a.sources
                + tuple(Source(s.grid, -s.weight, s.shift) for s in b.sources),
            )
        )
    return Function(tuple(pieces))


def continuous_readout(
    alive, hit, x, mean, variance, discount, barrier, reverse, params
):
    """D E[B + 1(no hit)*(A-B)], including derivatives of the hit factor.

    Reflection is fast and accurate at ordinary drift. A stable direct
    survival integral handles cases where the reflection multiplier would
    overflow; no barrier shift or discrete-monitoring approximation is used.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if variance == 0:
        shifted = (alive.shifted(mean, discount), hit.shifted(mean, discount))
        mapped = deterministic_continuous(shifted, ((0, 1),), barrier, reverse, mean)[0]
        return readout(mapped, x, 0.0, 0.0, 1.0, params)
    result = readout(hit, x, mean, variance, discount, params)
    safe = x < barrier if reverse else x > barrier
    at = np.abs(x - barrier) <= 8 * np.spacing(max(1.0, abs(barrier)))
    result[1:, :, at] = np.nan
    if not np.any(safe):
        return result
    diff = _difference(alive, hit, barrier, reverse)
    xx = x[safe]
    c = -2 * mean / variance
    log_weight = c * (xx - barrier)
    ordinary = np.abs(log_weight) < 500
    addition = np.zeros((3, alive.channels, len(xx)))
    if np.any(ordinary):
        points = xx[ordinary]
        forward = readout(diff, points, mean, variance, 1.0, params)
        image = readout(diff, 2 * barrier - points, mean, variance, 1.0, params)
        w = np.exp(log_weight[ordinary])
        addition[0, :, ordinary] = (forward[0] - w * image[0]).T
        addition[1, :, ordinary] = (forward[1] - w * (c * image[0] - image[1])).T
        addition[2, :, ordinary] = (
            forward[2] - w * (c * c * image[0] - 2 * c * image[1] + image[2])
        ).T
    for i in np.flatnonzero(~ordinary):
        addition[:, :, i] = _stable_survival(
            diff, xx[i], mean, variance, barrier, params
        )
    result[:, :, safe] += discount * addition
    result[1:, :, at] = np.nan
    return result


def _stable_survival(function, x, mean, variance, barrier, params):
    sd = np.sqrt(variance)
    gx, gw = gauss(params.readout_order)
    result = np.zeros((3, function.channels))
    for p in function.pieces:
        left = max(p.lo, x + mean - params.tail_sd * sd)
        right = min(p.hi, x + mean + params.tail_sd * sd)
        if left >= right:
            continue
        cuts = [left, right]
        for s in p.sources:
            edges = s.grid.mesh.edges - s.shift
            cuts.extend(edges[(edges > left) & (edges < right)])
        zcuts = x + mean + np.arange(-params.tail_sd, params.tail_sd + 1, 1.0) * sd
        cuts.extend(zcuts[(zcuts > left) & (zcuts < right)])
        cuts = np.unique(cuts)
        if (len(cuts) - 1) * len(gx) * 8 * (
            function.channels + 12
        ) > params.max_work_bytes:
            raise NumericalError(
                "QUAD V2 survival integration exceeds workspace budget"
            )
        a, b = (cuts[:-1] - x - mean) / sd, (cuts[1:] - x - mean) / sd
        half, mid = (b - a) / 2, (b + a) / 2
        z = mid[:, None] + half[:, None] * gx
        y = x + mean + sd * z
        exponent = -2 * (x - barrier) * (y - barrier) / variance
        hit = np.exp(exponent)
        survive = -np.expm1(exponent)
        ax = -2 * (y - barrier) / variance
        score = z / sd
        kernels = (
            survive,
            score * survive - hit * ax,
            (score * score - 1 / variance) * survive
            - 2 * score * hit * ax
            - hit * ax * ax,
        )
        weights = half[:, None] * gw * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        values = p.value(y)
        for d, kernel in enumerate(kernels):
            result[d] += np.einsum("cpq,pq->c", values, weights * kernel)
    return result


def deterministic_continuous(states, pairs, barrier, reverse, mean):
    result = list(states)
    # Deterministic log paths are straight lines for constant coefficients;
    # hitting is determined by the two interval endpoints.
    boundary = min(barrier, barrier - mean) if reverse else max(barrier, barrier - mean)
    for alive, hit in pairs:
        pieces = []
        for source, lo, hi in (
            ((states[alive], -np.inf, boundary), (states[hit], boundary, np.inf))
            if reverse
            else ((states[hit], -np.inf, boundary), (states[alive], boundary, np.inf))
        ):
            for p in source.pieces:
                a, b = max(lo, p.lo), min(hi, p.hi)
                if a < b:
                    pieces.append(Piece(a, b, p.cash, p.asset, p.sources))
        p = next(p for p in states[hit].pieces if p.lo <= boundary < p.hi)
        result[alive] = Function(tuple(pieces), ((boundary, p),))
    return tuple(result)


def continuous_grid_readout(alive, hit, mean, variance, discount, barrier, reverse, op, rolled_hit=None):
    """Value-only reflection on the barrier-aligned Gaussian grid.

    Reflection permutes Gaussian nodes exactly. Plain native convolutions can
    therefore supply both terms; only mirrored queries outside the mesh and
    extreme reflection multipliers need the existing pointwise integral.
    """
    mesh, params = op.mesh, op.params
    x = mesh.nodes.ravel()
    coordinate = (2*barrier - 2*mesh.lo) / mesh.h
    aligned = abs(coordinate - round(coordinate)) <= 1e-9
    shifted = any(s.shift != 0 or s.grid.mesh != mesh for f in (alive, hit) for p in f.pieces for s in p.sources)
    if params.continuous_grid_backend == 'pointwise' or not aligned or shifted:
        return continuous_readout(alive, hit, x, mean, variance, discount, barrier, reverse, params)[0].reshape(alive.channels, mesh.cells, mesh.order)
    base = op.integrate(hit, mean, variance, discount) if rolled_hit is None else rolled_hit
    result = base.reshape(alive.channels, -1).copy()
    safe = x < barrier if reverse else x > barrier
    indices = np.flatnonzero(safe)
    if not len(indices):
        return result.reshape(alive.channels, mesh.cells, mesh.order)
    diff = _difference(alive, hit, barrier, reverse)
    forward = op.integrate(diff, mean, variance, 1.0)
    cells = indices // mesh.order
    nodes = indices % mesh.order
    mirror_cells = int(round(coordinate)) - cells - 1
    mirror_nodes = mesh.order - nodes - 1
    inside = (mirror_cells >= 0) & (mirror_cells < mesh.cells)
    image = np.zeros((alive.channels, len(indices)))
    if inside.any():
        image[:, inside] = forward[:, mirror_cells[inside], mirror_nodes[inside]]
    if (~inside).any():
        image[:, ~inside] = readout(diff, 2*barrier-x[indices[~inside]], mean, variance, 1.0, params)[0]
    log_weight = -2*mean/variance * (x[indices]-barrier)
    ordinary = np.abs(log_weight) < 500
    result[:, indices[ordinary]] += discount * (
        forward.reshape(alive.channels, -1)[:, indices[ordinary]] - np.exp(log_weight[ordinary])*image[:, ordinary]
    )
    if (~ordinary).any():
        result[:, indices[~ordinary]] = continuous_readout(
            alive, hit, x[indices[~ordinary]], mean, variance, discount, barrier, reverse, params,
        )[0]
    op.backends_used.add('continuous_reflection_grid')
    return result.reshape(alive.channels, mesh.cells, mesh.order)
