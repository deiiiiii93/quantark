"""Constant-coefficient continuous KI with full survival-kernel derivatives."""
import numpy as np
from quantark.param import FlatRateCurve, FlatVolSurface, ContinuousDividendYield
from quantark.param.rrf.rate_curve import LogLinearRateCurve, ParallelShiftRateCurve
from quantark.param.vol.vol_surface import (
    TermStructureVolSurface,
    ParallelShiftVolSurface,
)
from quantark.param.div.dividend_yield import NoDividend, ParallelShiftDividendYield
from quantark.util.exceptions import NumericalError, ValidationError
from .basis import gauss
from .contract import Function, Piece, Source
from .gaussian import readout


def continuous_time_grid(env, times, params):
    """Split at known coefficient knots; arbitrary terms require explicit approximation.

    Log-linear discount factors and piecewise-linear total variance imply
    constant coefficients between knots. Linear zero-yield curves generally
    do not. Merely sampling two equal endpoints is never an exactness test.
    """
    rate, div, vol = env.rate_curve, env.div_yield, env.vol_surface
    while type(rate) is ParallelShiftRateCurve:
        rate = rate.base_curve
    while type(div) is ParallelShiftDividendYield:
        div = div.base
    unwrapped_vol = vol
    while type(unwrapped_vol) is ParallelShiftVolSurface:
        unwrapped_vol = unwrapped_vol.base
    # A shift of flat volatility is still constant. A shift applied to the
    # interpolated term vol need not preserve piecewise-linear total variance.
    exact_vol = (
        type(vol) in {FlatVolSurface, TermStructureVolSurface}
        or type(unwrapped_vol) is FlatVolSurface
    )
    exact = (
        type(rate) in {FlatRateCurve, LogLinearRateCurve}
        and type(div) in {ContinuousDividendYield, NoDividend}
        and exact_vol
    )
    if not exact and params.continuous_term_structure != "piecewise_constant":
        raise NotImplementedError(
            "continuous KI with these term curves requires continuous_term_structure='piecewise_constant' and a time-refinement qualification"
        )
    if not times or any(not np.isfinite(t) or t < 0 for t in times):
        raise ValidationError("event times must be finite and nonnegative")
    maturity = max(times)
    nodes = set(times) | {0.0}
    for curve in (rate, div, unwrapped_vol):
        nodes.update(
            float(t) for t in getattr(curve, "times", ()) if 0 < float(t) < maturity
        )
        nodes.update(
            float(p[0])
            for p in getattr(curve, "pillars", ())
            if 0 < float(p[0]) < maturity
        )
    if len(nodes) - (0.0 not in times) > params.max_events:
        raise NumericalError("QUAD V2 event count exceeds max_events")
    if not exact:
        base = sorted(nodes)
        subdivisions = []
        predicted = len(nodes) - (0.0 not in times)
        for a, b in zip(base[:-1], base[1:]):
            count = max(1, int(np.ceil((b - a) * params.continuous_steps_per_year)))
            predicted += count - 1
            if predicted > params.max_events:
                raise NumericalError(
                    f"QUAD V2 continuous subdivision requires at least {predicted} events; max_events={params.max_events}"
                )
            subdivisions.append((a, b, count))
        for a, b, count in subdivisions:
            nodes.update(a + (b - a) * j / count for j in range(1, count))
    nodes.discard(0.0)
    if 0.0 in times:
        nodes.add(0.0)
    return (
        sorted(nodes),
        "piecewise_constant_exact" if exact else "piecewise_constant_approximation",
    )


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
