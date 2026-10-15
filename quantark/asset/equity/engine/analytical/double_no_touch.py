"""The probability that a geometric Brownian motion never leaves an open corridor before maturity (spec 3C
§5.8, decision 28), with a bound on its own error.

The method of images in log space, centred on the spot: the barriers are at +d_up = ln(U/S) and -d_lo = -ln(S/L),
s = vol sqrt(T), m = (r - q - vol^2/2) T. An image source sits at h = 2 (p d_up + q d_lo) for integers p, q of
one sign, with weight +1 when q = p and -1 when q = p - 1:

    P = sum of +/- T(h),   T(h) = exp(h m / s^2) [Phi((d_up - h - m)/s) - Phi((-d_lo - h - m)/s)]

Two large numbers are never added. For an endpoint w at distance g = w - h from the source,
h m / s^2 - (g - m)^2 / (2 s^2) = -[(w - m)^2 + Q] / (2 s^2) with Q = g^2 - w^2 >= 0 factored into same-sign
sums of d_up and d_lo, so with Phi(-z) = exp(-z^2/2) erfcx(z / sqrt 2) / 2 each tail piece is a product of
two factors in [0, 1]. The image count follows from a drift-free tail bound; tau = s^2 / ln(U/L)^2 >= 8 is a
proven zero. The bound charges the omitted images, 8 ulps per retained piece and the rounding of g - m, which
is the problem's own conditioning (the terminal distribution's distance to a barrier in its own widths).
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.special import erfcx

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import Tolerance, pow_elementwise

__all__ = ["double_no_touch", "double_no_touch_with_bound"]

_EPS = sys.float_info.epsilon
_SQRT2 = math.sqrt(2.0)
_TAU_ZERO = 8.0              # from here the survival is below 2 exp(-pi^2 tau / 2 + 1 / (2 tau)) < 2e-17
_LONG_TIME_BOUND = 2e-17
_TAIL = 1e-17                # the omitted images
_C_ROUND = 8.0               # ulps per retained piece
_C_INPUT = 4.0               # ulps on (|g| + (|r - q| + vol^2 / 2) T) for the argument g - m


def _validate(spot, maturity, rate, div, vol, lower, upper) -> None:
    named = (("spot", spot), ("maturity", maturity), ("rate", rate), ("div", div), ("vol", vol),
             ("lower", lower), ("upper", upper))
    for name, value in named:
        if not math.isfinite(float(value)):
            raise ValidationError(f"double_no_touch: {name} must be finite, got {value!r}")
    for name, value in (("spot", spot), ("vol", vol), ("lower", lower), ("upper", upper)):
        if not float(value) > 0.0:
            raise ValidationError(f"double_no_touch: {name} must be positive, got {value!r}")
    if not float(lower) < float(upper):
        raise ValidationError(f"double_no_touch: lower {lower!r} must be below upper {upper!r}")


def _evaluate(spot, maturity, rate, div, vol, lower, upper):
    """(probability, bound on its absolute error, last image index summed). No clipping, no refusal."""
    spot, maturity, rate, div, vol, lower, upper = (float(v) for v in (spot, maturity, rate, div, vol, lower, upper))
    if not lower < spot < upper:
        return 0.0, 0.0, 0
    if maturity <= 0.0:
        return 1.0, 0.0, 0
    # relatively accurate however close the spot is to a barrier (U - S and S - L are exact there)
    d_up, d_lo = math.log1p((upper - spot) / spot), -math.log1p((lower - spot) / spot)
    a = d_up + d_lo
    s = vol * math.sqrt(maturity)
    s2 = s * s
    if not s2 > 0.0:
        return math.nan, math.inf, 0
    if s2 >= _TAU_ZERO * a * a:
        return 0.0, _LONG_TIME_BOUND, 0
    m = (rate - div - 0.5 * vol * vol) * maturity
    m_scale = (abs(rate - div) + 0.5 * vol * vol) * maturity      # what m's rounding is relative to
    two_s2 = 2.0 * s2
    base_up, base_lo = (d_up - m) ** 2, (d_lo + m) ** 2             # (w - m)^2 at the two endpoints
    u0_up, u0_lo = abs(d_up - m) / s, abs(d_lo + m) / s
    ratio = 2.0 * a * a / s2                                        # 2 / tau
    total = mass = cond = 0.0
    k = 0
    while True:
        for p in ((k, -k) if k else (0,)):
            for q, sign in ((p, 1.0), (p - 1, -1.0)):
                src = p * d_up + q * d_lo                            # h / 2: one sign throughout
                g_up = (1 - 2 * p) * d_up - 2 * q * d_lo
                g_lo = -2 * p * d_up - (1 + 2 * q) * d_lo
                q_up = 4.0 * src * ((p - 1) * d_up + q * d_lo)       # g_up^2 - d_up^2 >= 0
                q_lo = 4.0 * (p * d_up + (q + 1) * d_lo) * src       # g_lo^2 - d_lo^2 >= 0
                u_up, u_lo = (g_up - m) / s, (g_lo - m) / s
                e_up, e_lo = -(base_up + q_up) / two_s2, -(base_lo + q_lo) / two_s2
                piece_up = 0.5 * math.exp(e_up) * float(erfcx(abs(u_up) / _SQRT2))
                piece_lo = 0.5 * math.exp(e_lo) * float(erfcx(abs(u_lo) / _SQRT2))
                if u_lo >= 0.0:                                      # the source lies below the corridor
                    term, full, e_full = piece_lo - piece_up, 0.0, 0.0
                elif u_up <= 0.0:                                    # above it
                    term, full, e_full = piece_up - piece_lo, 0.0, 0.0
                else:                                                # its mean is inside: the mass less two tails
                    e_full = 2.0 * src * m / s2                      # <= 0 there
                    full = math.exp(e_full)
                    term = full - piece_lo - piece_up
                total += sign * term
                mass += piece_up * (1.0 + abs(e_up)) + piece_lo * (1.0 + abs(e_lo)) + full * (1.0 + abs(e_full))
                cond += (piece_up * (u0_up + 1.0) * (abs(g_up) + m_scale)
                         + piece_lo * (u0_lo + 1.0) * (abs(g_lo) + m_scale))
        if k >= 1:                                                   # index 1 holds the barriers' own reflections
            lead = math.exp(-ratio * k * k)                          # every term of index k + 1 is below this
            if 4.0 * lead / -math.expm1(-ratio * (2 * k + 1)) <= _TAIL:
                break
        k += 1
    bound = _TAIL + _C_ROUND * _EPS * mass + _C_INPUT * _EPS * cond / s
    return total, bound, k


def _evaluate_array(spot, maturity, rate, div, vol, lower, upper):
    """``_evaluate`` elementwise on broadcast arrays of spots and maturities: (probabilities, bounds).

    The same operations in the same order per element; each element sums the images up to its own last index (the
    stopping test reads that element's tau), the others are masked.
    """
    spot, maturity = np.broadcast_arrays(np.asarray(spot, dtype=float), np.asarray(maturity, dtype=float))
    rate, div, vol, lower, upper = (float(v) for v in (rate, div, vol, lower, upper))
    total = np.zeros(spot.shape)
    bound = np.zeros(spot.shape)
    inside = (lower < spot) & (spot < upper)
    total[inside & (maturity <= 0.0)] = 1.0
    live = inside & (maturity > 0.0)
    if not live.any():
        return total, bound
    sp, mt = spot[live], maturity[live]
    d_up, d_lo = np.log1p((upper - sp) / sp), -np.log1p((lower - sp) / sp)
    a = d_up + d_lo
    s = vol * np.sqrt(mt)
    s2 = s * s
    degenerate = ~(s2 > 0.0)
    proven_zero = ~degenerate & (s2 >= _TAU_ZERO * a * a)
    summed = ~degenerate & ~proven_zero
    m = (rate - div - 0.5 * vol * vol) * mt
    m_scale = (abs(rate - div) + 0.5 * vol * vol) * mt
    two_s2 = 2.0 * s2
    base_up, base_lo = pow_elementwise(d_up - m, 2), pow_elementwise(d_lo + m, 2)
    acc = np.zeros(sp.shape)
    mass = np.zeros(sp.shape)
    cond = np.zeros(sp.shape)
    active = summed.copy()
    k = 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u0_up, u0_lo = np.abs(d_up - m) / s, np.abs(d_lo + m) / s
        ratio = 2.0 * a * a / s2
        while active.any():
            for p in ((k, -k) if k else (0,)):
                for q, sign in ((p, 1.0), (p - 1, -1.0)):
                    src = p * d_up + q * d_lo
                    g_up = (1 - 2 * p) * d_up - 2 * q * d_lo
                    g_lo = -2 * p * d_up - (1 + 2 * q) * d_lo
                    q_up = 4.0 * src * ((p - 1) * d_up + q * d_lo)
                    q_lo = 4.0 * (p * d_up + (q + 1) * d_lo) * src
                    u_up, u_lo = (g_up - m) / s, (g_lo - m) / s
                    e_up, e_lo = -(base_up + q_up) / two_s2, -(base_lo + q_lo) / two_s2
                    piece_up = 0.5 * np.exp(e_up) * erfcx(np.abs(u_up) / _SQRT2)
                    piece_lo = 0.5 * np.exp(e_lo) * erfcx(np.abs(u_lo) / _SQRT2)
                    below = u_lo >= 0.0                              # the source lies below the corridor
                    above = ~below & (u_up <= 0.0)                   # above it
                    middle = ~below & ~above                         # its mean is inside
                    e_full = np.where(middle, 2.0 * src * m / s2, 0.0)
                    full = np.where(middle, np.exp(e_full), 0.0)
                    term = np.where(below, piece_lo - piece_up,
                                    np.where(above, piece_up - piece_lo, full - piece_lo - piece_up))
                    acc = np.where(active, acc + sign * term, acc)
                    mass = np.where(active, mass + (piece_up * (1.0 + np.abs(e_up)) + piece_lo * (1.0 + np.abs(e_lo))
                                                    + full * (1.0 + np.abs(e_full))), mass)
                    cond = np.where(active, cond + (piece_up * (u0_up + 1.0) * (np.abs(g_up) + m_scale)
                                                    + piece_lo * (u0_lo + 1.0) * (np.abs(g_lo) + m_scale)), cond)
            if k >= 1:
                lead = np.exp(-ratio * k * k)
                active &= ~(4.0 * lead / -np.expm1(-ratio * (2 * k + 1)) <= _TAIL)
            k += 1
        summed_bound = _TAIL + _C_ROUND * _EPS * mass + _C_INPUT * _EPS * cond / s
    total[live] = np.where(summed, acc, np.where(proven_zero, 0.0, math.nan))
    bound[live] = np.where(summed, summed_bound, np.where(proven_zero, _LONG_TIME_BOUND, math.inf))
    return total, bound


def _validate_array(spot, maturity, rate, div, vol, lower, upper) -> None:
    """``_validate`` on arrays of spots and maturities."""
    for name, value in (("spot", spot), ("maturity", maturity)):
        if not np.all(np.isfinite(value)):
            raise ValidationError(f"double_no_touch: {name} must be finite, got {value!r}")
    if not np.all(np.asarray(spot) > 0.0):
        raise ValidationError(f"double_no_touch: spot must be positive, got {spot!r}")
    _validate(1.0, 0.0, rate, div, vol, lower, upper)    # the scalar inputs (a valid spot and maturity stand in)


def _double_no_touch_array(spot, maturity, rate, div, vol, lower, upper) -> np.ndarray:
    """``double_no_touch`` elementwise; any element refused refuses the call."""
    _validate_array(spot, maturity, rate, div, vol, lower, upper)
    value, bound = _evaluate_array(spot, maturity, rate, div, vol, lower, upper)
    unresolved = ~(bound <= Tolerance.ZERO)
    if unresolved.any():
        i = int(np.flatnonzero(unresolved)[0])
        raise NumericalError(
            f"double_no_touch: error bound {bound.flat[i]:.2e} exceeds Tolerance.ZERO ({Tolerance.ZERO:.0e}): the "
            f"inputs do not determine the probability (spot {np.broadcast_to(spot, bound.shape).flat[i]!r}, "
            f"T {np.broadcast_to(maturity, bound.shape).flat[i]!r}, r {rate!r}, q {div!r}, vol {vol!r}, "
            f"corridor ({lower!r}, {upper!r}))")
    outside = (value < -bound) | (value > 1.0 + bound)
    if outside.any():
        i = int(np.flatnonzero(outside)[0])
        raise NumericalError(
            f"double_no_touch: {value.flat[i]!r} lies outside [0, 1] by more than its error bound {bound.flat[i]:.2e}")
    return np.minimum(np.maximum(value, 0.0), 1.0)


def double_no_touch_with_bound(spot, maturity, rate, div, vol, lower, upper):
    """``(probability, bound on its absolute error)``; refuses only invalid input."""
    _validate(spot, maturity, rate, div, vol, lower, upper)
    value, bound, _ = _evaluate(spot, maturity, rate, div, vol, lower, upper)
    return value, bound


def double_no_touch(spot, maturity, rate, div, vol, lower, upper) -> float:
    """P[L < S_t < U for all t in [0, T]] under GBM with constant r, q, vol, to within ``Tolerance.ZERO``.

    Raises ``NumericalError`` when the inputs do not determine the probability to that tolerance (the bound
    exceeds it, spec 3C decision 28) or when the sum lies outside [0, 1] by more than its bound.

    ``spot`` and ``maturity`` may be arrays (broadcast together): the probabilities elementwise, each as the scalar
    call computes it; any element refused refuses the call.
    """
    if isinstance(spot, np.ndarray) or isinstance(maturity, np.ndarray):
        return _double_no_touch_array(spot, maturity, rate, div, vol, lower, upper)
    _validate(spot, maturity, rate, div, vol, lower, upper)
    value, bound, _ = _evaluate(spot, maturity, rate, div, vol, lower, upper)
    if not bound <= Tolerance.ZERO:
        raise NumericalError(
            f"double_no_touch: error bound {bound:.2e} exceeds Tolerance.ZERO ({Tolerance.ZERO:.0e}): the inputs "
            f"do not determine the probability (spot {spot!r}, T {maturity!r}, r {rate!r}, q {div!r}, vol {vol!r}, "
            f"corridor ({lower!r}, {upper!r}))")
    if value < -bound or value > 1.0 + bound:
        raise NumericalError(
            f"double_no_touch: {value!r} lies outside [0, 1] by more than its error bound {bound:.2e}")
    return min(max(value, 0.0), 1.0)
