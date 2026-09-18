"""Where the resolved context's market coefficients may change, and whether that is provable.

Two intraday claims need this. A closed-form first-passage formula is exact only
if ln S has a constant drift per unit variance up to expiry; a LOCAL theta is a
derivative only inside one piece of the coefficients. Both are statements about
whole INTERVALS, and no amount of pointwise sampling establishes either: a curve
evaluated at two ends says nothing about the law between them.

So every family this module admits declares two things: the instants where its
law may change, and the polynomial DEGREE of its cumulative quantity between
them -- total variance against calendar time, -ln DF, the cumulative dividend
carry q(t) t. A flat coefficient means that cumulative quantity is c * t. A
polynomial of degree d equals c * t on a piece iff it matches at d + 1 points of
it, so sampling at every declared break (plus the midpoint of every piece whose
law is quadratic; the origin matches for free, since every cumulative quantity
is zero there) is then a proof rather than a spot check. A family that declares
neither is INADMISSIBLE -- the caller refuses instead of guessing, because a
wrapper or an interpolation this module has not seen can bend the coefficient
anywhere between two matching samples.

Admitted families and their laws between breaks:

* ``FlatVolSurface``: w(u) = sigma^2 u, one piece, no break.
* ``TermStructureVolSurface``: total variance is linearly interpolated between
  its pillars and the vol is held flat outside them (so the two outer pieces are
  w = v^2 u through the origin). Breaks: every pillar.
* ``FlatRateCurve`` / ``NoDividend`` / ``ContinuousDividendYield``: constant, degree 1.
* ``LogLinearRateCurve``: ln DF is linear between pillars and extrapolates at the
  end pillars' zero rates -- degree 1. Breaks: every pillar.
* ``LinearRateCurve``: the ZERO RATE is linear between pillars, so -ln DF = r(t) t
  is quadratic -- degree 2. Equal zero rates at the two ends of a piece do NOT make
  the forward flat across it (pillars (0, 1%) and (T, 10%) have one piece whose
  zero rate is 5.5% at T/2); its midpoint is sampled as well (review 2026-09-16
  R2). Breaks: every pillar.
* ``TermStructureDividendYield``: the yield is linearly interpolated and flat
  outside, so q(t) t is quadratic -- degree 2. Breaks: every pillar.
* The parallel-shift wrappers add a constant to the rate or yield, i.e. a linear
  term to the cumulative quantity: same breaks, same degree -- with one exception:
  ``ParallelShiftVolSurface`` shifts sigma, and (sigma(u) + s)^2 u is affine in u
  only when sigma is already constant. A shifted TERM surface is inadmissible.
* The frozen-market roll wrappers of ``quantark.intraday.roll`` translate their
  inner family by the roll: ``ShiftedTradingVolSurface`` has w'(u) = w(u + du) -
  w(du), ``ShiftedRateCurve`` and ``ShiftedDividendYield`` the cumulative carry
  C'(t) = C(t + s) - C(s). Each is the inner law restricted to the later window,
  so it keeps the inner degree and the inner breaks moved back by the roll. A
  local theta reprices exactly these (review 2026-09-16 R6).

Deliberately NOT admitted: ``CubicSplineRateCurve`` (equal pillar values do not
make a spline flat between them), ``GridVolSurface`` and every smile surface
(no single barrier volatility exists), and anything unrecognized -- including a
subclass of an admitted family, which may override its interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CoefficientBreaks:
    """Calendar taus in ``(0, t_end)`` where a coefficient's law may change."""

    variance: Tuple[float, ...]         # dW/dtau may change here (clock segments and volatility pillars)
    carry: Tuple[float, ...]            # the zero rate or the dividend yield may change here
    curve: Tuple[float, ...]            # the CURVE pillars alone, without the clock's own segment boundaries
    qualified: bool
    reason: str                         # why not, when ``qualified`` is False
    #: Per carry coefficient: its own pillars inside the window, and the polynomial degree of its cumulative
    #: quantity (-ln DF, resp. q(t) t) between them. Total variance is affine between variance breaks for every
    #: admitted family, so it needs no degree.
    rate: Tuple[float, ...] = ()
    rate_degree: int = 1
    div: Tuple[float, ...] = ()
    div_degree: int = 1

    @property
    def all_taus(self) -> Tuple[float, ...]:
        return tuple(sorted(set(self.variance) | set(self.carry)))


def _inside(taus, t_end: float) -> Tuple[float, ...]:
    return tuple(sorted({float(t) for t in taus if 0.0 < float(t) < float(t_end)}))


def _translated(pillars: Tuple[float, ...], shift: float) -> Tuple[float, ...]:
    """The pillars of an inner law seen ``shift`` later: moved back, the ones already behind dropped."""
    return tuple(float(t) - float(shift) for t in pillars if float(t) - float(shift) > 0.0)


def _vol_pillars_in_trading_time(surface) -> Optional[Tuple[float, ...]]:
    """Maturities (on the surface's own axis) where its total-variance law changes, or None."""
    from quantark.intraday.roll import ShiftedTradingVolSurface
    from quantark.param.vol.vol_surface import (FlatVolSurface, ParallelShiftVolSurface, TermStructureVolSurface)

    kind = type(surface)
    if kind is FlatVolSurface:
        return ()
    if kind is TermStructureVolSurface:
        return tuple(float(t) for t in surface.times)
    if kind is ParallelShiftVolSurface:
        base = _vol_pillars_in_trading_time(surface.base)
        # (sigma(u) + shift)^2 * u is affine in u only where sigma is already constant.
        return () if base == () else None
    if kind is ShiftedTradingVolSurface:
        base = _vol_pillars_in_trading_time(surface.inner)
        return None if base is None else _translated(base, surface.delta_u)
    return None


def _rate_law(curve) -> Optional[Tuple[Tuple[float, ...], int]]:
    """(breaks, degree of -ln DF between them) of a rate curve, or None when it declares no law."""
    from quantark.intraday.roll import ShiftedRateCurve
    from quantark.param.rrf.rate_curve import (FlatRateCurve, LinearRateCurve, LogLinearRateCurve,
                                               ParallelShiftRateCurve)

    kind = type(curve)
    if kind is FlatRateCurve:
        return (), 1
    if kind is LogLinearRateCurve:
        return tuple(float(t) for t, _ in curve.pillars), 1
    if kind is LinearRateCurve:
        return tuple(float(t) for t, _ in curve.pillars), 2
    if kind is ParallelShiftRateCurve:
        return _rate_law(curve.base_curve)         # + shift * t: a linear term keeps breaks and degree
    if kind is ShiftedRateCurve:
        base = _rate_law(curve.inner)
        return None if base is None else (_translated(base[0], curve.shift), base[1])
    return None


def _div_law(div) -> Optional[Tuple[Tuple[float, ...], int]]:
    """(breaks, degree of q(t) t between them) of a dividend yield, or None when it declares no law."""
    from quantark.intraday.roll import ShiftedDividendYield
    from quantark.param.div.dividend_yield import (ContinuousDividendYield, NoDividend,
                                                   ParallelShiftDividendYield, TermStructureDividendYield)

    kind = type(div)
    if div is None or kind in (NoDividend, ContinuousDividendYield):
        return (), 1
    if kind is TermStructureDividendYield:
        return tuple(float(t) for t in div.times), 2
    if kind is ParallelShiftDividendYield:
        return _div_law(div.base)
    if kind is ShiftedDividendYield:
        base = _div_law(div.inner)
        return None if base is None else (_translated(base[0], div.shift), base[1])
    return None


def _calendar_of_trading(time_map, u: float) -> Optional[float]:
    """Smallest calendar tau whose variance clock reads ``u``.

    NOT ``IntradayTimeMap.to_calendar``: that is the CARRY map, which collapses a
    zero-weight plateau onto its end, so it would place a volatility pillar after
    the lunch break that the pillar actually falls inside. The variance rate
    changes where the clock first reaches ``u``, which is where the surface starts
    quoting the next piece.
    """
    for segment in time_map.segments:
        if segment.u_end < u or segment.u_end == segment.u_start:
            continue                    # already passed, or a plateau that accrues nothing
        span = segment.u_end - segment.u_start
        frac = min(max((u - segment.u_start) / span, 0.0), 1.0)
        return float(segment.tau_start + frac * (segment.tau_end - segment.tau_start))
    return None


def coefficient_breaks(ctx, t_end: float) -> CoefficientBreaks:
    """Every calendar tau in ``(0, t_end)`` where a coefficient of ``ctx`` may change its law.

    The variance breaks are the clock's own segment boundaries (the slope
    du/dtau changes there) UNION the volatility surface's pillars mapped back
    onto the calendar axis. The carry breaks are the rate and dividend pillars,
    which are already quoted on the calendar axis in intraday mode.
    """
    from quantark.param.vol import TradingClockVolSurface

    env, time_map = ctx.pricing_env, ctx.time_map
    clock = _inside([s.tau_end for s in time_map.segments] + [s.tau_start for s in time_map.segments], t_end)

    def inadmissible(reason):
        # The clock's own segments are always known; only the CURVE law is in doubt,
        # so a caller that just needs somewhere safe to stop still gets the segments.
        return CoefficientBreaks(clock, clock, (), False, reason)

    surface = env.vol_surface
    inner = surface.inner if isinstance(surface, TradingClockVolSurface) else surface
    pillars_u = _vol_pillars_in_trading_time(inner)
    if pillars_u is None:
        return inadmissible(f"{type(inner).__name__} does not declare where its total-variance law changes, "
                           "so a constant variance rate cannot be established between samples")
    rate_law = _rate_law(env.rate_curve)
    if rate_law is None:
        return inadmissible(f"{type(env.rate_curve).__name__} does not declare where its forward rate changes, "
                           "so a flat forward cannot be established between samples")
    div_law = _div_law(env.div_yield)
    if div_law is None:
        return inadmissible(f"{type(env.div_yield).__name__} does not declare where its yield changes, "
                           "so a flat carry cannot be established between samples")
    (rate_taus, rate_degree), (div_taus, div_degree) = rate_law, div_law
    horizon_u = float(time_map.to_trading(float(t_end)))
    vol_taus = [tau for tau in (_calendar_of_trading(time_map, float(u))
                                for u in pillars_u if 0.0 < float(u) <= horizon_u) if tau is not None]
    return CoefficientBreaks(variance=_inside(list(clock) + vol_taus, t_end),
                             carry=_inside(list(clock) + list(rate_taus) + list(div_taus), t_end),
                             curve=_inside(vol_taus + list(rate_taus) + list(div_taus), t_end),
                             qualified=True, reason="", rate=_inside(rate_taus, t_end), rate_degree=rate_degree,
                             div=_inside(div_taus, t_end), div_degree=div_degree)
