"""Where the resolved context's market coefficients may change, and whether that is provable.

Two intraday claims need this. A closed-form first-passage formula is exact only
if ln S has a constant drift per unit variance up to expiry; a LOCAL theta is a
derivative only inside one piece of the coefficients. Both are statements about
whole INTERVALS, and no amount of pointwise sampling establishes either: a curve
evaluated at two ends says nothing about the law between them.

So every family this module admits declares two things: the instants where its
law may change, and that between them the law is affine in the quantity the
caller compares (total variance against calendar time, the zero rate, the
dividend yield). Sampling at every declared break is then a proof rather than a
spot check. A family that declares neither is UNQUALIFIED — the caller refuses
instead of guessing, because a wrapper or an interpolation this module has not
seen can bend the coefficient anywhere between two matching samples.

Admitted families and why their law is affine between breaks:

* ``FlatVolSurface``: w(u) = sigma^2 u, one piece, no break.
* ``TermStructureVolSurface``: total variance is linearly interpolated between
  its pillars and the vol is held flat outside them (so the two outer pieces are
  w = v^2 u through the origin). Breaks: every pillar.
* ``FlatRateCurve`` / ``NoDividend`` / ``ContinuousDividendYield``: constant.
* ``LinearRateCurve`` / ``LogLinearRateCurve``: the zero rate is linear, resp.
  ln DF is linear, between pillars; equal zero rates at both ends of a piece
  therefore mean a flat forward across all of it. Breaks: every pillar.
* ``TermStructureDividendYield``: the yield is linearly interpolated, flat
  outside. Breaks: every pillar.
* The parallel-shift wrappers add a constant to the quantity compared, so they
  inherit their base's breaks and preserve flatness -- with one exception:
  ``ParallelShiftVolSurface`` shifts sigma, and (sigma(u) + s)^2 u is affine in u
  only when sigma is already constant. A shifted TERM surface is unqualified.

Deliberately NOT admitted: ``CubicSplineRateCurve`` (equal pillar values do not
make a spline flat between them), ``GridVolSurface`` and every smile surface
(no single barrier volatility exists), and anything unrecognized.
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

    @property
    def all_taus(self) -> Tuple[float, ...]:
        return tuple(sorted(set(self.variance) | set(self.carry)))


def _inside(taus, t_end: float) -> Tuple[float, ...]:
    return tuple(sorted({float(t) for t in taus if 0.0 < float(t) < float(t_end)}))


def _vol_pillars_in_trading_time(surface) -> Optional[Tuple[float, ...]]:
    """Maturities (on the surface's own axis) where its total-variance law changes, or None."""
    from quantark.param.vol.vol_surface import (FlatVolSurface, ParallelShiftVolSurface, TermStructureVolSurface)

    if isinstance(surface, FlatVolSurface):
        return ()
    if isinstance(surface, TermStructureVolSurface):
        return tuple(float(t) for t in surface.times)
    if isinstance(surface, ParallelShiftVolSurface):
        base = _vol_pillars_in_trading_time(surface.base)
        # (sigma(u) + shift)^2 * u is affine in u only where sigma is already constant.
        return () if base == () else None
    return None


def _rate_pillars(curve) -> Optional[Tuple[float, ...]]:
    from quantark.param.rrf.rate_curve import (FlatRateCurve, LinearRateCurve, LogLinearRateCurve,
                                               ParallelShiftRateCurve)

    if isinstance(curve, FlatRateCurve):
        return ()
    if isinstance(curve, (LinearRateCurve, LogLinearRateCurve)):
        return tuple(float(t) for t, _ in curve.pillars)
    if isinstance(curve, ParallelShiftRateCurve):
        return _rate_pillars(curve.base_curve)     # a constant shift of the zero rate keeps flatness
    return None


def _div_pillars(div) -> Optional[Tuple[float, ...]]:
    from quantark.param.div.dividend_yield import (ContinuousDividendYield, NoDividend,
                                                   ParallelShiftDividendYield, TermStructureDividendYield)

    if div is None or isinstance(div, (NoDividend, ContinuousDividendYield)):
        return ()
    if isinstance(div, TermStructureDividendYield):
        return tuple(float(t) for t in div.times)
    if isinstance(div, ParallelShiftDividendYield):
        return _div_pillars(div.base)
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

    def unqualified(reason):
        # The clock's own segments are always known; only the CURVE law is in doubt,
        # so a caller that just needs somewhere safe to stop still gets the segments.
        return CoefficientBreaks(clock, clock, (), False, reason)

    surface = env.vol_surface
    inner = surface.inner if isinstance(surface, TradingClockVolSurface) else surface
    pillars_u = _vol_pillars_in_trading_time(inner)
    if pillars_u is None:
        return unqualified(f"{type(inner).__name__} does not declare where its total-variance law changes, "
                           "so a constant variance rate cannot be established between samples")
    rate_taus = _rate_pillars(env.rate_curve)
    if rate_taus is None:
        return unqualified(f"{type(env.rate_curve).__name__} does not declare where its forward rate changes, "
                           "so a flat forward cannot be established between samples")
    div_taus = _div_pillars(env.div_yield)
    if div_taus is None:
        return unqualified(f"{type(env.div_yield).__name__} does not declare where its yield changes, "
                           "so a flat carry cannot be established between samples")
    horizon_u = float(time_map.to_trading(float(t_end)))
    vol_taus = [tau for tau in (_calendar_of_trading(time_map, float(u))
                                for u in pillars_u if 0.0 < float(u) <= horizon_u) if tau is not None]
    return CoefficientBreaks(variance=_inside(list(clock) + vol_taus, t_end),
                             carry=_inside(list(clock) + list(rate_taus) + list(div_taus), t_end),
                             curve=_inside(vol_taus + list(rate_taus) + list(div_taus), t_end),
                             qualified=True, reason="")
