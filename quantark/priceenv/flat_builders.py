"""Canonical flat-curve builders (spec WP3.4).

The Q1 constants are a rate, a yield, and a vol; the Q2 curve coordinates
are D(0,T) = exp(-r*T), B(T) = (r - q)*T, and V(T) = sigma^2*T. Never set
B = q or V = sigma. Times are ACT/365F.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from quantark.param.div.forward_carry_curve import ForwardCarryCurve


def build_flat_curves(r: float, q: float, sigma: float, tenors: Sequence[float]):
    """(rate_curve, carry_curve, vol_surface) with the canonical flat mapping."""
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.param.vol.vol_surface import TermStructureVolSurface

    tenors = [float(t) for t in tenors]
    rate_curve = LinearRateCurve([(t, float(r)) for t in tenors])
    carry_curve = ForwardCarryCurve([(t, (float(r) - float(q)) * t) for t in tenors])
    vol_surface = TermStructureVolSurface(
        times=tenors, vols=[float(sigma)] * len(tenors)
    )
    return rate_curve, carry_curve, vol_surface


def build_flat_env(
    r: float,
    q: float,
    sigma: float,
    spot: float,
    tenors: Sequence[float],
    valuation_date: Optional[datetime] = None,
):
    """PricingEnvironment on flat term-structure curves; the carry curve is
    attached as ``env.carry_curve`` (PricingEnvironment itself consumes only
    the derived ``div_yield`` — documented adapter route, spec WP3.1)."""
    from quantark.param import SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.util.calendar import DayCountConvention

    rate_curve, carry_curve, vol_surface = build_flat_curves(r, q, sigma, tenors)
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot)),
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=carry_curve.to_dividend_yield(rate_curve),
        valuation_date=valuation_date,
        day_count_convention=DayCountConvention.ACT_365,
    )
    env.carry_curve = carry_curve
    return env
