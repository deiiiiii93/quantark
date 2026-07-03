"""Shared term-structured environments + exact European reference price.

Used by the engine term-structure upgrade phases (spec
2026-07-03-engine-term-structure-upgrade-design.md, test layer 2).
"""
from datetime import datetime

import numpy as np
from scipy import stats

from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.param.vol.vol_surface import TermStructureVolSurface
from quantark.priceenv import PricingEnvironment

_SHAPES = {
    # times,   rates,             carries,             vols
    "up": (
        [0.25, 0.5, 1.0, 2.0],
        [0.020, 0.025, 0.030, 0.038],
        [0.005, 0.010, 0.015, 0.020],
        [0.18, 0.20, 0.22, 0.25],
    ),
    "down": (
        [0.25, 0.5, 1.0, 2.0],
        [0.038, 0.030, 0.025, 0.020],
        [0.020, 0.015, 0.010, 0.005],
        [0.25, 0.22, 0.20, 0.18],
    ),
    "kinked": (
        [0.25, 0.5, 1.0, 2.0],
        [0.020, 0.035, 0.025, 0.030],
        [-0.015, 0.020, -0.005, 0.010],  # negative-carry segments
        [0.22, 0.18, 0.24, 0.20],
    ),
}


def make_term_env(shape, spot=100.0, valuation_date=datetime(2026, 7, 3)):
    """Deterministic PricingEnvironment for shape in {flat, up, down, kinked}."""
    if shape == "flat":
        return PricingEnvironment(
            rate_curve=FlatRateCurve(0.03),
            valuation_date=valuation_date,
            spot_quote=SpotQuote(spot),
            vol_surface=FlatVolSurface(0.20),
            div_yield=ContinuousDividendYield(0.01),
        )
    times, rates, carries, vols = _SHAPES[shape]
    return PricingEnvironment(
        rate_curve=LinearRateCurve(list(zip(times, rates))),
        valuation_date=valuation_date,
        spot_quote=SpotQuote(spot),
        vol_surface=TermStructureVolSurface(times=list(times), vols=list(vols)),
        div_yield=TermStructureDividendYield(
            times=list(times), yields=list(carries)
        ),
    )


def reference_european_call_price(env, strike, maturity):
    """Exact European call under term structures via cumulative-to-T inputs."""
    S = env.spot
    T = float(maturity)
    vol = env.get_vol(strike, T)
    w = vol * vol * T
    df = env.get_discount_factor(T)
    fwd = S * np.exp((env.get_rate(T) - env.get_div_yield(T)) * T)
    d1 = (np.log(fwd / strike) + 0.5 * w) / np.sqrt(w)
    d2 = d1 - np.sqrt(w)
    return float(df * (fwd * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2)))
