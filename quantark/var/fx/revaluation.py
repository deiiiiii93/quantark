"""
FX environment bumping and portfolio revaluation helpers for VaR.

These primitives build a perturbed :class:`FxPricingEnvironment` for a single
currency pair (shifting spot, vol, domestic or foreign rate) and revalue an
FX portfolio under a set of per-pair environment overrides. They underpin both
finite-difference sensitivities (parametric VaR) and full revaluation
(historical / Monte Carlo VaR).
"""
from __future__ import annotations

import dataclasses
from typing import Dict

from quantark.param import (
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
    TermStructureVolSurface,
)
from quantark.param.rrf.rate_curve import InterpolatedRateCurve
from quantark.param.vol.vannavolga import VannaVolgaVolSurface, SmileQuotes
from quantark.priceenv import FxPricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

_MIN_VOL = 1e-6


def _shift_curve(curve, shift: float):
    if is_zero(shift):
        return curve
    if isinstance(curve, FlatRateCurve):
        return FlatRateCurve(rate=curve.get_rate(1.0) + shift)
    if isinstance(curve, InterpolatedRateCurve):
        return curve.__class__([(t, r + shift) for t, r in curve.pillars])
    raise ValidationError(
        f"Cannot bump FX curve type {type(curve).__name__} for VaR; "
        "only FlatRateCurve or interpolated curves are supported."
    )


def _shift_vol(surface, change: float):
    if is_zero(change) or surface is None:
        return surface
    if isinstance(surface, FlatVolSurface):
        return FlatVolSurface(volatility=max(_MIN_VOL, surface.volatility + change))
    if isinstance(surface, TermStructureVolSurface):
        return TermStructureVolSurface(
            times=list(surface.times),
            vols=[max(_MIN_VOL, float(v) + change) for v in surface.vols],
        )
    if isinstance(surface, VannaVolgaVolSurface):
        # Additive parallel vol move: shift the ATM level and keep RR/BF fixed.
        # RR (= sigma_25c - sigma_25p) and BF are invariant under an additive
        # parallel shift, so moving only sigma_atm IS the full-quote move here.
        q = surface.quotes
        return surface.with_quotes(
            SmileQuotes(
                sigma_atm=max(_MIN_VOL, q.sigma_atm + change),
                rr25=q.rr25,
                bf25_2vol=q.bf25_2vol,
            )
        )
    raise ValidationError(
        f"Cannot bump FX vol surface type {type(surface).__name__} for VaR."
    )


def bump_env(
    env: FxPricingEnvironment,
    spot_return: float = 0.0,
    vol_change: float = 0.0,
    dom_shift: float = 0.0,
    for_shift: float = 0.0,
) -> FxPricingEnvironment:
    """Return a new FX environment with the requested factor perturbations."""
    new = env
    if not is_zero(spot_return):
        new_spot = env.spot_quote.spot * (1.0 + spot_return)
        new = dataclasses.replace(
            new,
            spot_quote=SpotQuote(
                spot=new_spot,
                timestamp=env.spot_quote.timestamp,
                asset_name=env.spot_quote.asset_name,
            ),
        )
    if not is_zero(vol_change):
        new = dataclasses.replace(new, vol_surface=_shift_vol(new.vol_surface, vol_change))
    if not is_zero(dom_shift):
        new = dataclasses.replace(
            new, domestic_curve=_shift_curve(new.domestic_curve, dom_shift)
        )
    if not is_zero(for_shift):
        new = dataclasses.replace(
            new, foreign_curve=_shift_curve(new.foreign_curve, for_shift)
        )
    return new


def portfolio_value(portfolio, envs: Dict[str, FxPricingEnvironment]) -> float:
    """Total market value of the portfolio under a per-pair environment map."""
    return sum(
        pos.get_market_value(envs[pos.underlying])
        for pos in portfolio.positions.values()
    )


def position_value(position, env: FxPricingEnvironment) -> float:
    return position.get_market_value(env)
