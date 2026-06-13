"""
Credit environment bumping and portfolio revaluation helpers for VaR.

Builds a perturbed CreditPricingEnvironment (shifting the hazard intensity or
the discount rate) and revalues a credit portfolio under per-entity overrides.
These underpin finite-difference sensitivities (parametric VaR) and full
revaluation (historical / Monte-Carlo VaR).
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

from quantark.priceenv import CreditPricingEnvironment
from quantark.util.numerical import is_zero


def bump_env(
    env: CreditPricingEnvironment,
    hazard_change: float = 0.0,
    rate_shift: float = 0.0,
    spread_change: Optional[float] = None,
) -> CreditPricingEnvironment:
    """
    Return a new credit environment with the requested factor perturbations.

    ``hazard_change`` is an absolute shift of the hazard intensity (the shared
    curve-level risk factor). Quoted-spread shocks must be converted to a hazard
    shift via :func:`~quantark.asset.credit.conventions.spread_shift_to_hazard_shift`
    before calling this.

    ``spread_change`` is a deprecated alias of ``hazard_change`` retained for
    backward compatibility; the curve-level factor is the hazard intensity, so
    the old name was a misnomer (it never applied a recovery conversion).
    """
    if spread_change is not None:
        warnings.warn(
            "bump_env(spread_change=...) is deprecated; use 'hazard_change'. The "
            "credit curve-level factor is the hazard intensity (no recovery "
            "conversion is applied here).",
            DeprecationWarning,
            stacklevel=2,
        )
        hazard_change = spread_change
    new = env
    if not is_zero(hazard_change):
        new = new.with_hazard_shift(hazard_change)
    if not is_zero(rate_shift):
        new = new.with_rate_shift(rate_shift)
    return new


def portfolio_value(portfolio, envs: Dict[str, CreditPricingEnvironment]) -> float:
    """Total market value of the portfolio under a per-entity environment map."""
    return sum(
        pos.get_market_value(envs[pos.reference_entity])
        for pos in portfolio.positions.values()
    )
