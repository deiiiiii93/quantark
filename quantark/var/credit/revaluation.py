"""
Credit environment bumping and portfolio revaluation helpers for VaR.

Builds a perturbed CreditPricingEnvironment (shifting the hazard intensity or
the discount rate) and revalues a credit portfolio under per-entity overrides.
These underpin finite-difference sensitivities (parametric VaR) and full
revaluation (historical / Monte-Carlo VaR).
"""
from __future__ import annotations

from typing import Dict

from quantark.priceenv import CreditPricingEnvironment
from quantark.util.numerical import is_zero


def bump_env(
    env: CreditPricingEnvironment,
    spread_change: float = 0.0,
    rate_shift: float = 0.0,
) -> CreditPricingEnvironment:
    """Return a new credit environment with the requested factor perturbations."""
    new = env
    if not is_zero(spread_change):
        new = new.with_hazard_shift(spread_change)
    if not is_zero(rate_shift):
        new = new.with_rate_shift(rate_shift)
    return new


def portfolio_value(portfolio, envs: Dict[str, CreditPricingEnvironment]) -> float:
    """Total market value of the portfolio under a per-entity environment map."""
    return sum(
        pos.get_market_value(envs[pos.reference_entity])
        for pos in portfolio.positions.values()
    )
