"""
Market scenarios for scenario-based hedging.

A MarketScenario describes one joint market move (e.g. spot -10% with vol
+8 points). Scenario-based hedging sizes hedges on full-revaluation P&L
under such moves instead of local Greeks: Greeks are derivatives at the
current point, while scenario P&L captures the large, messy moves that
actually hurt structured-product books.

apply_scenario() supports the flat market parameterization used by the
backtest engine (SpotQuote / FlatVolSurface / FlatRateCurve). Shifting a
non-flat surface or curve raises ValidationError rather than silently
approximating a parallel shift.
"""

from dataclasses import dataclass, replace
from typing import Optional

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero


@dataclass(frozen=True)
class MarketScenario:
    """
    One joint market move for scenario-based hedging.

    Attributes:
        name: Scenario identifier, used as the measure key in hedge targets
        spot_shift: Relative spot move (-0.10 = spot down 10%)
        vol_shift: Absolute volatility move in decimal (+0.08 = vol up 8 points)
        rate_shift: Absolute rate move in decimal (+0.01 = rates up 100 bps)
        weight: Relative importance when scenarios cannot all be hedged
            exactly (least-squares regime)
    """

    name: str
    spot_shift: float = 0.0
    vol_shift: float = 0.0
    rate_shift: float = 0.0
    weight: float = 1.0

    def __post_init__(self):
        if not self.name:
            raise ValidationError("MarketScenario name must be non-empty")
        if self.spot_shift <= -1.0:
            raise ValidationError(
                f"spot_shift must be > -1 (spot stays positive), "
                f"got {self.spot_shift}"
            )
        if self.weight <= 0:
            raise ValidationError(
                f"MarketScenario weight must be positive, got {self.weight}"
            )
        if (
            is_zero(self.spot_shift)
            and is_zero(self.vol_shift)
            and is_zero(self.rate_shift)
        ):
            raise ValidationError(
                f"MarketScenario '{self.name}' has no shifts; "
                "at least one of spot/vol/rate must move"
            )


def apply_scenario(
    pricing_env: PricingEnvironment, scenario: MarketScenario
) -> PricingEnvironment:
    """
    Build a new pricing environment with the scenario's shifts applied.

    The original environment is not modified; unaffected fields (valuation
    date, day count convention, calendar, basis yield, dividend yield) are
    carried over unchanged.

    Args:
        pricing_env: Base pricing environment
        scenario: Scenario to apply

    Returns:
        New PricingEnvironment under the scenario

    Raises:
        ValidationError: If a shift targets market data the environment
            lacks, or a vol/rate shift targets a non-flat surface/curve
    """
    updates = {}

    if not is_zero(scenario.spot_shift):
        if pricing_env.spot_quote is None:
            raise ValidationError(
                f"Scenario '{scenario.name}' shifts spot but the pricing "
                "environment has no spot quote"
            )
        updates["spot_quote"] = SpotQuote(
            spot=pricing_env.spot_quote.spot * (1.0 + scenario.spot_shift),
            asset_name=pricing_env.spot_quote.asset_name,
        )

    if not is_zero(scenario.vol_shift):
        if not isinstance(pricing_env.vol_surface, FlatVolSurface):
            raise ValidationError(
                f"Scenario '{scenario.name}' shifts volatility but the "
                "environment's vol surface is "
                f"{type(pricing_env.vol_surface).__name__}; only "
                "FlatVolSurface supports scenario shifts"
            )
        updates["vol_surface"] = FlatVolSurface(
            volatility=pricing_env.vol_surface.volatility + scenario.vol_shift
        )

    if not is_zero(scenario.rate_shift):
        if not isinstance(pricing_env.rate_curve, FlatRateCurve):
            raise ValidationError(
                f"Scenario '{scenario.name}' shifts rates but the "
                "environment's rate curve is "
                f"{type(pricing_env.rate_curve).__name__}; only "
                "FlatRateCurve supports scenario shifts"
            )
        updates["rate_curve"] = FlatRateCurve(
            rate=pricing_env.rate_curve.rate + scenario.rate_shift
        )

    return replace(pricing_env, **updates)


def instrument_scenario_pnl(
    product: BaseEquityProduct,
    engine: BaseEngine,
    pricing_env: PricingEnvironment,
    scenario: MarketScenario,
    base_price: Optional[float] = None,
) -> float:
    """
    Per-unit P&L of one contract under a scenario (full revaluation).

    Args:
        product: Contract to revalue
        engine: Pricing engine for the contract
        pricing_env: Base pricing environment
        scenario: Scenario to apply
        base_price: Pre-computed base price (computed if omitted)

    Returns:
        price(scenario) - price(base) per unit
    """
    if base_price is None:
        base_price = engine.price(product, pricing_env)
    bumped_env = apply_scenario(pricing_env, scenario)
    return engine.price(product, bumped_env) - base_price


def portfolio_scenario_pnl(
    portfolio: Portfolio,
    underlying: str,
    pricing_env: PricingEnvironment,
    scenario: MarketScenario,
) -> float:
    """
    Portfolio P&L under a scenario (full revaluation of every position).

    Temporarily swaps the bumped environment into the portfolio, revalues,
    and restores the original environment.

    Args:
        portfolio: Portfolio to revalue
        underlying: Underlying whose environment the scenario shifts
        pricing_env: Base pricing environment for that underlying
        scenario: Scenario to apply

    Returns:
        value(scenario) - value(base)
    """
    base_value = portfolio.get_portfolio_value()
    bumped_env = apply_scenario(pricing_env, scenario)
    original_env = portfolio.pricing_environments[underlying]
    portfolio.pricing_environments[underlying] = bumped_env
    try:
        bumped_value = portfolio.get_portfolio_value()
    finally:
        portfolio.pricing_environments[underlying] = original_env
    return bumped_value - base_value
