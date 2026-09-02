"""Bumped pricing environments and finite-difference helpers.

Bodies moved verbatim from GreeksCalculator (R1a pure code motion); the
facade keeps same-named private methods as one-line delegates.
"""

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Optional, Tuple

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention, calculate_year_fraction
from quantark.util.exceptions import ValidationError


def ensure_base_price(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float],
) -> float:
    """Return base price, computing it if needed."""
    return (
        base_price if base_price is not None else engine.price(product, pricing_env)
    )


def resolve_bump_engine(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
) -> BaseEngine:
    """Return the engine context used for numerical bump repricing."""
    create_context = getattr(engine, "create_bump_context", None)
    if not callable(create_context):
        return engine
    bump_engine = create_context(product, pricing_env)
    return bump_engine if bump_engine is not None else engine


def calculate_sensitivity(
    base_price: float,
    price_up: float,
    price_down: Optional[float] = None,
    bump: float = 1.0,
    scale: float = 1.0,
    mode: str = "central",
) -> float:
    """Generic finite-difference sensitivity helper."""
    if mode == "central":
        if price_down is None:
            raise ValidationError("central mode requires price_down")
        return (price_up - price_down) / (2.0 * scale * bump)
    if mode == "second_order":
        if price_down is None:
            raise ValidationError("second_order mode requires price_down")
        return (price_up - 2.0 * base_price + price_down) / (
            scale * bump
        ) ** 2
    if mode == "one_sided":
        return price_up - base_price
    raise ValidationError(f"Unknown sensitivity mode: {mode}")


def spot_bumped_prices(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    bump: float,
    base_price: Optional[float] = None,
    reuse: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    """
    Compute base, up, and down spot bump prices, optionally reusing bumps.
    """
    base_price = ensure_base_price(product, pricing_env, engine, base_price)
    if reuse is not None:
        price_up_spot, price_down_spot = reuse
    else:
        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= 1 + bump
        price_up_spot = engine.price(product, env_up)

        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= 1 - bump
        price_down_spot = engine.price(product, env_down)

    return base_price, price_up_spot, price_down_spot


def build_vol_bumped_env(
    pricing_env: PricingEnvironment,
    product: BaseEquityProduct,
    current_vol: float,
    vol_bump: float,
    *,
    direction: float,
) -> PricingEnvironment:
    from quantark.param.vol import FlatVolSurface, TermStructureVolSurface

    new_vol = current_vol + direction * vol_bump
    if new_vol <= 0:
        raise ValidationError(
            f"Stressed volatility must be positive, got {new_vol}"
        )

    env = deepcopy(pricing_env)
    if isinstance(pricing_env.vol_surface, TermStructureVolSurface):
        new_vols = [float(v) + direction * vol_bump for v in pricing_env.vol_surface.vols]
        if any(v <= 0 for v in new_vols):
            raise ValidationError("Stressed term-structure vol must be positive.")
        env.vol_surface = TermStructureVolSurface(
            times=list(pricing_env.vol_surface.times), vols=new_vols
        )
    else:
        env.vol_surface = FlatVolSurface(new_vol)
    return env


def build_div_bumped_env(
    pricing_env: PricingEnvironment,
    product: BaseEquityProduct,
    current_div: float,
    div_bump: float,
    *,
    direction: float,
) -> PricingEnvironment:
    from quantark.param.div import ContinuousDividendYield, TermStructureDividendYield

    new_div = current_div + direction * div_bump

    env = deepcopy(pricing_env)
    if isinstance(pricing_env.div_yield, TermStructureDividendYield):
        new_yields = [
            float(y) + direction * div_bump for y in pricing_env.div_yield.yields
        ]
        env.div_yield = TermStructureDividendYield(
            times=list(pricing_env.div_yield.times), yields=new_yields
        )
    else:
        env.div_yield = ContinuousDividendYield(new_div)
    return env


def advance_theta_bump(
    pricing_env: PricingEnvironment,
    time_bump_days: int,
    time_bump_mode: str,
) -> Tuple[datetime, float, str]:
    """Advance the theta valuation date and return date, year fraction, mode."""
    mode = resolve_theta_bump_mode(pricing_env, time_bump_mode)
    if mode == "calendar_days":
        bumped_date = pricing_env.valuation_date + timedelta(days=time_bump_days)
    else:
        calendar = getattr(pricing_env, "calendar", None)
        if calendar is None or not hasattr(calendar, "add_business_days"):
            raise ValidationError(
                "time_bump_mode='business_days' requires pricing_env.calendar "
                "with add_business_days()"
            )
        bumped_date = calendar.add_business_days(
            pricing_env.valuation_date, time_bump_days
        )

    time_bump = calculate_year_fraction(
        pricing_env.valuation_date,
        bumped_date,
        pricing_env.day_count_convention,
        pricing_env.bus_days_in_year,
        calendar=getattr(pricing_env, "calendar", None),
    )
    return bumped_date, time_bump, mode


def time_days_per_year(
    pricing_env: PricingEnvironment, clock: Optional[str], time_bump_mode: str
) -> float:
    """Days-per-year divisor of a per-day time greek under a clock.

    ``"1d"`` is one calendar day (365), ``"1td"`` one trading day (the
    env's ``bus_days_in_year``); a bare name (``None``) follows the resolved
    theta bump mode, so a BUSINESS_DAYS env with a calendar reports per
    trading day and every other env per calendar day.
    """
    if clock == "1d":
        return 365.0
    if clock == "1td":
        return float(pricing_env.bus_days_in_year)
    if clock is not None:
        raise ValidationError(f"Unknown clock qualifier: {clock}")
    if resolve_theta_bump_mode(pricing_env, time_bump_mode) == "business_days":
        return float(pricing_env.bus_days_in_year)
    return 365.0


def resolve_theta_bump_mode(
    pricing_env: PricingEnvironment, time_bump_mode: str
) -> str:
    """Resolve auto theta mode against the pricing environment."""
    mode = time_bump_mode.lower()
    if mode not in {"auto", "calendar_days", "business_days"}:
        raise ValidationError(
            "time_bump_mode must be one of 'auto', 'calendar_days', "
            f"or 'business_days', got {time_bump_mode!r}"
        )
    if mode != "auto":
        return mode

    if (
        pricing_env.day_count_convention == DayCountConvention.BUSINESS_DAYS
        and getattr(pricing_env, "calendar", None) is not None
    ):
        return "business_days"
    return "calendar_days"
