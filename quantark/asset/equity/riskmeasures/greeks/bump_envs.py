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
    """Parallel-shift the vol surface by ``direction * vol_bump``; the SHAPE
    (term structure, smile, grid type) is preserved via ``parallel_shifted``.
    ``current_vol`` is the (K, T) vol the caller read: it gates the bump (the
    legacy check that a stressed vol must stay positive)."""
    new_vol = current_vol + direction * vol_bump
    if new_vol <= 0:
        raise ValidationError(
            f"Stressed volatility must be positive, got {new_vol}"
        )

    env = deepcopy(pricing_env)
    env.vol_surface = shift_vol_surface(env.vol_surface, direction * vol_bump)
    return env


def shift_vol_surface(surface, shift: float):
    """``surface.parallel_shifted(shift)`` for the BlackImpliedVolSurface
    hierarchy (flat stays flat with the legacy floats, a grid stays a grid);
    any other surface object (SVIVolSurface, a sticky-moneyness view — whose
    ``__getattr__`` would forward ``parallel_shifted`` to its base and drop the
    view) is wrapped so ``get_vol`` reads base + shift."""
    from quantark.param.vol import BlackImpliedVolSurface, ParallelShiftVolSurface

    if isinstance(surface, BlackImpliedVolSurface):
        return surface.parallel_shifted(shift)
    return ParallelShiftVolSurface(surface, shift)


def build_div_bumped_env(
    pricing_env: PricingEnvironment,
    product: BaseEquityProduct,
    current_div: float,
    div_bump: float,
    *,
    direction: float,
) -> PricingEnvironment:
    """Parallel-shift the dividend yield by ``direction * div_bump``; the term
    SHAPE is preserved via ``parallel_shifted``. ``current_div`` is kept for
    signature symmetry with the vol helper (the shift needs no level)."""
    env = deepcopy(pricing_env)
    env.div_yield = shift_dividend_yield(env.div_yield, direction * div_bump)
    return env


def shift_dividend_yield(div_yield, shift: float):
    """``div_yield.parallel_shifted(shift)``; ``None`` (PricingEnvironment reads
    it as a zero yield) becomes the legacy continuous constant."""
    from quantark.param.div import (
        ContinuousDividendYield, DividendYield, ParallelShiftDividendYield,
    )

    if div_yield is None:
        return ContinuousDividendYield(0.0 + shift)
    if isinstance(div_yield, DividendYield):
        return div_yield.parallel_shifted(shift)
    return ParallelShiftDividendYield(div_yield, shift)


def build_rate_bumped_env(
    pricing_env: PricingEnvironment,
    rate_bump: float,
    *,
    direction: float,
) -> PricingEnvironment:
    """Parallel-shift the rate curve by ``direction * rate_bump``
    (continuously compounded, per CALENDAR year).

    Delegates to ``RateCurve.parallel_shifted``: a flat curve stays flat
    (``FlatRateCurve(rate + shift)``, the legacy formula, bitwise), any other
    curve keeps its SHAPE (``FlatRateCurve(r(T) + bump)`` differs from the base
    by ``r(T) - r(t) + bump`` at every ``t < T`` and measures a curve reshaping
    of hundreds of bp, not a 1bp rate move), and a TradingClock wrapper
    re-exposes the shifted calendar-quoted inner through the same time map.
    """
    env = deepcopy(pricing_env)
    env.rate_curve = env.rate_curve.parallel_shifted(direction * rate_bump)
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
