"""
Normalization helpers for term-sheet RFQ inputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.rfq.models import RFQTermsheetInput
from quantark.rfq.registry import ENGINE_BUILDERS, PRODUCT_BUILDERS
from quantark.util.exceptions import ValidationError


def build_product_from_termsheet(termsheet: RFQTermsheetInput) -> Any:
    """Build product instance from term-sheet input."""
    return PRODUCT_BUILDERS.build(termsheet.product_type, termsheet.product_kwargs)


def build_engine_from_termsheet(termsheet: RFQTermsheetInput) -> Any:
    """Build engine instance from term-sheet engine spec."""
    return ENGINE_BUILDERS.build(termsheet.engine_spec)


def build_pricing_env_from_market_kwargs(market_kwargs: Dict[str, Any]) -> PricingEnvironment:
    """Build pricing environment from a normalized market kwargs mapping."""
    allowed = {
        "valuation_date",
        "rate_curve",
        "rate",
        "spot_quote",
        "spot",
        "asset_name",
        "vol_surface",
        "volatility",
        "div_yield",
        "dividend_yield",
        "q",
        "basis_yield",
        "day_count_convention",
        "bus_days_in_year",
        "calendar",
    }
    unknown = set(market_kwargs) - allowed
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValidationError(f"Unsupported market_kwargs: {unknown_list}")

    valuation_date = market_kwargs.get("valuation_date")
    if valuation_date is None:
        raise ValidationError("market_kwargs.valuation_date is required")
    if not isinstance(valuation_date, datetime):
        raise ValidationError("market_kwargs.valuation_date must be datetime")

    rate_curve = market_kwargs.get("rate_curve")
    if rate_curve is None:
        rate = market_kwargs.get("rate")
        if rate is None:
            raise ValidationError(
                "market_kwargs requires either rate_curve or rate"
            )
        rate_curve = FlatRateCurve(rate=float(rate))

    spot_quote = market_kwargs.get("spot_quote")
    if spot_quote is None and "spot" in market_kwargs:
        spot_quote = SpotQuote(
            spot=float(market_kwargs["spot"]),
            timestamp=valuation_date,
            asset_name=market_kwargs.get("asset_name"),
        )

    vol_surface = market_kwargs.get("vol_surface")
    if vol_surface is None and "volatility" in market_kwargs:
        vol_surface = FlatVolSurface(volatility=float(market_kwargs["volatility"]))

    div_yield = market_kwargs.get("div_yield")
    if div_yield is None:
        if "dividend_yield" in market_kwargs:
            div_yield = ContinuousDividendYield(
                div_yield=float(market_kwargs["dividend_yield"])
            )
        elif "q" in market_kwargs:
            div_yield = ContinuousDividendYield(div_yield=float(market_kwargs["q"]))

    return PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        div_yield=div_yield,
        basis_yield=market_kwargs.get("basis_yield"),
        day_count_convention=market_kwargs.get("day_count_convention"),
        bus_days_in_year=market_kwargs.get("bus_days_in_year", 252),
        calendar=market_kwargs.get("calendar"),
    )
