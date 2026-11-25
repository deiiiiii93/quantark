"""How a cash curve is quoted: per calendar year or per trading year (0.15.0).

A ``BUSINESS_DAYS`` environment measures every time in trading years. A rate curve, dividend yield or futures basis
quoted per calendar year (the usual ACT/365 desk quote) must be read through the trading clock
(``TradingClockRateCurve`` / ``TradingClockDividendYield`` / ``TradingClockBasisYield``); one quoted per trading
year is read as is. The environment cannot tell the two apart from the curve, so it is declared per curve on
``PricingEnvironment`` (``rate_curve_quoting`` / ``div_yield_quoting`` / ``basis_yield_quoting``).
"""
from enum import Enum


class CurveQuotingBasis(Enum):
    """The year a cash curve's rates are quoted per."""

    #: Quoted per calendar year. On a BUSINESS_DAYS environment the environment wraps the curve onto its trading
    #: clock (needs ``trading_clock_horizon``); on any other environment the curve is read as is.
    CALENDAR_YEAR = "calendar_year"
    #: Quoted per trading year: read as is on a BUSINESS_DAYS environment; refused on any other environment,
    #: which has no trading clock to read it on.
    TRADING_YEAR = "trading_year"
