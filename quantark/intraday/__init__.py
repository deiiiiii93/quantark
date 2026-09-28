"""Intraday pricing: timezone-aware valuation timestamps, sub-day variance
profiles, provisional fixings and a structured result (design
docs/superpowers/specs/2026-09-15-intraday-pricing-design.md)."""
from quantark.intraday.batch import (
    IntradayFailure,
    IntradayPortfolioResult,
    SpotCurvePoint,
    aggregate_intraday,
    spot_curve,
    value_intraday_many,
)
from quantark.intraday.context import IntradayValuationContext, resolve_context
from quantark.intraday.events import EventKind, EventPhase
from quantark.intraday.fixings import AssumedFixing, ContinuousHistoryAssumption, Fixing
from quantark.intraday.profile import IntradayTimeMap, SegmentKind, VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.result import CashflowComponent, GreekValue, IntradayValuationResult
from quantark.intraday.roll import roll_context, roll_through_events
from quantark.intraday.service import value_intraday
from quantark.intraday.session import TradingSession, TradingSessionCalendar
from quantark.intraday.timestamp import (
    SECONDS_PER_YEAR,
    add_year_fraction,
    calendar_year_fraction,
    require_aware,
    same_instant,
    seconds_between,
    to_utc,
)

__all__ = [
    "AssumedFixing",
    "CashflowComponent",
    "ContinuousHistoryAssumption",
    "EventKind",
    "EventPhase",
    "Fixing",
    "GreekValue",
    "IntradayFailure",
    "IntradayPortfolioResult",
    "IntradayTimeMap",
    "IntradayValuationContext",
    "IntradayValuationRequest",
    "IntradayValuationResult",
    "SECONDS_PER_YEAR",
    "SegmentKind",
    "SpotCurvePoint",
    "TradingSession",
    "TradingSessionCalendar",
    "VarianceProfile",
    "add_year_fraction",
    "aggregate_intraday",
    "calendar_year_fraction",
    "require_aware",
    "resolve_context",
    "roll_context",
    "roll_through_events",
    "same_instant",
    "seconds_between",
    "spot_curve",
    "to_utc",
    "value_intraday",
    "value_intraday_many",
]
