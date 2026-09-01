"""Calendar-quoted curve re-expressed in trading time.

Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md §4.3.
DF_td(u) = DF_cal(to_calendar(u)). ``to_calendar`` is the CONTINUOUS
knot-to-knot map, so a holiday's carry spreads across its adjacent trading
tick instead of appearing as a jump (review iter-2 P2a). DF ratios
telescope exactly; the D5 invariant (same DF for the same date) holds at
every trading-date knot.
"""
from __future__ import annotations

import math

from quantark.param.rrf.rate_curve import RateCurve
from quantark.util.calendar.trading_clock import BusinessTimeMap
from quantark.util.exceptions import ValidationError


class TradingClockRateCurve(RateCurve):
    """Expose a calendar-quoted RateCurve on the trading-time axis."""

    def __init__(self, inner: RateCurve, time_map: BusinessTimeMap) -> None:
        if not isinstance(inner, RateCurve):
            raise ValidationError("inner must be a RateCurve")
        self.inner = inner
        self.time_map = time_map
        self.clock = time_map.clock

    def get_discount_factor(self, time_to_maturity: float) -> float:
        if time_to_maturity < 0:
            raise ValidationError(
                f"Time to maturity must be non-negative, got {time_to_maturity}"
            )
        if time_to_maturity == 0.0:
            return 1.0
        c = float(self.time_map.to_calendar(float(time_to_maturity)))
        return self.inner.get_discount_factor(c)

    def get_rate(self, time_to_maturity: float) -> float:
        if time_to_maturity <= 0.0:
            raise ValidationError(
                f"Time to maturity must be positive, got {time_to_maturity}"
            )
        df = self.get_discount_factor(time_to_maturity)
        return -math.log(df) / float(time_to_maturity)

    def __repr__(self) -> str:
        return f"TradingClockRateCurve({self.inner!r}, {self.time_map!r})"
