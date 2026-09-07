"""Cumulative-yield-preserving dividend wrapper for the trading axis.

Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md §4.3
(review iter-1 P1). ``forward_carry_on_grid`` differences q(t)*t, so an
argument-only remap would accrue q_cal(c)*u instead of the true q_cal(c)*c.
Define q_td(u) = q_cal(c)*c/u so that q_td(u)*u == q_cal(c)*c for u > 0;
0 at u = 0 (the forward_carry_on_grid convention at t = 0). The rate-curve
wrapper needs no such correction because it is DF-based — cumulative by
construction.
"""
from __future__ import annotations

from quantark.param.div.dividend_yield import DividendYield
from quantark.util.calendar.trading_clock import BusinessTimeMap
from quantark.util.exceptions import ValidationError


class TradingClockDividendYield(DividendYield):
    """Expose a calendar-quoted DividendYield on the trading-time axis."""

    def __init__(self, inner: DividendYield, time_map: BusinessTimeMap) -> None:
        if not isinstance(inner, DividendYield):
            raise ValidationError("inner must be a DividendYield")
        self.inner = inner
        self.time_map = time_map
        self.clock = time_map.clock

    def get_yield(self, time_to_maturity: float) -> float:
        u = float(time_to_maturity)
        if u <= 0.0:
            return 0.0
        c = float(self.time_map.to_calendar(u))
        return self.inner.get_yield(c) * c / u

    def parallel_shifted(self, shift: float) -> "TradingClockDividendYield":
        """Shift the CALENDAR-quoted inner (q lives on the calendar clock) and
        re-expose it through the same map."""
        return TradingClockDividendYield(self.inner.parallel_shifted(shift), self.time_map)

    def to_inner_time(self, time_to_maturity: float) -> float:
        """A time on THIS yield's axis (trading) expressed on the inner's axis
        (calendar). Lets a caller read the inner at one shared coordinate
        instead of letting each wrapper's own anchor pick a different point."""
        return float(self.time_map.to_calendar(float(time_to_maturity)))

    def with_time_map(self, time_map: BusinessTimeMap) -> "TradingClockDividendYield":
        """The same inner yield re-expressed through another map (e.g. re-anchored)."""
        return TradingClockDividendYield(self.inner, time_map)

    def __repr__(self) -> str:
        return f"TradingClockDividendYield({self.inner!r}, {self.time_map!r})"
