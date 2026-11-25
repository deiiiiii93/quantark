"""Cumulative-preserving futures-basis wrapper for the trading axis (0.15.0).

A futures price is ``F = S·exp(b(T)·T)``: what matters is the cumulative basis ``b(T)·T``, exactly as the engines
difference ``q(t)·t`` for the dividend yield. An argument-only remap would accrue ``b_cal(c)·u`` instead of the
true ``b_cal(c)·c``, so this wrapper follows ``TradingClockDividendYield``: ``b_td(u) = b_cal(c)·c/u`` with
``c = to_calendar(u)``, and 0 at ``u = 0`` (where ``b·u`` is 0 whatever ``b`` is).
"""
from __future__ import annotations

from quantark.param.basis.basis_yield import BasisYield
from quantark.util.calendar.trading_clock import BusinessTimeMap
from quantark.util.exceptions import ValidationError


class TradingClockBasisYield(BasisYield):
    """Expose a calendar-quoted BasisYield on the trading-time axis."""

    def __init__(self, inner: BasisYield, time_map: BusinessTimeMap) -> None:
        if not isinstance(inner, BasisYield):
            raise ValidationError("inner must be a BasisYield")
        self.inner = inner
        self.time_map = time_map
        self.clock = time_map.clock

    def get_basis_yield(self, time_to_maturity: float) -> float:
        u = float(time_to_maturity)
        if u <= 0.0:
            return 0.0
        c = float(self.time_map.to_calendar(u))
        return self.inner.get_basis_yield(c) * c / u

    def to_inner_time(self, time_to_maturity: float) -> float:
        """A time on THIS basis's axis (trading) expressed on the inner's axis (calendar)."""
        return float(self.time_map.to_calendar(float(time_to_maturity)))

    def with_time_map(self, time_map: BusinessTimeMap) -> "TradingClockBasisYield":
        """The same inner basis re-expressed through another map (e.g. re-anchored)."""
        return TradingClockBasisYield(self.inner, time_map)

    def __repr__(self) -> str:
        return f"TradingClockBasisYield({self.inner!r}, {self.time_map!r})"
