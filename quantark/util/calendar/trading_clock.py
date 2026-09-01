"""Trading-clock ↔ calendar-clock time maps.

Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md §4.1.
``to_trading`` places VARIANCE: piecewise linear, slope (365/D) across a
trading day, exactly flat (stored knot value, no arithmetic) across
holidays. ``to_calendar`` places CARRY: the continuous piecewise-linear map
between consecutive trading-date knots — deliberately NOT the pointwise
inverse of ``to_trading``; the two agree exactly at trading-date knots.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from quantark.util.calendar.business_calendar import Calendar
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class TradingClock:
    """A calendar paired with its annualization denominator (244 SSE, 252 CFETS)."""

    calendar: Calendar
    days_per_year: int

    def __post_init__(self) -> None:
        if self.days_per_year <= 0:
            raise ValidationError(
                f"days_per_year must be positive, got {self.days_per_year}"
            )


class BusinessTimeMap:
    """Precomputed daily-knot map anchored at the valuation date."""

    def __init__(
        self,
        clock: TradingClock,
        anchor_date: datetime,
        horizon_date: datetime,
        extend_weekdays: bool = False,
    ) -> None:
        if horizon_date <= anchor_date:
            raise ValidationError("horizon_date must be after anchor_date")
        self.clock = clock
        self.anchor_date = anchor_date
        self.extend_weekdays = bool(extend_weekdays)
        n_days = (horizon_date - anchor_date).days
        inv_d = 1.0 / float(clock.days_per_year)

        # td_start[i] = trading time at the START of calendar day i;
        # is_td[i] = day i is a trading day. td_start has n_days+1 entries.
        is_td = np.zeros(n_days, dtype=bool)
        day = anchor_date
        for i in range(n_days):
            is_td[i] = clock.calendar.is_business_day(day)
            day += timedelta(days=1)
        td_start = np.concatenate(([0.0], np.cumsum(np.where(is_td, inv_d, 0.0))))
        self._is_td = is_td
        self._td_start = td_start
        self._n_days = n_days
        self._inv_d = inv_d
        # knots for to_calendar: (u_k, c_k) at the END of each trading day
        td_idx = np.nonzero(is_td)[0]
        self._u_knots = np.concatenate(([0.0], td_start[td_idx + 1]))
        self._c_knots = np.concatenate(([0.0], (td_idx + 1) / 365.0))

    def _day_frac(self, tau_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = tau_cal * 365.0
        i = np.floor(f).astype(int)
        return i, f - i

    def to_trading(self, tau_cal):
        """Calendar year fraction → trading year fraction (variance clock)."""
        arr = np.asarray(tau_cal, dtype=float)
        scalar = arr.ndim == 0
        a = np.atleast_1d(arr)
        if np.any(a < 0.0):
            raise ValidationError("tau_cal must be non-negative")
        i, frac = self._day_frac(a)
        over = i >= self._n_days
        if np.any(over):
            if not self.extend_weekdays:
                raise ValidationError(
                    "time beyond BusinessTimeMap horizon; extend the horizon or "
                    "construct with extend_weekdays=True"
                )
            raise ValidationError(
                "extend_weekdays horizon extension not yet implemented for this "
                "query range; extend horizon_date"
            )
        # trading day: knot + frac/D; holiday: the stored knot VALUE, no
        # arithmetic — plateau bitwise-equality contract (spec §4.2a).
        out = np.where(
            self._is_td[i],
            self._td_start[i] + frac * self._inv_d,
            self._td_start[i],
        )
        return float(out[0]) if scalar else out

    def to_calendar(self, tau_td):
        """Trading year fraction → calendar year fraction (carry clock)."""
        arr = np.asarray(tau_td, dtype=float)
        scalar = arr.ndim == 0
        a = np.atleast_1d(arr)
        if np.any(a < 0.0):
            raise ValidationError("tau_td must be non-negative")
        if np.any(a > self._u_knots[-1]):
            raise ValidationError("tau_td beyond BusinessTimeMap horizon")
        out = np.interp(a, self._u_knots, self._c_knots)
        return float(out[0]) if scalar else out

    def initial_slope(self) -> float:
        """d(to_trading)/d(tau_cal) at 0+ — 365/D on a trading day, else 0."""
        if self._n_days == 0 or not self._is_td[0]:
            return 0.0
        return 365.0 * self._inv_d

    def __repr__(self) -> str:
        return (
            f"BusinessTimeMap(D={self.clock.days_per_year}, "
            f"anchor={self.anchor_date.date()}, days={self._n_days}, "
            f"extend_weekdays={self.extend_weekdays})"
        )
