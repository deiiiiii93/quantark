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
from types import SimpleNamespace

import numpy as np

from quantark.util.calendar.business_calendar import BusinessDayConvention, Calendar
from quantark.util.calendar.day_counter import DayCountConvention
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
        self.horizon_date = horizon_date
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
        self._u_knots.setflags(write=False)
        self._c_knots.setflags(write=False)

    @property
    def u_knots(self) -> np.ndarray:
        """Trading-time knots of ``to_calendar`` (read-only); the last one is the furthest readable trading time."""
        return self._u_knots

    def re_anchored(self, anchor_date: datetime) -> "BusinessTimeMap":
        """The same clock and horizon seen from another valuation date (patch spec 2026-09-03 §8)."""
        return BusinessTimeMap(self.clock, anchor_date, self.horizon_date, extend_weekdays=self.extend_weekdays)

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


class DateKnotTimeMap(BusinessTimeMap):
    """Trading time -> calendar time with one knot per business DATE (run 7 spec D2; run 8 D1 moved it here).

    For every business date d after the anchor, up to and including the FIRST business date on or after the horizon
    (run 8 R1: it always reaches at least as far as ``BusinessTimeMap``), the knot is (u_d, c_d): u_d is the env time
    of d exactly as ``quantark.priceenv.date_axis.env_times`` computes it on a BUSINESS_DAYS environment with this
    clock's calendar and ``bus_days_in_year`` (integer business-day count / D, the anchor rolled by
    ``holiday_time_adjustment``) and c_d = (d - anchor).days / 365 (ACT/365, the calendar quote axis). The anchor
    itself is the knot (0, 0); knots are whole calendar dates (the anchor is normalised to midnight, like the env
    axis). ``to_calendar`` is piecewise linear between consecutive knots, so it is exact at every business date,
    Mondays and the first day after a holiday included -- unlike ``BusinessTimeMap.to_calendar``, whose knots sit
    at the END of each trading day (a Monday maps back to the Saturday). ``to_trading`` is its exact inverse. A
    read past the last knot raises.

    It is the map of a ``CalendarQuotedVolSurface`` and, on a BUSINESS_DAYS environment, of every CALENDAR_YEAR-
    declared cash curve (run 8 D2). It subclasses ``BusinessTimeMap`` so the environment's trading-clock machinery
    (``clock`` identity checks, ``re_anchored`` on a valuation-date move) treats it as the calendar trading clock
    it is; none of the parent's day-cell state is built or used (``initial_slope`` / ``to_trading`` have the
    opposite meaning: they are the calendar-per-trading slope and the inverse of ``to_calendar``).
    """

    def __init__(
        self,
        clock: TradingClock,
        anchor_date: datetime,
        horizon_date: datetime,
        holiday_time_adjustment: BusinessDayConvention = BusinessDayConvention.PRECEDING,
    ) -> None:
        if horizon_date <= anchor_date:
            raise ValidationError("horizon_date must be after anchor_date")
        self.clock = clock
        self.anchor_date = anchor_date
        self.horizon_date = horizon_date
        self.extend_weekdays = False
        self.holiday_time_adjustment = holiday_time_adjustment
        a = _midnight(anchor_date)
        end = _midnight(horizon_date)
        limit = end + timedelta(days=_REACH_SEARCH_DAYS)
        dates, d = [], a + timedelta(days=1)
        while True:
            if d > limit:
                raise ValidationError(
                    f"no business date of the clock's calendar in ({end.date()}, {limit.date()}]; the date-knot "
                    "map needs a business date on or after its horizon"
                )
            if clock.calendar.is_business_day(d):
                dates.append(d)
                if d >= end:
                    break
            d += timedelta(days=1)
        # function-local: date_axis -> quantark.util.calendar -> this module would be an import cycle (run 8 R6)
        from quantark.priceenv.date_axis import env_times

        resolver = SimpleNamespace(
            valuation_date=a, day_count_convention=DayCountConvention.BUSINESS_DAYS, calendar=clock.calendar,
            holiday_time_adjustment=holiday_time_adjustment, bus_days_in_year=clock.days_per_year,
        )
        u = np.asarray(env_times(resolver, dates), dtype=float)
        if not u[0] > 0.0:
            # TODO: an anchor rolled FORWARD onto a later business date (FOLLOWING from a holiday) puts that date at
            # u = 0 while (d - anchor).days / 365 > 0 of calendar time (carry, variance) has accrued: not
            # representable on this clock.
            raise ValidationError(
                f"anchor {a.date()} is a holiday rolled forward by {holiday_time_adjustment.name}: the business date "
                f"{dates[0].date()} sits at trading time 0 with {(dates[0] - a).days} calendar days of carry and "
                "variance; a calendar-quoted curve or vol cannot be placed on that clock (value on a business date "
                "or use holiday_time_adjustment=PRECEDING)"
            )
        self._dates = (a, *dates)
        self._u = np.r_[0.0, u]
        self._c = np.r_[0.0, [(x - a).days / 365.0 for x in dates]]
        self._u.setflags(write=False)
        self._c.setflags(write=False)

    @property
    def knot_dates(self) -> tuple:
        """The anchor (normalised to midnight) and every business date up to the first one on or after the horizon."""
        return self._dates

    @property
    def u_knots(self) -> np.ndarray:
        return self._u

    @property
    def c_knots(self) -> np.ndarray:
        return self._c

    def re_anchored(self, anchor_date: datetime) -> "DateKnotTimeMap":
        """The same clock, horizon and holiday convention seen from another valuation date: knots rebuilt from it."""
        return DateKnotTimeMap(self.clock, anchor_date, self.horizon_date, self.holiday_time_adjustment)

    def to_calendar(self, tau_td):
        """Trading year fraction -> calendar (ACT/365) year fraction, linear between business-date knots."""
        arr = np.asarray(tau_td, dtype=float)
        a = np.atleast_1d(arr)
        if np.any(a < 0.0):
            raise ValidationError("tau_td must be non-negative")
        if np.any(a > self._u[-1]):
            raise ValidationError(
                f"tau_td beyond the date-knot map's last business date {self._dates[-1].date()} "
                f"(horizon {self.horizon_date}); extend the horizon"
            )
        out = np.interp(a, self._u, self._c)
        return float(out[0]) if arr.ndim == 0 else out

    def to_trading(self, tau_cal):
        """Calendar year fraction -> trading year fraction: the exact inverse of ``to_calendar``."""
        arr = np.asarray(tau_cal, dtype=float)
        a = np.atleast_1d(arr)
        if np.any(a < 0.0):
            raise ValidationError("tau_cal must be non-negative")
        if np.any(a > self._c[-1]):
            raise ValidationError(
                f"tau_cal beyond the date-knot map's last business date {self._dates[-1].date()} "
                f"(horizon {self.horizon_date}); extend the horizon"
            )
        out = np.interp(a, self._c, self._u)
        return float(out[0]) if arr.ndim == 0 else out

    def initial_slope(self) -> float:
        """d(to_calendar)/d(tau_td) at 0+: calendar time per trading time to the first business date."""
        return float(self._c[1] / self._u[1])

    def __repr__(self) -> str:
        return (
            f"DateKnotTimeMap(D={self.clock.days_per_year}, anchor={self.anchor_date.date()}, "
            f"horizon={self.horizon_date.date()}, knots={len(self._dates)})"
        )


#: How far past its horizon a ``DateKnotTimeMap`` looks for its last business date before refusing.
_REACH_SEARCH_DAYS = 366


def _midnight(value: datetime) -> datetime:
    return datetime(value.year, value.month, value.day)
