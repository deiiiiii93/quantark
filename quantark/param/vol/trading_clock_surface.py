"""Wrap a trading-time-quoted implied surface for calendar-axis engines.

Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md §4.2:
get_vol(K, tau_cal) = sigma_inner(K, tau_td) * sqrt(tau_td/tau_cal)
preserves total variance exactly: w_cal(tau) = w_td(to_trading(tau)).
``total_variance`` carries the FULL get_vol query signature so smile
surfaces reproduce the identical inner query, and computes w once from
tau_td so holiday plateaus difference to Dw == 0.0 exactly downstream.

Contract: engine times must be ACT/365 calendar fractions of real dates,
and the map's anchor_date must equal the environment's valuation_date.

``CalendarQuotedVolSurface`` (run 7, .ship/2026-11-26-replay-business-day-clock D2) is the opposite direction: a
calendar-quoted inner on the BUSINESS_DAYS trading axis, through a ``DateKnotTimeMap`` whose knots are the env times
of business dates (exact total variance at every business date).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

from quantark.param.vol.vol_surface import BlackImpliedVolSurface
from quantark.util.calendar import BusinessDayConvention, DayCountConvention
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class TradingClockVolSurface(BlackImpliedVolSurface):
    inner: BlackImpliedVolSurface
    time_map: BusinessTimeMap
    is_smile: bool = field(init=False, default=False)

    #: Explicit opt-in for the exact total-variance protocol in
    #: TermCoefficients.from_env — the marker (not the method name alone)
    #: gates the fast path, because ``total_variance`` also exists on
    #: SVIVolSurface with a different (strike, t) arity.
    exposes_exact_total_variance = True

    def __post_init__(self) -> None:
        if not isinstance(self.inner, BlackImpliedVolSurface):
            raise ValidationError("inner must be a BlackImpliedVolSurface")
        object.__setattr__(self, "is_smile", bool(self.inner.is_smile))

    def total_variance(self, strike: float, time_to_maturity, spot: float):
        """w_cal(tau_cal) = sigma_inner(K, tau_td)^2 * tau_td, vectorized."""
        tau_td = self.time_map.to_trading(time_to_maturity)
        arr = np.atleast_1d(np.asarray(tau_td, dtype=float))
        out = np.empty_like(arr)
        for j, u in enumerate(arr):
            if u <= 0.0:
                out[j] = 0.0
            else:
                v = float(self.inner.get_vol(float(strike), float(u), float(spot)))
                out[j] = v * v * u
        if np.ndim(tau_td) == 0:
            return float(out[0])
        return out

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        tau_cal = float(time_to_maturity)
        if tau_cal <= 0.0:
            slope = self.time_map.initial_slope()
            u = 1.0 / (365.0 * 10.0)  # one-tenth day in trading units for the inner lookup
            v = float(self.inner.get_vol(float(strike), u, float(spot)))
            return v * float(np.sqrt(slope))
        tau_td = float(self.time_map.to_trading(tau_cal))
        if tau_td <= 0.0:
            return 0.0  # pure-holiday horizon: zero accrued variance (documented)
        v = float(self.inner.get_vol(float(strike), tau_td, float(spot)))
        return v * float(np.sqrt(tau_td / tau_cal))

    def parallel_shifted(self, shift: float) -> "TradingClockVolSurface":
        """Shift the TRADING-quoted inner and re-expose it through the same map.

        The bump unit is one point of sigma_td (desk decision 2026-09-07,
        patch spec 2026-09-03 §14). The calendar-axis vol this surface returns
        then moves by ``shift * sqrt(tau_td/tau_cal)``, which is tenor-dependent:
        a constant shift of sigma_cal is not a parallel shift of anything the
        surface stores, and it would accrue ``2*shift*sqrt(w*tau_cal)`` of
        variance across a holiday, where the clock's defining property is that
        total variance is flat. The rate and dividend wrappers shift their
        (calendar-quoted) inners for the same reason.
        """
        return TradingClockVolSurface(self.inner.parallel_shifted(shift), self.time_map)

    def to_inner_time(self, time_to_maturity: float) -> float:
        """A time on THIS surface's axis (calendar) expressed on the inner's
        axis (trading). Lets a caller read the inner at one shared coordinate
        instead of letting each wrapper's own anchor pick a different point."""
        return float(self.time_map.to_trading(float(time_to_maturity)))

    def with_time_map(self, time_map: BusinessTimeMap) -> "TradingClockVolSurface":
        """The same inner surface re-expressed through another map (e.g. re-anchored)."""
        return TradingClockVolSurface(self.inner, time_map)

    def __repr__(self) -> str:
        return f"TradingClockVolSurface({self.inner!r}, {self.time_map!r})"


class DateKnotTimeMap(BusinessTimeMap):
    """Trading time -> calendar time with one knot per business DATE (run 7 spec D2).

    For every business date d in [anchor, horizon] the knot is (u_d, c_d): u_d is the env time of d exactly as
    ``quantark.priceenv.date_axis.env_times`` computes it on a BUSINESS_DAYS environment with this clock's calendar
    and ``bus_days_in_year`` (integer business-day count / D, the anchor rolled by ``holiday_time_adjustment``) and
    c_d = (d - anchor).days / 365 (ACT/365, the vol-quote axis). The anchor itself is the knot (0, 0).
    ``to_calendar`` is piecewise linear between consecutive knots, so it is exact at every business date, Mondays
    and the first day after a holiday included -- unlike ``BusinessTimeMap.to_calendar``, whose knots sit at the
    END of each trading day (a Monday maps back to the Saturday). ``to_trading`` is its exact inverse.

    It subclasses ``BusinessTimeMap`` so the environment's trading-clock machinery (``clock`` identity checks,
    ``re_anchored`` on a valuation-date move) treats it as the calendar trading clock it is; none of the parent's
    day-cell state is built or used.
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
        dates, d, end = [], a + timedelta(days=1), _midnight(horizon_date)
        while d <= end:
            if clock.calendar.is_business_day(d):
                dates.append(d)
            d += timedelta(days=1)
        if not dates:
            raise ValidationError(
                f"no business date of the clock's calendar in ({a.date()}, {end.date()}]; the date-knot map needs "
                "at least one business date after its anchor"
            )
        from quantark.priceenv.date_axis import env_times

        resolver = SimpleNamespace(
            valuation_date=a, day_count_convention=DayCountConvention.BUSINESS_DAYS, calendar=clock.calendar,
            holiday_time_adjustment=holiday_time_adjustment, bus_days_in_year=clock.days_per_year,
        )
        u = np.asarray(env_times(resolver, dates), dtype=float)
        if not u[0] > 0.0:
            # TODO: an anchor rolled FORWARD onto a later business date (FOLLOWING from a holiday) puts that date at
            # u = 0 while (d - anchor).days / 365 > 0 of calendar variance has accrued: not representable.
            raise ValidationError(
                f"anchor {a.date()} is a holiday rolled forward by {holiday_time_adjustment.name}: the business date "
                f"{dates[0].date()} sits at trading time 0 with {(dates[0] - a).days} calendar days of variance; a "
                "calendar-quoted vol cannot be placed on that clock"
            )
        self._dates = (a, *dates)
        self._u = np.r_[0.0, u]
        self._c = np.r_[0.0, [(x - a).days / 365.0 for x in dates]]
        self._u.setflags(write=False)
        self._c.setflags(write=False)

    @property
    def knot_dates(self) -> tuple:
        """The anchor (normalised to midnight) and every business date up to the horizon."""
        return self._dates

    @property
    def u_knots(self) -> np.ndarray:
        return self._u

    @property
    def c_knots(self) -> np.ndarray:
        return self._c

    def re_anchored(self, anchor_date: datetime) -> "DateKnotTimeMap":
        """The same clock and horizon seen from another valuation date: the knots are rebuilt from it."""
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


@dataclass(frozen=True)
class CalendarQuotedVolSurface(BlackImpliedVolSurface):
    """A CALENDAR-quoted implied surface presented on the TRADING (BUSINESS_DAYS) axis (run 7 spec D2).

    The opposite direction of ``TradingClockVolSurface``. With c(u) = ``time_map.to_calendar(u)`` (a
    ``DateKnotTimeMap``)::

        get_vol(K, u) = inner.get_vol(K, c(u)) * sqrt(c(u) / u)        u > 0
        get_vol(K, 0) = inner.get_vol(K, 0) * sqrt(dc/du at 0+)         the limit

    so sigma^2 * u = sigma_cal(K, c_d)^2 * c_d -- the calendar-clock total variance -- at the env time of every
    business date. Between business dates c(u) is linear (the exactness class: exact at dates). u < 0 or past the
    map's last business date raises.

    Contract: engine times are env times of a BUSINESS_DAYS environment whose calendar / bus_days_in_year are the
    map's clock and whose valuation_date is the map's anchor (the environment re-anchors the map itself).
    """

    inner: BlackImpliedVolSurface
    time_map: DateKnotTimeMap
    is_smile: bool = field(init=False, default=False)

    #: Same explicit opt-in as TradingClockVolSurface: engines read w = inner(K, c)^2 * c directly.
    exposes_exact_total_variance = True

    def __post_init__(self) -> None:
        if not isinstance(self.inner, BlackImpliedVolSurface):
            raise ValidationError("inner must be a BlackImpliedVolSurface")
        if not isinstance(self.time_map, DateKnotTimeMap):
            raise ValidationError(
                "time_map must be a DateKnotTimeMap (business-date knots); BusinessTimeMap.to_calendar lags past "
                "non-trading days"
            )
        object.__setattr__(self, "is_smile", bool(self.inner.is_smile))

    def total_variance(self, strike: float, time_to_maturity, spot: float):
        """w(u) = inner(K, c(u))^2 * c(u), vectorized; 0.0 at u = 0."""
        arr = np.atleast_1d(np.asarray(time_to_maturity, dtype=float))
        cs = np.atleast_1d(self.time_map.to_calendar(arr))
        out = np.empty_like(arr)
        for j, (u, c) in enumerate(zip(arr, cs)):
            if u == 0.0:
                out[j] = 0.0
            else:
                v = float(self.inner.get_vol(float(strike), float(c), float(spot)))
                out[j] = v * v * float(c)
        if np.ndim(time_to_maturity) == 0:
            return float(out[0])
        return out

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        u = float(time_to_maturity)
        if u < 0.0:
            raise ValidationError(f"time_to_maturity must be non-negative, got {u}")
        if u == 0.0:
            v = float(self.inner.get_vol(float(strike), 0.0, float(spot)))
            return v * float(np.sqrt(self.time_map.initial_slope()))
        c = float(self.time_map.to_calendar(u))
        v = float(self.inner.get_vol(float(strike), c, float(spot)))
        return v * float(np.sqrt(c / u))

    def parallel_shifted(self, shift: float) -> "CalendarQuotedVolSurface":
        """Shift the CALENDAR-quoted inner (one point of sigma_cal, the quote) and re-expose it through the same map."""
        return CalendarQuotedVolSurface(self.inner.parallel_shifted(shift), self.time_map)

    def to_inner_time(self, time_to_maturity: float) -> float:
        """A time on THIS surface's axis (trading) expressed on the inner's axis (calendar)."""
        return float(self.time_map.to_calendar(float(time_to_maturity)))

    def with_time_map(self, time_map: DateKnotTimeMap) -> "CalendarQuotedVolSurface":
        """The same inner surface re-expressed through another map (e.g. re-anchored)."""
        return CalendarQuotedVolSurface(self.inner, time_map)

    def __repr__(self) -> str:
        return f"CalendarQuotedVolSurface({self.inner!r}, {self.time_map!r})"


def _midnight(value: datetime) -> datetime:
    return datetime(value.year, value.month, value.day)
