"""The pricing environment's date axis: where a calendar date sits on the env's clock.

``env_time(env, date, anchor=None)`` is the signed year fraction from ``anchor`` (default ``env.valuation_date``) to
``date`` on the environment's clock -- the same rule dated observation records use
(``calculate_year_fraction(anchor, date, env.day_count_convention, env.bus_days_in_year, calendar=env.calendar)``),
extended to past dates (negative) and to the anchor itself (``0.0``).

Calendar-day conventions (``CALENDAR_DAYS``, ``ACT_365``, ``ACT_360``, ...) call ``calculate_year_fraction`` per date,
so ``CALENDAR_DAYS`` / ``ACT_365`` are ``num_days / 365.0`` bit for bit, and a past date is the exact negation.

``BUSINESS_DAYS``: a date that is a holiday of ``env.calendar`` has no time of its own; it is placed by
``env.holiday_time_adjustment``. Both ends are rolled (``adj``, the identity on a business day) and the time is the
signed count of env business days between them over ``bus_days_in_year``::

    t(d) = +#bd(adj(a), adj(d)] / D    if adj(d) >= adj(a)
         = -#bd(adj(d), adj(a)] / D    otherwise

PRECEDING is plain business-day counting (it equals ``calculate_year_fraction(a, d, BUSINESS_DAYS)`` for every
``d > a``); every convention value gives times non-decreasing in the date, ``0.0`` for a date in the same holiday
run as a holiday anchor, and ``t <= 0`` for a past date.

Dates are normalised to calendar dates (midnight) on both sides first; a time of day never moves a date's time.
Spec: ``.ship/2026-11-24-dcn-time-axis/spec.md`` D9 / D13 / D14 / D20.
"""
from __future__ import annotations

from datetime import date as _date
from datetime import datetime, timedelta
from typing import Iterable, Optional

import numpy as np

from quantark.util.calendar import DayCountConvention
from quantark.util.calendar.day_counter import calculate_year_fraction
from quantark.util.exceptions import ValidationError

__all__ = ["env_time", "env_times"]


def _day(value) -> datetime:
    """The calendar date of ``value`` as a naive midnight ``datetime``."""
    if isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, _date):
        return datetime(value.year, value.month, value.day)
    raise ValidationError(f"a date is required, got {type(value).__name__}")


def _calendar_day_times(pricing_env, days, anchor: datetime) -> np.ndarray:
    convention = pricing_env.day_count_convention
    out = np.empty(len(days), dtype=float)
    for i, d in enumerate(days):
        if d > anchor:
            out[i] = calculate_year_fraction(anchor, d, convention, pricing_env.bus_days_in_year)
        elif d < anchor:
            out[i] = -calculate_year_fraction(d, anchor, convention, pricing_env.bus_days_in_year)
        else:
            out[i] = 0.0
    return out


def _business_day_times(pricing_env, days, anchor: datetime) -> np.ndarray:
    cal = pricing_env.calendar
    if cal is None or not hasattr(cal, "is_business_day") or not hasattr(cal, "adjust_date"):
        raise ValidationError(
            "BUSINESS_DAYS needs a business-day calendar with is_business_day() and adjust_date() on the "
            "pricing environment"
        )
    convention = pricing_env.holiday_time_adjustment
    cache = {}

    def adj(d: datetime) -> datetime:
        if d not in cache:
            cache[d] = d if cal.is_business_day(d) else _day(cal.adjust_date(d, convention))
        return cache[d]

    a = adj(anchor)
    rolled = [adj(d) for d in days]
    lo = min([a] + rolled)
    hi = max([a] + rolled)
    # count[k] = env business days in (lo, lo + k days]
    n = (hi - lo).days
    count = np.zeros(n + 1, dtype=np.int64)
    running = 0
    day = lo
    for k in range(1, n + 1):
        day += timedelta(days=1)
        if cal.is_business_day(day):
            running += 1
        count[k] = running
    base = int(count[(a - lo).days])
    per_year = float(pricing_env.bus_days_in_year)
    out = np.empty(len(days), dtype=float)
    for i, r in enumerate(rolled):
        out[i] = (int(count[(r - lo).days]) - base) / per_year
    return out


def env_times(pricing_env, dates: Iterable, anchor: Optional[datetime] = None) -> np.ndarray:
    """Signed env-axis year fractions from ``anchor`` (default ``pricing_env.valuation_date``) to each date."""
    days = [_day(d) for d in dates]
    a = _day(pricing_env.valuation_date if anchor is None else anchor)
    if not days:
        return np.empty(0, dtype=float)
    if pricing_env.day_count_convention == DayCountConvention.BUSINESS_DAYS:
        return _business_day_times(pricing_env, days, a)
    return _calendar_day_times(pricing_env, days, a)


def env_time(pricing_env, date, anchor: Optional[datetime] = None) -> float:
    """Signed env-axis year fraction from ``anchor`` (default ``pricing_env.valuation_date``) to ``date``."""
    return float(env_times(pricing_env, [date], anchor)[0])
