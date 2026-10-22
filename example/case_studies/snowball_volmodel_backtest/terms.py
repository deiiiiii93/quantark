"""Term sheet, trading calendar, and inception scheduler for the snowball
vol-model backtest fleet (plan Task 4.1).

The production term sheet is the locked 3Y snowball on 000852.SH
(docs/superpowers/specs/2026-07-23-snowball-volmodel-backtest-design.md):

- SELLER (short the snowball; the backtest config carries
  ``product_quantity = -1.0``), notional 50,000,000 CNY,
- KO 103% of the inception spot, monthly observations from month 3 through
  month 36 (34 observations, trading-calendar snapped, last == maturity),
- KI 75% of the inception spot, daily-discrete monitoring on every trading
  day in (inception, maturity] from the spot calendar,
- standard snowball payoff (``create_standard_snowball``,
  ``include_principal=False``), ko_rate == rebate_rate == solved fair coupon,
- r = 0.02 flat (fleet config, not data).

The product construction mirrors the PDE convergence gate
(``example/mo_volmodels/11_pde_convergence_gate.py``) exactly: same calendar
rules, same ACT/365 time fractions, same ``create_standard_snowball`` call.
Barriers/strike/notional are fixed off the INCEPTION spot from the spot CSV
(the contractual index fixing — the same series the lifecycle tracker
compares against), with ``contract_multiplier = notional / s0`` so one
product unit carries the full 50mio book.
"""

from __future__ import annotations

import calendar as _calendar_mod
import os
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.util.enum import ObservationType
from quantark.util.numerical import is_close

# ---------------------------------------------------------------------------
# Locked term-sheet constants (design doc section 2)
# ---------------------------------------------------------------------------

UNDERLYING_SYMBOL = "000852.SH"
UNDERLYING_NAME = "CSI1000"
NOTIONAL = 50_000_000.0
FLAT_RATE = 0.02
MATURITY_MONTHS = 36
LOCKOUT_MONTHS = 3
KO_PCT = 1.03
KI_PCT = 0.75
ACT = 365.0  # year-fraction convention for calendar-derived observation dates
PRODUCT_QUANTITY = -1.0  # SELLER

FIRST_INCEPTION_MONTH = (2023, 5)  # first month with spot data

SCHEMA_VERSION = 1


class SnowballFleetError(RuntimeError):
    """Raised when the snowball vol-model fleet cannot proceed (fail-closed)."""


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def add_months(d: date, months: int) -> date:
    """Add calendar months, clamping the day to the target month's length."""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, _calendar_mod.monthrange(y, m)[1])
    return date(y, m, day)


class TradingCalendar:
    """Trading-day calendar from the history's spot CSV.

    Beyond the last CSV date the calendar extends with plain weekdays
    (holiday schedule unknown); consumers record the extension.  Mirrors the
    gate's calendar (``11_pde_convergence_gate.py``) so KO/KI date
    generation is identical between the gate and the fleet.
    """

    def __init__(self, days: Sequence[date]) -> None:
        if not days:
            raise SnowballFleetError("TradingCalendar needs at least one day")
        self._days = sorted(set(days))
        self._set = set(self._days)

    @classmethod
    def from_spot_csv(cls, path: os.PathLike | str) -> "TradingCalendar":
        days: List[date] = []
        with open(path, "r", encoding="utf-8") as handle:
            header = handle.readline()
            if not header.lower().startswith("date"):
                raise SnowballFleetError(f"spot CSV {path} missing 'date' header")
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                days.append(datetime.strptime(line.split(",")[0], "%Y-%m-%d").date())
        return cls(days)

    @property
    def first(self) -> date:
        return self._days[0]

    @property
    def last(self) -> date:
        return self._days[-1]

    def __len__(self) -> int:
        return len(self._days)

    def is_trading_day(self, d: date) -> bool:
        if d in self._set:
            return True
        return d > self._days[-1] and d.weekday() < 5

    def next_trading_day(self, d: date) -> date:
        """First trading day on or after ``d`` (weekday extension past the CSV)."""
        i = bisect_left(self._days, d)
        if i < len(self._days):
            return self._days[i]
        cur = d
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        return cur

    def trading_days_between(self, start: date, end: date) -> List[date]:
        """All trading days t with ``start < t <= end``."""
        out: List[date] = []
        cur = start
        while cur < end:
            cur += timedelta(days=1)
            if self.is_trading_day(cur):
                out.append(cur)
        return out


def parse_yyyymmdd(value: str) -> date:
    """Parse ``YYYYMMDD`` or ``YYYY-MM-DD`` (fail-closed)."""
    value = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise SnowballFleetError(
        f"cannot parse date {value!r} (expected YYYYMMDD or YYYY-MM-DD)"
    )


# ---------------------------------------------------------------------------
# Term sheet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnowballTerms:
    """Production snowball term sheet, as year fractions from ``inception``."""

    inception: date
    maturity_date: date
    maturity_years: float
    ko_times: Tuple[float, ...]  # monthly, first at ~month 3, last == maturity
    ki_times: Tuple[float, ...]  # daily-discrete, from the trading calendar
    ko_pct: float = KO_PCT
    ki_pct: float = KI_PCT
    notional: float = NOTIONAL

    def summary(self) -> Dict[str, Any]:
        return {
            "inception": self.inception.isoformat(),
            "maturity_date": self.maturity_date.isoformat(),
            "maturity_years": self.maturity_years,
            "n_ko": len(self.ko_times),
            "n_ki": len(self.ki_times),
            "first_ko_time": self.ko_times[0] if self.ko_times else None,
            "ko_pct": self.ko_pct,
            "ki_pct": self.ki_pct,
            "notional": self.notional,
        }


def build_snowball_terms(
    inception: date,
    calendar: TradingCalendar,
    *,
    maturity_months: int = MATURITY_MONTHS,
    lockout_months: int = LOCKOUT_MONTHS,
    ko_pct: float = KO_PCT,
    ki_pct: float = KI_PCT,
    notional: float = NOTIONAL,
) -> SnowballTerms:
    """Build the production term sheet on ``inception``.

    KO: month anniversaries lockout..maturity snapped to the next trading
    day (34 dates at the production 36/3).  KI: every trading day in
    (inception, maturity].  Times are ACT/365 year fractions from inception.
    """
    if lockout_months < 1 or lockout_months > maturity_months:
        raise SnowballFleetError(
            f"lockout_months {lockout_months} must be in [1, {maturity_months}]"
        )
    if inception < calendar.first or inception > calendar.last:
        raise SnowballFleetError(
            f"inception {inception.isoformat()} outside the spot calendar "
            f"[{calendar.first.isoformat()}, {calendar.last.isoformat()}]"
        )
    maturity_date = calendar.next_trading_day(add_months(inception, maturity_months))
    ko_dates: List[date] = []
    for m in range(lockout_months, maturity_months + 1):
        kd = calendar.next_trading_day(add_months(inception, m))
        if kd > maturity_date:
            kd = maturity_date
        ko_dates.append(kd)
    ko_dates = sorted(set(ko_dates))
    ki_days = calendar.trading_days_between(inception, maturity_date)
    maturity_years = (maturity_date - inception).days / ACT
    ko_times = tuple((d - inception).days / ACT for d in ko_dates)
    ki_times = tuple((d - inception).days / ACT for d in ki_days)
    if not ko_times or not is_close(
        ko_times[-1], maturity_years, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SnowballFleetError("last KO observation must coincide with maturity")
    if not ki_times or ki_times[-1] > maturity_years + 1e-12:
        raise SnowballFleetError("KI schedule must end at or before maturity")
    return SnowballTerms(
        inception=inception,
        maturity_date=maturity_date,
        maturity_years=maturity_years,
        ko_times=ko_times,
        ki_times=ki_times,
        ko_pct=ko_pct,
        ki_pct=ki_pct,
        notional=notional,
    )


def build_snowball_product(terms: SnowballTerms, s0: float, coupon: float):
    """Create the SnowballOption: barriers/strike/notional off the INCEPTION spot.

    ``contract_multiplier = notional / s0`` sizes one product unit to the
    full 50mio book; the backtest then shorts one unit
    (``product_quantity = -1.0``).  ``coupon`` is the solved fair coupon,
    applied to both ``ko_rate`` and ``rebate_rate`` (standard snowball).
    """
    s0 = float(s0)
    if s0 <= 0.0:
        raise SnowballFleetError(f"s0 must be positive, got {s0}")
    product = create_standard_snowball(
        initial_price=s0,
        strike=s0,
        maturity=float(terms.maturity_years),
        contract_multiplier=float(terms.notional) / s0,
        ko_barrier=terms.ko_pct * s0,
        ko_rate=float(coupon),
        ki_barrier=terms.ki_pct * s0,
        num_observations=len(terms.ko_times),
        is_reverse=False,
        ko_observation_dates=list(terms.ko_times),
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(terms.ki_times),
        rebate_rate=float(coupon),
        include_principal=False,
    )
    # Contractual fixing date: lets the lifecycle tracker resolve schedule
    # dates from the inception instead of relying on the run start date.
    product.initial_date = datetime.combine(terms.inception, datetime.min.time())
    return product


# ---------------------------------------------------------------------------
# Inception scheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InceptionSchedule:
    """One monthly inception: trade window [inception, trade_end] + censor flag."""

    inception: date
    maturity_date: date
    trade_end: date
    censored: bool  # maturity beyond the data window (trade ends at data end)

    def summary(self) -> Dict[str, Any]:
        return {
            "inception": self.inception.isoformat(),
            "maturity_date": self.maturity_date.isoformat(),
            "trade_end": self.trade_end.isoformat(),
            "censored": self.censored,
        }


def enumerate_inceptions(
    calendar: TradingCalendar,
    *,
    first_month: Tuple[int, int] = FIRST_INCEPTION_MONTH,
    data_end: Optional[date] = None,
    maturity_months: int = MATURITY_MONTHS,
) -> List[InceptionSchedule]:
    """Monthly inceptions: first trading day of each month with data.

    Runs from ``first_month`` through the last month whose first trading day
    is inside the data window (every such inception can still trade at
    least one day; trades run to KO/maturity or censor at ``data_end``).
    """
    end = calendar.last if data_end is None else min(data_end, calendar.last)
    start_year, start_month = first_month
    if date(start_year, start_month, 1) > end:
        raise SnowballFleetError(
            f"first inception month {start_year}-{start_month:02d} is past the "
            f"data window end {end.isoformat()}"
        )
    first_of_month: Dict[Tuple[int, int], date] = {}
    for d in calendar._days:
        if d > end:
            break
        first_of_month.setdefault((d.year, d.month), d)
    schedules: List[InceptionSchedule] = []
    cursor = date(start_year, start_month, 1)
    while cursor <= end:
        inception = first_of_month.get((cursor.year, cursor.month))
        if inception is not None:
            maturity_date = calendar.next_trading_day(
                add_months(inception, maturity_months)
            )
            schedules.append(
                InceptionSchedule(
                    inception=inception,
                    maturity_date=maturity_date,
                    trade_end=min(maturity_date, end),
                    censored=maturity_date > end,
                )
            )
        cursor = add_months(cursor, 1)
    if not schedules:
        raise SnowballFleetError(
            f"no inceptions inside the data window ending {end.isoformat()}"
        )
    return schedules
