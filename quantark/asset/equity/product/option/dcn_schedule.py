"""DCN observation-schedule generator (spec WP1.1).

Rules fixed by the problem: U_k = initial_date + k months (relativedelta
keep-day-of-month with end-of-month clamp); O_k = Following(U_k) on SSE;
coincident adjusted dates merge (month indices kept, flags OR'd); payment
dates via Calendar.trading_days_after; fail loudly on maturity mismatch and
on calendar-coverage overrun.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

from dateutil.relativedelta import relativedelta

from quantark.util.calendar import BusinessDayConvention, Calendar
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class DCNMonthlyObservation:
    month_indices: Tuple[int, ...]
    benchmark_date: datetime
    observation_date: datetime
    is_coupon_obs: bool
    is_ko_obs: bool
    coupon_payment_date: datetime
    ko_payment_date: datetime


@dataclass(frozen=True)
class DCNScheduleSpec:
    """Everything needed to rebuild the schedule (e.g. for date-rolled theta:
    ``dataclasses.replace(spec, valuation_date=rolled).build(calendar)``)."""

    initial_date: datetime
    maturity_date: datetime
    tenor_months: int
    lock_months: int
    ko_lock_months: int
    coupon_settlement_offset: int
    ko_settlement_offset: int
    valuation_date: datetime
    settlement_date: datetime

    def build(self, calendar: Calendar) -> "DCNSchedule":
        return build_dcn_schedule(calendar=calendar, **self.__dict__)


@dataclass(frozen=True)
class DCNSchedule:
    monthly: Tuple[DCNMonthlyObservation, ...]
    daily_ki_dates: Tuple[datetime, ...]
    valuation_date: datetime
    maturity_date: datetime
    spec: DCNScheduleSpec
    calendar: Calendar

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "month_indices": list(m.month_indices),
                    "benchmark_date": m.benchmark_date,
                    "observation_date": m.observation_date,
                    "is_coupon_obs": m.is_coupon_obs,
                    "is_ko_obs": m.is_ko_obs,
                    "coupon_payment_date": m.coupon_payment_date,
                    "ko_payment_date": m.ko_payment_date,
                }
                for m in self.monthly
            ]
        )

    def daily_ki_dataframe(self):
        import pandas as pd

        return pd.DataFrame({"ki_observation_date": list(self.daily_ki_dates)})


def _benchmark_dates(
    initial_date: datetime, lock_months: int, tenor_months: int
) -> List[Tuple[int, datetime]]:
    """[(k, U_k)] for k = lock_months .. tenor_months (relativedelta EOM clamp)."""
    return [
        (k, initial_date + relativedelta(months=k))
        for k in range(lock_months, tenor_months + 1)
    ]


def _assert_calendar_coverage(calendar: Calendar, last_needed: datetime) -> None:
    holidays = getattr(calendar, "holidays", None)
    if not holidays:
        return  # weekend-only calendar covers all dates
    max_year = max(h.year for h in holidays)
    if last_needed.year > max_year:
        raise ValidationError(
            f"schedule requires dates through {last_needed:%Y-%m-%d} but the "
            f"calendar holiday file covers only through {max_year}-12-31"
        )


def build_dcn_schedule(
    initial_date: datetime,
    maturity_date: datetime,
    tenor_months: int,
    lock_months: int,
    ko_lock_months: int,
    coupon_settlement_offset: int,
    ko_settlement_offset: int,
    valuation_date: datetime,
    settlement_date: datetime,
    calendar: Calendar,
) -> DCNSchedule:
    if not (0 < lock_months <= ko_lock_months <= tenor_months):
        raise ValidationError(
            "require 0 < lock_months <= ko_lock_months <= tenor_months, got "
            f"{lock_months}/{ko_lock_months}/{tenor_months}"
        )
    if valuation_date < initial_date:
        raise ValidationError("valuation_date before initial_date")

    _assert_calendar_coverage(
        calendar, max(maturity_date, settlement_date) + timedelta(days=31)
    )

    # 1) benchmark -> adjusted observation dates (merge coincident dates)
    rows: dict = {}
    for k, u_k in _benchmark_dates(initial_date, lock_months, tenor_months):
        o_k = calendar.adjust_date(u_k, BusinessDayConvention.FOLLOWING)
        row = rows.setdefault(o_k, {"ks": [], "benchmark": u_k})
        row["ks"].append(k)

    # 2) maturity consistency — problem forbids silent override
    last_obs = max(rows)
    if last_obs != maturity_date:
        raise ValidationError(
            f"generated final observation {last_obs:%Y-%m-%d} != contract "
            f"maturity_date {maturity_date:%Y-%m-%d}"
        )

    # 3) monthly rows (only those on/after valuation_date enter the schedule)
    monthly: List[DCNMonthlyObservation] = []
    for o_k in sorted(rows):
        if o_k < valuation_date:
            continue
        ks = tuple(sorted(rows[o_k]["ks"]))
        monthly.append(
            DCNMonthlyObservation(
                month_indices=ks,
                benchmark_date=rows[o_k]["benchmark"],
                observation_date=o_k,
                is_coupon_obs=any(k >= lock_months for k in ks),
                is_ko_obs=any(k >= ko_lock_months for k in ks),
                coupon_payment_date=calendar.trading_days_after(
                    o_k, coupon_settlement_offset
                ),
                ko_payment_date=calendar.trading_days_after(
                    o_k, ko_settlement_offset
                ),
            )
        )

    # 4) daily KI grid: every SSE trading day in [valuation_date, maturity_date]
    daily: List[datetime] = []
    d = valuation_date
    while d <= maturity_date:
        if calendar.is_business_day(d):
            daily.append(d)
        d += timedelta(days=1)

    return DCNSchedule(
        monthly=tuple(monthly),
        daily_ki_dates=tuple(daily),
        valuation_date=valuation_date,
        maturity_date=maturity_date,
        spec=DCNScheduleSpec(
            initial_date=initial_date,
            maturity_date=maturity_date,
            tenor_months=tenor_months,
            lock_months=lock_months,
            ko_lock_months=ko_lock_months,
            coupon_settlement_offset=coupon_settlement_offset,
            ko_settlement_offset=ko_settlement_offset,
            valuation_date=valuation_date,
            settlement_date=settlement_date,
        ),
        calendar=calendar,
    )
