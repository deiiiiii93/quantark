"""KO-truncated accrual cash legs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from quantark.cashleg.base import CashLeg
from quantark.cashleg.base_amount import BaseAmount
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.leg_schedule import LegSchedule
from quantark.util.calendar.business_calendar import Calendar
from quantark.util.calendar.day_counter import (
    DayCountConvention,
    calculate_year_fraction,
)
from quantark.util.exceptions import ValidationError


class PaymentConvention(Enum):
    """Timing convention for accrual payments."""

    AT_PERIOD_END = "at_period_end"
    AT_KO = "at_ko"
    AT_MATURITY = "at_maturity"


class KOBehavior(Enum):
    """Whether KO truncates the accrual stream."""

    TRUNCATE_AT_KO = "truncate_at_ko"
    PAY_FULL_SCHEDULE = "pay_full_schedule"


class SurvivalBasis(Enum):
    """Survival probability used for each accrual period."""

    ENTER_PERIOD = "enter_period"
    COMPLETE_PERIOD = "complete_period"


# Conventions whose fraction depends on the coupon frequency, which an
# AccrualLeg does not carry; a dated leg refuses them rather than let
# calculate_year_fraction assume semi-annual.
_FREQUENCY_CONVENTIONS = frozenset(
    {
        DayCountConvention.ACT_ACT_ISMA,
        DayCountConvention.ACT_ACT_BOND,
        DayCountConvention.ACT_ACT_CHINABOND,
    }
)


@dataclass(frozen=True)
class AccrualLeg(CashLeg):
    """Periodic accrual stream, optionally shortened by KO.

    Each period accrues ``rate * base * fraction``. When the schedule carries
    period dates, ``fraction`` is ``calculate_year_fraction(start_date,
    end_date, day_count)``; with times only it is ``period_end - period_start``
    and ``day_count`` is not read. A dated ``BUSINESS_DAYS`` leg needs
    ``calendar`` and ``bus_days_in_year``; those two fields are refused on any
    other leg, which would never read them.
    """

    rate: float = 0.0
    base: BaseAmount | None = None
    schedule: LegSchedule | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    payment_convention: PaymentConvention = PaymentConvention.AT_PERIOD_END
    ko_behavior: KOBehavior = KOBehavior.TRUNCATE_AT_KO
    survival_basis: SurvivalBasis = SurvivalBasis.ENTER_PERIOD
    calendar: Optional[Calendar] = None
    bus_days_in_year: Optional[int] = None

    def __post_init__(self) -> None:
        if self.base is None:
            raise ValidationError("AccrualLeg requires base")
        if self.schedule is None:
            raise ValidationError("AccrualLeg requires schedule")
        if not isinstance(self.day_count, DayCountConvention):
            raise ValidationError(f"Invalid DayCountConvention: {self.day_count}")
        if not isinstance(self.payment_convention, PaymentConvention):
            raise ValidationError(
                f"Invalid PaymentConvention: {self.payment_convention}"
            )
        if not isinstance(self.ko_behavior, KOBehavior):
            raise ValidationError(f"Invalid KOBehavior: {self.ko_behavior}")
        if not isinstance(self.survival_basis, SurvivalBasis):
            raise ValidationError(f"Invalid SurvivalBasis: {self.survival_basis}")
        self._validate_day_count_inputs()

    def _validate_day_count_inputs(self) -> None:
        dated = self.schedule.has_period_dates()
        business = self.day_count is DayCountConvention.BUSINESS_DAYS
        if (self.calendar is not None or self.bus_days_in_year is not None) and not (
            dated and business
        ):
            raise ValidationError(
                "AccrualLeg calendar / bus_days_in_year are read only by a "
                "BUSINESS_DAYS leg whose schedule carries period dates; got "
                f"day_count={self.day_count.name}, dated schedule={dated}"
            )
        if not dated:
            return
        if business:
            if self.calendar is None:
                raise ValidationError(
                    "AccrualLeg BUSINESS_DAYS on a dated schedule requires a "
                    "business-day calendar"
                )
            if self.bus_days_in_year is None:
                raise ValidationError(
                    "AccrualLeg BUSINESS_DAYS on a dated schedule requires "
                    "bus_days_in_year (the year length is not guessed)"
                )
            if isinstance(self.bus_days_in_year, bool) or not isinstance(
                self.bus_days_in_year, int
            ) or self.bus_days_in_year <= 0:
                raise ValidationError(
                    "AccrualLeg bus_days_in_year must be a positive int, got "
                    f"{self.bus_days_in_year!r}"
                )
        if self.day_count in _FREQUENCY_CONVENTIONS:
            # TODO: ACT/ACT ISMA/ICMA/BOND/CHINABOND need the coupon frequency
            # (and the reference period); AccrualLeg carries neither.
            raise ValidationError(
                f"AccrualLeg day_count {self.day_count.name} needs a coupon "
                "frequency the leg does not carry; not supported on a dated schedule"
            )

    def required_event_types(self) -> frozenset:
        """KO drives the survival truncation; a full-schedule leg needs nothing."""

        if self.ko_behavior is KOBehavior.TRUNCATE_AT_KO:
            return frozenset({EventType.KO})
        return frozenset()

    def value(
        self, event_dist: EventDistribution, env, position_notional: float
    ) -> float:
        """Return signed expected PV of the accrual stream."""

        base_amount = self.base.resolve(position_notional)
        n_periods = len(self.schedule.period_starts)

        dcf = self.accrual_fractions()

        if self.ko_behavior is KOBehavior.PAY_FULL_SCHEDULE:
            survival = np.ones(n_periods, dtype=float)
        else:
            reference_times = (
                self.schedule.period_starts
                if self.survival_basis is SurvivalBasis.ENTER_PERIOD
                else self.schedule.period_ends
            )
            survival = np.array(
                [event_dist.survival_at(float(t)) for t in reference_times],
                dtype=float,
            )

        payment_times = self._payment_times(event_dist)
        dfs = np.array([env.get_discount_factor(float(t)) for t in payment_times])

        return (
            self.sign()
            * float(self.rate)
            * float(base_amount)
            * float(np.sum(dcf * survival * dfs))
        )

    def _payment_times(self, event_dist: EventDistribution) -> np.ndarray:
        if self.payment_convention is PaymentConvention.AT_MATURITY:
            return np.full(
                len(self.schedule.payment_times),
                float(
                    event_dist.payment_times_for(
                        EventType.MATURITY_NO_KO
                    )
                ),
            )
        return self.schedule.payment_times

    def accrual_fractions(self) -> np.ndarray:
        """Per-period accrual fraction: ``day_count`` over the period dates
        when the schedule has them, else ``period_end - period_start``."""

        if not self.schedule.has_period_dates():
            return np.asarray(
                self.schedule.period_ends - self.schedule.period_starts, dtype=float
            )
        business = (
            {"bus_days_in_year": self.bus_days_in_year, "calendar": self.calendar}
            if self.day_count is DayCountConvention.BUSINESS_DAYS
            else {}
        )
        return np.array(
            [
                calculate_year_fraction(start, end, self.day_count, **business)
                for start, end in zip(
                    self.schedule.period_start_dates, self.schedule.period_end_dates
                )
            ],
            dtype=float,
        )
