"""KO-truncated accrual cash legs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from quantark.cashleg.base import CashLeg
from quantark.cashleg.base_amount import BaseAmount
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.leg_schedule import LegSchedule
from quantark.util.calendar.day_counter import DayCountConvention
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


@dataclass(frozen=True)
class AccrualLeg(CashLeg):
    """Periodic accrual stream, optionally shortened by KO."""

    rate: float = 0.0
    base: BaseAmount | None = None
    schedule: LegSchedule | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    payment_convention: PaymentConvention = PaymentConvention.AT_PERIOD_END
    ko_behavior: KOBehavior = KOBehavior.TRUNCATE_AT_KO
    survival_basis: SurvivalBasis = SurvivalBasis.ENTER_PERIOD

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

        dcf = np.array(
            [
                self._dcf(
                    float(self.schedule.period_starts[i]),
                    float(self.schedule.period_ends[i]),
                )
                for i in range(n_periods)
            ],
            dtype=float,
        )

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
                len(self.schedule.payment_times), float(event_dist.event_times[-1])
            )
        return self.schedule.payment_times

    def _dcf(self, start: float, end: float) -> float:
        # LegSchedule stores year fractions; the schedule builder owns exact
        # calendar day-count conversion when actual dates are needed.
        return float(end - start)
