"""KO-truncated accrual cash legs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from quantark.cashleg.base import CashLeg, roll_payment_time
from quantark.cashleg.base_amount import BaseAmount
from quantark.cashleg.event_distribution import EventDistribution, EventType, SurvivalAnchor
from quantark.cashleg.leg_schedule import LegSchedule
from quantark.util.calendar.business_calendar import Calendar
from quantark.util.calendar.day_counter import (
    DayCountConvention,
    calculate_year_fraction,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import almost_equal


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

    ``survival_until`` (spec D32) says when a terminated note stops accruing for a ``TRUNCATE_AT_KO`` leg:
    ``SurvivalAnchor.DETERMINATION`` (the default, spec D29) at the termination's determination;
    ``SurvivalAnchor.PAYMENT`` (opt-in) at its payment -- every knock-out, never-knocked-in maturity, knocked-in
    settlement and REBASED path settlement counts as alive until its own payment time and gone after it. A
    distribution without the termination payment times refuses a PAYMENT leg (``ValidationError``). A
    ``PAY_FULL_SCHEDULE`` leg reads no survival and refuses ``PAYMENT``.
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
    #: per-period accrual fractions carried by ``time_shift`` on a schedule without dates, so a rolled period
    #: keeps its exact original amount (``(end - dt) - (start - dt)`` is not ``end - start`` in floating point)
    fixed_fractions: Optional[Tuple[float, ...]] = None
    survival_until: SurvivalAnchor = SurvivalAnchor.DETERMINATION

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
        if not isinstance(self.survival_until, SurvivalAnchor):
            raise ValidationError(f"AccrualLeg survival_until must be a SurvivalAnchor, got {self.survival_until!r}")
        if (self.survival_until is SurvivalAnchor.PAYMENT
                and self.ko_behavior is KOBehavior.PAY_FULL_SCHEDULE):
            raise ValidationError(
                "AccrualLeg survival_until=PAYMENT is read only by a TRUNCATE_AT_KO leg; a PAY_FULL_SCHEDULE leg "
                "reads no survival"
            )
        self._validate_day_count_inputs()
        if self.fixed_fractions is not None and len(self.fixed_fractions) != len(
            self.schedule.period_starts
        ):
            raise ValidationError(
                f"AccrualLeg fixed_fractions has {len(self.fixed_fractions)} entries for "
                f"{len(self.schedule.period_starts)} periods"
            )

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
        """KO drives the survival truncation; a full-schedule leg needs nothing. Survival to the payment time
        (spec D32) also reads the maturity outcomes' payment times."""

        if self.ko_behavior is KOBehavior.TRUNCATE_AT_KO:
            if self.survival_until is SurvivalAnchor.PAYMENT:
                return frozenset({EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI})
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
            # spec D29: a note settled at a period's start does not enter it (``survival_after``); a period ending
            # at a settlement is complete (``survival_at`` holds the settled note at its own time); spec D32: the same
            # rule counted from each termination's payment time under ``survival_until=PAYMENT``
            if self.survival_basis is SurvivalBasis.ENTER_PERIOD:
                survival = np.array(
                    [event_dist.survival_after(float(t), anchor=self.survival_until)
                     for t in self.schedule.period_starts], dtype=float
                )
            else:
                survival = np.array(
                    [event_dist.survival_at(float(t), anchor=self.survival_until)
                     for t in self.schedule.period_ends], dtype=float
                )

        payment_times = self._payment_times(event_dist)
        dfs = np.array([env.get_discount_factor(float(t)) for t in payment_times])

        return (
            self.sign()
            * float(self.rate)
            * float(base_amount)
            * float(np.sum(dcf * survival * dfs))
        )

    def time_shift(self, dt: float, *, keep_landing: bool = False) -> "Optional[AccrualLeg]":
        """The leg seen ``dt`` years later (spec 2026-10-23 portfolio-trade-value D10).

        Every schedule time moves by ``-dt`` and each kept period keeps its exact accrual fraction (a period
        start may pass 0; a past survival reference time reads as survived). A period paid on its schedule
        time drops once that payment is no longer ahead (``roll_payment_time``; a landing payment kept at 0
        only with ``keep_landing``); ``None`` when every period has dropped. ``AT_MATURITY`` periods pay at the
        maturity settlement, read from the distribution, so none drops on its schedule time — the owner ends
        the leg with the note. ``dt == 0`` drops the periods paid at 0 and otherwise returns the leg itself.
        """
        dt = float(dt)
        if dt < 0.0:
            return self
        schedule = self.schedule
        fractions = self.accrual_fractions()
        if self.payment_convention is PaymentConvention.AT_MATURITY:
            keep = [True] * len(schedule.payment_times)
            payments = [max(0.0, float(t) - dt) for t in schedule.payment_times]
        else:
            rolled = [
                roll_payment_time(t, dt, keep_landing=keep_landing)
                for t in schedule.payment_times
            ]
            keep = [t is not None for t in rolled]
            payments = [t for t in rolled if t is not None]
            if not payments:
                return None
        if almost_equal(dt, 0.0, tol=0.0) and all(keep):
            return self
        idx = np.flatnonzero(keep)

        def pick(dates):
            return None if dates is None else [dates[i] for i in idx]

        new_schedule = LegSchedule(
            period_starts=schedule.period_starts[idx] - dt,
            period_ends=schedule.period_ends[idx] - dt,
            payment_times=np.asarray(payments, dtype=float),
            period_start_dates=pick(schedule.period_start_dates),
            period_end_dates=pick(schedule.period_end_dates),
            payment_dates=pick(schedule.payment_dates),
        )
        carried = None if schedule.has_period_dates() else tuple(float(f) for f in fractions[idx])
        return replace(self, schedule=new_schedule, fixed_fractions=carried)

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

        if self.fixed_fractions is not None:
            return np.asarray(self.fixed_fractions, dtype=float)
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
