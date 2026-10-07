"""
Configuration classes for Snowball (autocallable) options.

These classes group related parameters to simplify the SnowballOption API.
"""

from dataclasses import dataclass, replace
from math import isfinite
from typing import List, Optional, Union

from quantark.util.calendar.business_calendar import Calendar
from quantark.util.enum import (
    AccrualEndType,
    ObservationType,
    CouponPayType,
    ProtectionType,
)
from quantark.util.numerical import is_zero

from .observation_schedule import ObservationSchedule


@dataclass(frozen=True)
class BarrierConfig:
    """
    Configuration for knock-out (KO) and knock-in (KI) barriers.

    Attributes:
        ko_barrier: Knock-out barrier level(s), up barrier by default
        ko_rate: Knock-out return rate(s)
        ko_observation_type: DISCRETE or CONTINUOUS monitoring for KO
        ko_observation_dates: Year fractions for KO observations (legacy)
        ko_observation_schedule: ObservationSchedule for KO (preferred)
        ki_barrier: Optional knock-in barrier level(s), down barrier by default
        ki_observation_type: DISCRETE or CONTINUOUS monitoring for KI
        ki_observation_dates: Year fractions for KI observations (legacy)
        ki_observation_schedule: ObservationSchedule for KI (preferred)
        ki_continuous: If True, KI monitored continuously
        disable_ko_after_ki: If True, disable KO after KI is triggered. "After" is strict: a
            knock-out observed at the SAME instant as a discretely observed knock-in wins the tie
            and is not disabled by it, on every autocallable and every engine. (A continuously
            monitored level breached at an observation was touched before it, so that knock-in
            does come first.) One spot breaches both only when the knock-out level is at or
            below the knock-in level.
    """

    # Knock-out barrier (required)
    ko_barrier: Union[float, List[float]]
    ko_rate: Union[float, List[float]]
    ko_observation_type: ObservationType = ObservationType.DISCRETE
    ko_observation_dates: Optional[List[float]] = None
    ko_observation_schedule: Optional[ObservationSchedule] = None

    # Knock-in barrier (optional)
    ki_barrier: Optional[Union[float, List[float]]] = None
    ki_observation_type: ObservationType = ObservationType.DISCRETE
    ki_observation_dates: Optional[List[float]] = None
    ki_observation_schedule: Optional[ObservationSchedule] = None
    ki_continuous: bool = False

    # Interaction
    disable_ko_after_ki: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate ko_barrier is positive
        self._validate_barrier_positive(self.ko_barrier, "ko_barrier")
        
        # Validate ki_barrier is positive if provided
        if self.ki_barrier is not None:
            self._validate_barrier_positive(self.ki_barrier, "ki_barrier")
        
        # Validate observation types are enums
        if not isinstance(self.ko_observation_type, ObservationType):
            raise ValueError(f"ko_observation_type must be ObservationType, got {type(self.ko_observation_type)}")
        if not isinstance(self.ki_observation_type, ObservationType):
            raise ValueError(f"ki_observation_type must be ObservationType, got {type(self.ki_observation_type)}")

    def time_shift(
        self, time_bump: float, bumped_date, pricing_env, *, keep_landing: bool = False
    ) -> tuple["BarrierConfig", bool]:
        """
        Shift KO/KI observation schedules and legacy dates for theta bumps.

        Returns a new BarrierConfig and a flag saying the shift left the
        contract with no observations of EITHER kind.

        The two schedules exhaust independently: a snowball whose knock-in
        dates have all passed while knock-out dates remain is a live
        contract, and so is the reverse.  Callers read the flag as "there
        is nothing left to price" and keep the unshifted schedule instead,
        which would bring observations that have already happened back to
        life, so it may only be raised when both sides are spent.
        """
        ko_schedule, ko_dates = self._shift_side(
            self.ko_observation_schedule,
            self.ko_observation_dates,
            time_bump,
            bumped_date,
            pricing_env,
            keep_landing,
        )
        ki_schedule, ki_dates = self._shift_side(
            self.ki_observation_schedule,
            self.ki_observation_dates,
            time_bump,
            bumped_date,
            pricing_env,
            keep_landing,
        )

        had_any = self._has_future(
            self.ko_observation_schedule, self.ko_observation_dates
        ) or self._has_future(self.ki_observation_schedule, self.ki_observation_dates)
        has_any = self._has_future(ko_schedule, ko_dates) or self._has_future(
            ki_schedule, ki_dates
        )
        dropped_all = had_any and not has_any

        return (
            replace(
                self,
                ko_observation_schedule=ko_schedule,
                ki_observation_schedule=ki_schedule,
                ko_observation_dates=ko_dates,
                ki_observation_dates=ki_dates,
            ),
            dropped_all,
        )

    @property
    def ko_observation_count(self) -> int:
        """How many knock-out observations this config carries."""
        if self.ko_observation_schedule is not None:
            return len(self.ko_observation_schedule.records)
        return len(self.ko_observation_dates or ())

    @staticmethod
    def _has_future(schedule, dates) -> bool:
        """Whether this side still carries an observation that has not happened."""
        if schedule is not None:
            return bool(schedule.records)
        return bool(dates)

    @staticmethod
    def _shift_side(schedule, dates, time_bump, bumped_date, pricing_env, keep_landing=False):
        """Shift one side's schedule and legacy dates, returning both.

        ``ObservationSchedule.time_shift`` returns ``None`` once every
        record has passed, which a config cannot tell apart from "this
        contract never had this kind of observation" -- and the resolvers
        reject the second.  A contract whose knock-in dates have all passed
        still HAS knock-in observations; what it has none of is FUTURE
        ones.  That is an empty schedule, so the emptied side keeps one.
        """
        if schedule is None:
            if dates:
                if keep_landing:
                    # theta (spec 2026-10-07 theta boundary R1): the date the bump lands on is observed at time 0
                    return None, [0.0 if is_zero(t - time_bump) else t - time_bump
                                  for t in dates if is_zero(t - time_bump) or t - time_bump > 0]
                return None, [t - time_bump for t in dates if t - time_bump > 0]
            return None, dates

        if schedule.uses_dates():
            pricing_env.valuation_date = bumped_date
        shifted = schedule.time_shift(time_bump, bumped_date, keep_landing=keep_landing)
        if shifted is None:
            shifted = ObservationSchedule(
                records=[],
                aggregation_mode=schedule.aggregation_mode,
                frequency=schedule.frequency,
            )
        if not shifted.records:
            return shifted, []
        if shifted.uses_times():
            return shifted, shifted.times
        return shifted, dates

    @staticmethod
    def _validate_barrier_positive(barrier: Union[float, List[float]], name: str) -> None:
        """Validate that barrier level(s) are positive."""
        if isinstance(barrier, list):
            if not barrier:
                raise ValueError(f"{name} list cannot be empty")
            for i, b in enumerate(barrier):
                if not isinstance(b, (int, float)) or b <= 0:
                    raise ValueError(f"{name}[{i}] must be positive number, got {b}")
        else:
            if not isinstance(barrier, (int, float)) or barrier <= 0:
                raise ValueError(f"{name} must be positive number, got {barrier}")


@dataclass(frozen=True)
class PayoffConfig:
    """
    Configuration for payoff features (rebate, protection, participation).

    Attributes:
        rebate_rate: Fixed rebate rate for V0 maturity payoff
        call_rebate_enabled: If True, use call-style rebate instead of fixed
        call_strike: Strike for call rebate
        call_participation_rate: Participation rate for call rebate
        include_principal: Whether principal is part of payouts
        participation_rate: Downside participation rate after KI
        protection_type: NONE, PARTIAL, or FULL protection
        protection_rate: Rate for partial protection floor
    """

    # Rebate (V0 maturity payoff)
    rebate_rate: float = 0.0
    call_rebate_enabled: bool = False
    call_strike: Optional[float] = None
    call_participation_rate: float = 1.0

    # Participation and protection (V1 maturity payoff)
    include_principal: bool = True
    participation_rate: float = 1.0
    protection_type: ProtectionType = ProtectionType.NONE
    protection_rate: float = 0.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate participation rate is positive
        if self.participation_rate <= 0:
            raise ValueError(f"participation_rate must be positive, got {self.participation_rate}")
        
        # Validate call rebate parameters if enabled
        if self.call_rebate_enabled:
            if self.call_strike is None:
                raise ValueError("call_strike required when call_rebate_enabled is True")
            if self.call_strike <= 0:
                raise ValueError(f"call_strike must be positive, got {self.call_strike}")
            if self.call_participation_rate <= 0:
                raise ValueError(f"call_participation_rate must be positive, got {self.call_participation_rate}")
        
        # Validate protection type is enum
        if not isinstance(self.protection_type, ProtectionType):
            raise ValueError(f"protection_type must be ProtectionType, got {type(self.protection_type)}")
        
        # Validate protection rate for partial protection
        if self.protection_type == ProtectionType.PARTIAL:
            if not 0 <= self.protection_rate <= 1:
                raise ValueError(f"protection_rate must be in [0, 1], got {self.protection_rate}")


@dataclass(frozen=True)
class AccrualConfig:
    """
    Configuration for accrual and coupon payment settings.

    Attributes:
        coupon_pay_type: INSTANT (at KO date) or EXPIRY (discounted to maturity)
        is_annualized: Backward-compatible flag for annualized accruals (default for all)
        is_annualized_ko: If True, KO return accrues with year fraction
        is_annualized_ki: If True, KI return accrues with year fraction
        is_annualized_rebate: If True, rebate accrues with year fraction
        is_annualized_coupon: If True, a Phoenix coupon rate is per annum and the
            coupon pays principal x rate x the period's year fraction; if False the
            rate IS the period's amount and its fraction is 1
        accrual_factors: Optional externally supplied accrual factors by
            KO/coupon observation
        accrued_offset: Year fraction already accrued before the valuation
            date, banked by ``time_shift`` so that ageing a contract with
            no ``initial_date`` does not shorten its coupons
    """

    coupon_pay_type: CouponPayType = CouponPayType.INSTANT
    is_annualized: bool = True
    is_annualized_ko: Optional[bool] = None
    is_annualized_ki: Optional[bool] = None
    is_annualized_rebate: Optional[bool] = None
    is_annualized_coupon: Optional[bool] = None
    accrual_factors: Optional[List[float]] = None
    #: Year fraction already accrued before the valuation date.
    #:
    #: A time-based observation schedule measures its records from the
    #: product's own time origin, so a contract with no ``initial_date``
    #: says "my accrual starts where my schedule starts": at inception the
    #: accrual factor of a coupon IS its observation time.  Ageing moves
    #: that origin forward and shortens every observation time with it, so
    #: the elapsed period is banked here and added back -- otherwise each
    #: surviving coupon quietly loses exactly that much, on a contract
    #: that is right on its trade date and wrong from the first day
    #: onward.  ``time_shift`` accumulates it; it stays 0.0 at inception,
    #: and is ignored wherever the accrual has a real anchor
    #: (``initial_date``, dated observations, or ``accrual_factors``).
    accrued_offset: float = 0.0
    #: Where the first remaining coupon period started, as time since the contract's origin: 0.0 until
    #: an ageing (``time_shift``) drops an observation, then the last dropped observation's time. A tenor
    #: contract's first remaining period is ``accrued_offset`` + its observation time minus this.
    #: ``None`` when an ageing dropped an observation that carries no time: the period start is unknown
    #: and a per-annum coupon refuses (spec 3B decision 13, as amended by the user 2026-10-04).
    first_period_start: Optional[float] = 0.0
    #: Where every record's accrual ends (R2): its observation, or its authoritative settlement endpoint.
    accrual_end: AccrualEndType = AccrualEndType.OBSERVATION
    #: KO-reset schedule 2's explicit factors, positionally (accrual time spec §4.3). ``accrual_factors`` is
    #: schedule 1's only.
    post_accrual_factors: Optional[List[float]] = None
    #: The accrual basis's calendar and denominator for a ``BUSINESS_DAYS`` annualized leg (spec §4.7), frozen
    #: with the contract: no accrual reads the pricing environment's.
    accrual_calendar: Optional[Calendar] = None
    accrual_bus_days_in_year: Optional[float] = None

    def shifted(self, time_bump: float, dropped_observations: int = 0,
                schedule=None, dropped_post: int = 0) -> "AccrualConfig":
        """The accrual of the same contract after it has been aged.

        Two things move.  ``accrued_offset`` banks the elapsed period, so
        an unanchored time-based schedule still accrues from where the
        contract started rather than from today.  ``accrual_factors`` are
        addressed by POSITION in the knock-out schedule, so the records
        the shift took off its front have to come off theirs too --
        otherwise every survivor is paid the factor of an observation that
        has already happened.

        ``dropped_observations`` is a count off the FRONT, which is what a
        shift removes: a schedule is ordered by observation time (its own
        validation enforces that, and an unordered one cannot resolve), so
        the records that fall behind the valuation date are a prefix.

        ``schedule`` is the knock-out schedule BEFORE the shift: its last dropped record is where the
        first remaining period now starts (``first_period_start``). ``dropped_post`` takes the same prefix off
        ``post_accrual_factors`` (KO-reset schedule 2). The factors only keep a rebuilt aged product's length
        checks true: every record's accrual period already holds them.
        """
        factors = self.accrual_factors
        post = self.post_accrual_factors
        if post is not None and int(dropped_post) > 0:
            post = list(post[int(dropped_post):])
        start = self.first_period_start
        dropped = int(dropped_observations)
        if dropped > 0:
            if factors is not None:
                factors = list(factors[dropped:])
            records = schedule.records if schedule is not None else []
            last = records[dropped - 1].observation_time if dropped <= len(records) else None
            start = (None if start is None or last is None
                     else float(self.accrued_offset) + float(last))
        return replace(
            self,
            accrual_factors=factors,
            post_accrual_factors=post,
            accrued_offset=float(self.accrued_offset) + float(time_bump),
            first_period_start=start,
        )

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate coupon_pay_type is enum
        if not isinstance(self.coupon_pay_type, CouponPayType):
            raise ValueError(f"coupon_pay_type must be CouponPayType, got {type(self.coupon_pay_type)}")
        offset = float(self.accrued_offset)
        if not isfinite(offset) or offset < 0.0:
            raise ValueError(
                "accrued_offset must be a finite non-negative year fraction, "
                f"got {self.accrued_offset}"
            )
        self._check_factor_list("accrual_factors", self.accrual_factors)
        self._check_factor_list("post_accrual_factors", self.post_accrual_factors)
        if not isinstance(self.accrual_end, AccrualEndType):
            raise ValueError(f"accrual_end must be AccrualEndType, got {self.accrual_end!r}")
        if self.accrual_calendar is not None and not isinstance(self.accrual_calendar, Calendar):
            raise ValueError(
                f"accrual_calendar must be Calendar, got {type(self.accrual_calendar).__name__}"
            )
        if self.accrual_bus_days_in_year is not None:
            days = float(self.accrual_bus_days_in_year)
            if not isfinite(days) or days <= 0.0:
                raise ValueError(
                    f"accrual_bus_days_in_year must be positive, got {self.accrual_bus_days_in_year}"
                )

    @staticmethod
    def _check_factor_list(name: str, values) -> None:
        if values is None:
            return
        if not isinstance(values, list):
            raise ValueError(f"{name} must be a list, got {type(values)}")
        for i, factor in enumerate(values):
            if isinstance(factor, bool) or not isinstance(factor, (int, float)):
                raise ValueError(f"{name}[{i}] must be numeric, got {factor}")
            if factor < 0:
                raise ValueError(f"{name}[{i}] must be non-negative, got {factor}")

@dataclass(frozen=True)
class AirbagConfig:
    """
    Configuration for airbag features.

    Attributes:
        airbag_barrier: Barrier level for airbag protection.
                        If spot > airbag_barrier, principal is fully protected (subject to other terms).
                        If spot < airbag_barrier, investor participates in downside.
        airbag_participation_rate: Participation rate when spot < airbag_barrier (default: 1.0).
        airbag_strike: Strike price for airbag payoff calculation (optional).
                       If None, defaults to the product's strike.
    """

    airbag_barrier: Optional[float] = None
    airbag_participation_rate: float = 1.0
    airbag_strike: Optional[float] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.airbag_barrier is not None:
            if not isinstance(self.airbag_barrier, (int, float)) or self.airbag_barrier <= 0:
                raise ValueError(f"airbag_barrier must be a positive number, got {self.airbag_barrier}")
        
        if self.airbag_participation_rate <= 0:
            raise ValueError(f"airbag_participation_rate must be positive, got {self.airbag_participation_rate}")

        if self.airbag_strike is not None:
            if not isinstance(self.airbag_strike, (int, float)) or self.airbag_strike <= 0:
                raise ValueError(f"airbag_strike must be a positive number, got {self.airbag_strike}")
