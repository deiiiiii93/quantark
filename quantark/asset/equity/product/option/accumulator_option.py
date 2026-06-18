"""
Accumulator option product definition.

An accumulator ("accumulator forward") is a call-only structured forward. On each
observation date the buyer accumulates ``daily_share_accumulation`` shares at a
strike ``K`` set below spot. Settlement on an observation date (when the contract
has not knocked out) is:

* ``K <= S < KO`` : ``(S - K) * daily_shares``                (linear gain leg)
* ``S < K``       : ``gearing * (S - K) * daily_shares``      (geared loss leg)
* ``S >= KO``     : knock-out (handled by the pricing engine)

The upper knock-out barrier ``KO`` either terminates the whole contract
(:class:`AccumulatorKnockOutType.TERMINATION`) or cancels only that day's accrual
(:class:`AccumulatorKnockOutType.SINGLE_DAY`).

Only CALL accumulators are modelled here; a PUT-style structure is a decumulator
and is intentionally rejected.
"""

from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import List, Optional, Sequence, Tuple

from quantark.util.enum import (
    AccumulatorKnockOutType,
    ExerciseType,
    ObservationAggregation,
    ObservationFrequency,
    ObservationType,
    OptionType,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, is_zero

from .base_equity_option import BaseEquityOption
from .observation_schedule import ObservationSchedule


@dataclass
class AccumulatorOption(BaseEquityOption):
    """
    Call-only accumulator forward with an upper knock-out barrier.

    Attributes:
        strike: Accumulation strike ``K`` (set below spot at inception).
        knock_out_barrier: Upper knock-out barrier ``KO`` (must be above strike).
        initial_price: Reference price used to derive shares from notional.
        knock_out_type: TERMINATION or SINGLE_DAY knock-out behavior.
        knock_out_rebate_rate: Rebate paid on a TERMINATION knock-out, expressed
            as a fraction of notional. Ignored for SINGLE_DAY.
        gearing: Leverage applied to the loss leg below the strike (default 2.0).
        daily_share_accumulation: Shares accumulated per observation. Derived from
            ``notional / initial_price`` when left at 0.
        notional: Notional used to derive ``daily_share_accumulation``.
        settlement_at_expiry: If True, each observation's accrual settles at
            maturity; otherwise it settles on the observation date.
        extra_shares_at_expiry: Additional terminal share leg.
        past_observations: Realized ``(observation_time, observed_spot)`` records
            whose accrual is already locked in.
        observation_type: Monitoring style (DISCRETE by default).
        observation_dates: Discrete observation times in year fractions.
        observation_schedule: Preferred discrete observation schedule.
        observation_frequency: Frequency used to generate a regular schedule.
        business_days_in_year: Business-day count for generated schedules.
        contract_multiplier: Underlying units represented by one contract.
    """

    knock_out_barrier: float = 0.0
    knock_out_type: AccumulatorKnockOutType = AccumulatorKnockOutType.TERMINATION
    knock_out_rebate_rate: float = 0.0
    gearing: float = 2.0
    daily_share_accumulation: float = 0.0
    notional: float = 0.0
    settlement_at_expiry: bool = False
    extra_shares_at_expiry: float = 0.0
    past_observations: Optional[List[Tuple[float, float]]] = None
    observation_type: ObservationType = ObservationType.DISCRETE
    observation_dates: Optional[List[float]] = None
    observation_schedule: Optional[ObservationSchedule] = None
    observation_frequency: ObservationFrequency = ObservationFrequency.CUSTOM
    use_business_days_for_frequency: bool = True
    business_days_in_year: float = 252.0

    def __init__(
        self,
        strike: float,
        knock_out_barrier: float,
        option_type: OptionType = OptionType.CALL,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        initial_price: float = 0.0,
        knock_out_type: AccumulatorKnockOutType = AccumulatorKnockOutType.TERMINATION,
        knock_out_rebate_rate: float = 0.0,
        gearing: float = 2.0,
        daily_share_accumulation: float = 0.0,
        notional: float = 0.0,
        settlement_at_expiry: bool = False,
        extra_shares_at_expiry: float = 0.0,
        past_observations: Optional[List[Tuple[float, float]]] = None,
        observation_type: ObservationType = ObservationType.DISCRETE,
        observation_dates: Optional[List[float]] = None,
        observation_schedule: Optional[ObservationSchedule] = None,
        observation_frequency: ObservationFrequency = ObservationFrequency.CUSTOM,
        use_business_days_for_frequency: bool = True,
        business_days_in_year: float = 252.0,
        contract_multiplier: float = 1.0,
    ):
        """
        Initialize an accumulator option.

        Args:
            strike: Accumulation strike ``K``.
            knock_out_barrier: Upper knock-out barrier ``KO`` (above strike).
            option_type: Must be CALL (PUT is rejected as a decumulator).
            maturity: Time to maturity in years (optional if exercise_date given).
            exercise_date: Expiration date (optional if maturity given).
            settlement_date: Settlement date.
            initial_price: Reference price for share derivation.
            knock_out_type: TERMINATION or SINGLE_DAY.
            knock_out_rebate_rate: Rebate fraction of notional on TERMINATION KO.
            gearing: Leverage on the loss leg (default 2.0).
            daily_share_accumulation: Shares per observation; derived from
                ``notional / initial_price`` when 0.
            notional: Notional used to derive shares.
            settlement_at_expiry: Defer accrual settlement to maturity.
            extra_shares_at_expiry: Additional terminal share leg.
            past_observations: Realized ``(time, spot)`` observation records.
            observation_type: Monitoring style (DISCRETE by default).
            observation_dates: Discrete observation times in years.
            observation_schedule: Preferred discrete schedule.
            observation_frequency: Frequency for generated schedules.
            use_business_days_for_frequency: Use business-day spacing.
            business_days_in_year: Business-day count for schedules.
            contract_multiplier: Underlying units per contract.

        Raises:
            ValidationError: If parameters are invalid.
        """
        if maturity is None and exercise_date is None:
            maturity = 0.0

        self.knock_out_barrier = knock_out_barrier
        self.knock_out_type = knock_out_type
        self.knock_out_rebate_rate = knock_out_rebate_rate
        self.gearing = gearing
        self.notional = notional
        self.settlement_at_expiry = settlement_at_expiry
        self.extra_shares_at_expiry = extra_shares_at_expiry
        self.past_observations = past_observations
        self.observation_type = observation_type
        self.observation_dates = observation_dates
        self.observation_schedule = observation_schedule
        self.observation_frequency = observation_frequency
        self.use_business_days_for_frequency = use_business_days_for_frequency
        self.business_days_in_year = business_days_in_year

        # Derive daily shares from notional when not explicitly provided.
        if is_zero(daily_share_accumulation) and not is_zero(notional):
            if initial_price <= 0.0:
                raise ValidationError(
                    "Cannot derive daily_share_accumulation from a nonzero notional "
                    "without a positive initial_price; provide initial_price or set "
                    "daily_share_accumulation explicitly."
                )
            daily_share_accumulation = notional / initial_price
        self.daily_share_accumulation = daily_share_accumulation

        super().__init__(
            strike=strike,
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            maturity=maturity,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            initial_price=initial_price,
            contract_multiplier=contract_multiplier,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate accumulator parameters.

        Raises:
            ValidationError: If any parameter is invalid.
        """
        super().validate()

        if self.option_type != OptionType.CALL:
            raise ValidationError(
                "AccumulatorOption only supports CALL; a PUT-style structure is a "
                "decumulator and should be modelled separately."
            )
        if self.knock_out_barrier <= 0:
            raise ValidationError(
                f"Knock-out barrier must be positive, got {self.knock_out_barrier}"
            )
        if self.knock_out_barrier <= self.strike:
            raise ValidationError(
                "Accumulator requires an upper barrier above strike, got "
                f"barrier={self.knock_out_barrier}, strike={self.strike}"
            )
        if self.gearing < 0:
            raise ValidationError(f"Gearing must be non-negative, got {self.gearing}")
        if self.daily_share_accumulation < 0:
            raise ValidationError(
                f"daily_share_accumulation must be non-negative, got "
                f"{self.daily_share_accumulation}"
            )
        if self.knock_out_rebate_rate < 0:
            raise ValidationError(
                f"knock_out_rebate_rate must be non-negative, got "
                f"{self.knock_out_rebate_rate}"
            )
        if not isinstance(self.knock_out_type, AccumulatorKnockOutType):
            raise ValidationError(
                f"Invalid knock_out_type: {self.knock_out_type}"
            )
        if not isinstance(self.observation_type, ObservationType):
            raise ValidationError(
                f"Invalid observation type: {self.observation_type}"
            )
        if not isinstance(self.observation_frequency, ObservationFrequency):
            raise ValidationError(
                f"Invalid observation frequency: {self.observation_frequency}"
            )
        if self.business_days_in_year <= 0:
            raise ValidationError(
                f"business_days_in_year must be positive, got "
                f"{self.business_days_in_year}"
            )

        self._normalize_observation_schedule()

    def _normalize_observation_schedule(self) -> None:
        """Normalize discrete monitoring inputs into an ObservationSchedule."""
        if self.observation_type in (
            ObservationType.CONTINUOUS,
            ObservationType.EXPIRY,
        ):
            if self.observation_schedule is not None:
                raise ValidationError(
                    "ObservationSchedule requires DISCRETE observation_type."
                )
            self.observation_dates = self.observation_dates or []
            return

        if self.observation_schedule is not None:
            if self.observation_schedule.aggregation_mode != self._aggregation_mode():
                raise ValidationError(
                    "Supplied observation_schedule aggregation "
                    f"({self.observation_schedule.aggregation_mode}) is inconsistent "
                    f"with knock_out_type {self.knock_out_type}; expected "
                    f"{self._aggregation_mode()}."
                )
            self._validate_observation_times(self.observation_schedule.times)
            if self.observation_schedule.times:
                self.observation_dates = self.observation_schedule.times
            return

        if self.observation_dates is None or len(self.observation_dates) == 0:
            self.observation_dates = self._generate_observation_dates()
        self._validate_observation_times(self.observation_dates)

        self.observation_schedule = ObservationSchedule.from_legacy(
            observation_dates=self.observation_dates,
            default_barrier=self.knock_out_barrier,
            default_payoff=self.get_knock_out_rebate_cash() * self.contract_multiplier,
            aggregation_mode=self._aggregation_mode(),
            frequency=self.observation_frequency,
        )
        self.observation_dates = self.observation_schedule.times

    def _aggregation_mode(self) -> ObservationAggregation:
        """Aggregation semantics implied by the knock-out type.

        TERMINATION stops at the first barrier hit; SINGLE_DAY cancels only the
        breached observation and keeps accruing, so observations accumulate.
        """
        if self.knock_out_type == AccumulatorKnockOutType.TERMINATION:
            return ObservationAggregation.STOP_FIRST_HIT
        return ObservationAggregation.ACCUMULATE

    def _validate_observation_times(self, times: Sequence[float]) -> None:
        """Validate numeric observation times when present."""
        if any(t < 0 for t in times):
            raise ValidationError("Observation dates must be non-negative.")
        if list(times) != sorted(times):
            raise ValidationError(
                "Observation dates must be sorted in ascending order."
            )
        if self.maturity is not None:
            if any(t > self.maturity for t in times):
                raise ValidationError(
                    "Observation dates must fall within the option maturity "
                    f"({self.maturity}); got a later observation in {list(times)}."
                )

    def _generate_observation_dates(self) -> List[float]:
        """Generate a regular discrete schedule from ``observation_frequency``."""
        if self.observation_frequency == ObservationFrequency.CUSTOM:
            raise ValidationError(
                "Observation dates required for discrete accumulator monitoring "
                "when observation_frequency is CUSTOM."
            )
        if self.maturity is None or self.maturity <= 0:
            raise ValidationError(
                "Positive maturity is required to generate discrete observations."
            )

        days_in_year = (
            self.business_days_in_year
            if self.use_business_days_for_frequency
            else 365.0
        )
        dt = self.observation_frequency.to_year_fraction(
            use_business_days=self.use_business_days_for_frequency,
            days_in_year=days_in_year,
        )
        if dt <= 0:
            raise ValidationError(f"Observation frequency produced invalid dt={dt}")

        count = max(1, int(ceil(self.maturity / dt)))
        times = [min((idx + 1) * dt, self.maturity) for idx in range(count)]
        if not is_close(times[-1], self.maturity):
            times.append(self.maturity)
        return times

    # ------------------------------------------------------------------
    # Payoff helpers
    # ------------------------------------------------------------------

    def get_observation_payoff(self, spot: float) -> float:
        """
        Per-observation settlement assuming the contract has not knocked out.

        The geared loss leg below the strike makes this negative for ``spot < K``.

        Args:
            spot: Observed underlying price.

        Returns:
            Settlement cashflow for one observation, scaled by shares and the
            contract multiplier.
        """
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")

        moneyness = spot - self.strike
        leverage = 1.0 if spot >= self.strike else self.gearing
        return (
            leverage
            * moneyness
            * self.daily_share_accumulation
            * self.contract_multiplier
        )

    def get_knock_out_rebate_cash(self) -> float:
        """
        Cash rebate paid once on a TERMINATION knock-out (per contract).

        Expressed as ``knock_out_rebate_rate * notional``. Returns zero for
        SINGLE_DAY, where the rebate is documented as ignored.
        """
        if self.knock_out_type != AccumulatorKnockOutType.TERMINATION:
            return 0.0
        return self.knock_out_rebate_rate * self.notional

    def get_payoff(self, spot: float) -> float:
        """
        Single-observation settlement, knock-out aware.

        An accumulator has no single terminal payoff; this returns the
        settlement for one observation. A breach (``spot >= KO``) does not accrue
        the linear/geared leg: SINGLE_DAY pays zero and TERMINATION pays the
        knock-out rebate (zero if none configured). Otherwise it delegates to
        :meth:`get_observation_payoff`.
        """
        if spot >= self.knock_out_barrier:
            return self.get_knock_out_rebate_cash() * self.contract_multiplier
        return self.get_observation_payoff(spot)

    def get_realized_accrual(self) -> float:
        """
        Locked-in accrual from realized past observations (rate-free).

        Observations are processed in chronological order. A breach
        (``spot >= KO``) never accrues. For TERMINATION the contract ends at the
        first breach, so later observations are excluded; for SINGLE_DAY only the
        breached observation is skipped and accrual continues. Discounting (if
        settlement is deferred to expiry) is applied by the pricing engine.

        Returns:
            Sum of per-observation settlements for realized observations.
        """
        if not self.past_observations:
            return 0.0
        terminates = self.knock_out_type == AccumulatorKnockOutType.TERMINATION
        total = 0.0
        for _, spot in sorted(self.past_observations, key=lambda obs: obs[0]):
            if spot >= self.knock_out_barrier:
                if terminates:
                    break
                continue
            total += self.get_observation_payoff(spot)
        return total

    def get_observation_times(self) -> List[float]:
        """Return discrete observation times in years."""
        if self.observation_schedule is not None:
            return self.observation_schedule.times
        return self.observation_dates or []

    def time_shift(self, time_bump: float, bumped_date: datetime, pricing_env) -> bool:
        """Shift observation schedule and maturity for theta bumping."""
        schedule = getattr(self, "observation_schedule", None)
        if schedule is not None:
            if schedule.uses_dates():
                pricing_env.valuation_date = bumped_date
            bumped_schedule = schedule.time_shift(time_bump, bumped_date)
            if bumped_schedule is None:
                return True
            self.observation_schedule = bumped_schedule
            if bumped_schedule.uses_times():
                self.observation_dates = bumped_schedule.times

        if getattr(self, "exercise_date", None) is None and self.maturity is not None:
            self.maturity -= time_bump

        return False

    def __repr__(self) -> str:
        return (
            "AccumulatorOption("
            f"K={self.strike:.2f}, KO={self.knock_out_barrier:.2f}, "
            f"gearing={self.gearing:.2f}, T={self.maturity:.4f})"
        )
