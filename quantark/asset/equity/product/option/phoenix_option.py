"""
Phoenix option implementation.

Phoenix options are autocallable structured products with periodic coupon payments
when spot exceeds a coupon barrier at observation dates. Unlike Snowball options
which only pay coupons on knock-out events, Phoenix options pay coupons at each
observation where the coupon barrier condition is met.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, List, Optional, Union

from quantark.asset.equity.product.option.base_equity_option import BaseEquityOption
from quantark.util.calendar import calculate_year_fraction
from quantark.util.calendar.day_counter import DayCountConvention, calculate_day_count_fraction
from quantark.util.enum import (
    BarrierType,
    CouponPayType,
    ExerciseType,
    ObservationType,
    OptionType,
    ProtectionType,
    TenorEnd,
)
from quantark.util.exceptions import ValidationError

from .observation_schedule import (
    ObservationAggregation,
    ObservationRecord,
    ObservationSchedule,
    ResolvedObservationRecord,
)
from .phoenix_config import CouponBarrierConfig
from .snowball_config import AccrualConfig, AirbagConfig, BarrierConfig, PayoffConfig


@dataclass
class PhoenixOption(BaseEquityOption):
    """
    Phoenix autocallable structured product with periodic coupon payments.

    A Phoenix option is an autocallable product that:
    - Pays coupons at each observation where spot >= coupon_barrier (or <= for reverse)
    - Can accumulate missed coupons with memory coupon feature
    - Has knock-out (KO) barrier that terminates the product early
    - Has optional knock-in (KI) barrier that changes maturity payoff

    Product Types:
        Standard Phoenix (is_reverse=False):
            - KO barrier: UP (above initial price)
            - KI barrier: DOWN (below initial price)
            - Coupon barrier: Pays when spot >= coupon_barrier
            - Embedded option: PUT (investor is short put on KI)

        Reverse Phoenix (is_reverse=True):
            - KO barrier: DOWN (below initial price)
            - KI barrier: UP (above initial price)
            - Coupon barrier: Pays when spot <= coupon_barrier
            - Embedded option: CALL (investor is short call on KI)

    Payoff Scenarios:
        1. KO triggered: Principal + accumulated coupons + KO rate × accrued time
        2. At maturity, never KO and never KI (V0):
           Principal + accumulated coupons + rebate
        3. At maturity, never KO but KI happened (V1):
           Principal + participation × (Spot - strike), floored by protection

    Key Difference from Snowball:
        - Snowball: Coupons only paid on KO trigger
        - Phoenix: Coupons paid at each observation where coupon barrier is hit

    Core Attributes:
        initial_price: Reference price for payoff calculations
        strike: Strike for the embedded option (put for standard, call for reverse)
        contract_multiplier: Underlying units represented by one contract
        is_reverse: If True, reverse phoenix; if False (default), standard phoenix

    Barrier Attributes (via barrier_config):
        ko_barrier: Knock-out barrier level(s)
        ko_rate: Knock-out return rate(s)
        ki_barrier: Knock-in barrier level(s)

    Coupon Attributes (via coupon_config):
        coupon_barrier: Coupon barrier level(s)
        coupon_rate: Per-period coupon rate
        day_count_convention: Day count convention for year fraction calculation
        memory_coupon: If True, missed coupons accumulate

    Payoff Attributes (via payoff_config):
        rebate_rate: Fixed rebate rate for V0 maturity payoff
        participation_rate: Downside participation after KI
        protection_type: NONE, PARTIAL, or FULL protection
    """

    # Core parameters
    initial_price: float = 0.0
    strike: float = 0.0
    is_reverse: bool = False

    # Option type parameters
    option_type: OptionType = OptionType.PUT
    exercise_type: ExerciseType = ExerciseType.EUROPEAN

    # Date-based maturity
    initial_date: Optional[datetime] = None
    exercise_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    tenor: Optional[float] = None
    maturity: Optional[float] = None
    tenor_end: TenorEnd = TenorEnd.EXERCISE
    annualization_day_count: DayCountConvention = DayCountConvention.ACT_365

    # Configuration objects
    barrier_config: BarrierConfig = field(
        default_factory=lambda: BarrierConfig(ko_barrier=0.0, ko_rate=0.0)
    )
    coupon_config: CouponBarrierConfig = field(
        default_factory=lambda: CouponBarrierConfig(coupon_barrier=0.0)
    )
    payoff_config: PayoffConfig = field(default_factory=PayoffConfig)
    accrual_config: AccrualConfig = field(default_factory=AccrualConfig)
    airbag_config: AirbagConfig = field(default_factory=AirbagConfig)

    def __init__(
        self,
        initial_price: float,
        strike: float,
        barrier_config: BarrierConfig,
        coupon_config: CouponBarrierConfig,
        payoff_config: Optional[PayoffConfig] = None,
        accrual_config: Optional[AccrualConfig] = None,
        airbag_config: Optional[AirbagConfig] = None,
        contract_multiplier: float = 1.0,
        is_reverse: bool = False,
        maturity: Optional[float] = None,
        tenor: Optional[float] = None,
        initial_date: Optional[datetime] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        maturity_date: Optional[datetime] = None,
        tenor_end: TenorEnd = TenorEnd.EXERCISE,
        annualization_day_count: DayCountConvention = DayCountConvention.ACT_365,
    ):
        """
        Initialize Phoenix option.

        Args:
            initial_price: Reference price for payoff calculations
            strike: Strike for the embedded option
            barrier_config: BarrierConfig with KO/KI barrier settings
            coupon_config: CouponBarrierConfig with coupon barrier settings
            payoff_config: PayoffConfig with rebate/protection settings
            accrual_config: AccrualConfig with annualization flags
            airbag_config: AirbagConfig with airbag barrier settings
            contract_multiplier: Underlying units represented by one contract
            is_reverse: If True, creates a reverse phoenix
            maturity: Time to maturity from valuation (years)
            tenor: Contract tenor in years
            initial_date: Product start/issue date
            exercise_date: Expiration date
            settlement_date: Settlement date
            maturity_date: Explicit maturity date
            tenor_end: Tenor end-point selection
            annualization_day_count: Day count basis for annualization

        Raises:
            ValidationError: If parameters are invalid
        """
        # Set configuration objects
        self.barrier_config = barrier_config
        self.coupon_config = coupon_config
        self.payoff_config = (
            payoff_config if payoff_config is not None else PayoffConfig()
        )
        self.accrual_config = (
            accrual_config if accrual_config is not None else AccrualConfig()
        )
        self.airbag_config = (
            airbag_config if airbag_config is not None else AirbagConfig()
        )

        # Set subclass state before BaseEquityOption invokes polymorphic validation.
        self.initial_price = initial_price
        self.is_reverse = is_reverse
        # Set option type based on standard vs reverse
        self.option_type = OptionType.CALL if is_reverse else OptionType.PUT
        self.exercise_type = ExerciseType.EUROPEAN

        # Set base class attributes
        super().__init__(
            strike=strike,
            option_type=self.option_type,
            exercise_type=self.exercise_type,
            maturity=maturity,
            tenor=tenor,
            initial_date=initial_date,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            maturity_date=maturity_date,
            tenor_end=tenor_end,
            annualization_day_count=annualization_day_count,
            contract_multiplier=contract_multiplier,
        )

    def validate(self) -> None:
        """
        Validate Phoenix option parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        self._validate_core_parameters()
        self._validate_maturity_parameters()
        super().validate()
        self._validate_barrier_parameters()
        self._validate_coupon_parameters()
        self._validate_observation_parameters()
        self._validate_payoff_parameters()
        self._validate_accrual_parameters()
        self._build_observation_schedules()

    def _validate_core_parameters(self) -> None:
        """Validate core product parameters."""
        if self.initial_price <= 0:
            raise ValidationError(
                f"Initial price must be positive, got {self.initial_price}"
            )
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")

    def _validate_maturity_parameters(self) -> None:
        """Validate maturity, tenor, and date-related parameters."""
        if self.maturity is None and self.exercise_date is None:
            raise ValidationError("Either maturity or exercise_date must be provided")
        if self.maturity is not None and self.maturity <= 0:
            raise ValidationError(f"Maturity must be positive, got {self.maturity}")
        if self.tenor is not None and self.tenor <= 0:
            raise ValidationError(f"Tenor must be positive, got {self.tenor}")
        if not isinstance(self.tenor_end, TenorEnd):
            raise ValidationError(f"Invalid tenor_end: {self.tenor_end}")
        if not isinstance(self.annualization_day_count, DayCountConvention):
            raise ValidationError(
                f"annualization_day_count must be DayCountConvention, "
                f"got {self.annualization_day_count}"
            )

    def _validate_barrier_parameters(self) -> None:
        """Validate barrier configurations."""
        self._validate_barrier_array(self.barrier_config.ko_barrier, "KO barrier")
        self._validate_rate_array(self.barrier_config.ko_rate, "KO rate")

        if self.barrier_config.ki_barrier is not None:
            self._validate_barrier_array(self.barrier_config.ki_barrier, "KI barrier")
            # Validate continuous KI requires scalar barrier
            if (
                self.barrier_config.ki_continuous
                or self.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            ) and isinstance(self.barrier_config.ki_barrier, list):
                raise ValidationError("Continuous KI requires scalar ki_barrier")

    def _validate_coupon_parameters(self) -> None:
        """Validate coupon barrier configuration."""
        self._validate_barrier_array(
            self.coupon_config.coupon_barrier, "Coupon barrier"
        )
        if self.coupon_config.coupon_rate < 0:
            raise ValidationError(
                f"Coupon rate must be non-negative, got {self.coupon_config.coupon_rate}"
            )
        if not isinstance(self.coupon_config.day_count_convention, DayCountConvention):
            raise ValidationError(
                f"day_count_convention must be DayCountConvention, "
                f"got {self.coupon_config.day_count_convention}"
            )
        if not isinstance(self.coupon_config.coupon_pay_type, CouponPayType):
            raise ValidationError(
                f"coupon_pay_type must be CouponPayType, "
                f"got {self.coupon_config.coupon_pay_type}"
            )

    def _validate_observation_parameters(self) -> None:
        """Validate observation types and discrete observation requirements."""
        if not isinstance(self.barrier_config.ko_observation_type, ObservationType):
            raise ValidationError(
                f"Invalid KO observation type: {self.barrier_config.ko_observation_type}"
            )

        if self.barrier_config.ko_observation_type == ObservationType.DISCRETE:
            if (
                self.barrier_config.ko_observation_schedule is None
                and self.barrier_config.ko_observation_dates is None
            ):
                raise ValidationError(
                    "KO observation dates or schedule required for discrete monitoring"
                )
            self._validate_observation_dates(
                self.barrier_config.ko_observation_dates, "KO"
            )

        # Validate KI observation type
        if not isinstance(self.barrier_config.ki_observation_type, ObservationType):
            raise ValidationError(
                f"Invalid KI observation type: {self.barrier_config.ki_observation_type}"
            )

        # Validate discrete KI observations (if KI barrier provided)
        if (
            self.barrier_config.ki_barrier is not None
            and self.barrier_config.ki_observation_type == ObservationType.DISCRETE
            and not self.barrier_config.ki_continuous
        ):
            if (
                self.barrier_config.ki_observation_schedule is None
                and self.barrier_config.ki_observation_dates is None
            ):
                raise ValidationError(
                    "KI observation dates or schedule required for discrete monitoring"
                )
            self._validate_observation_dates(
                self.barrier_config.ki_observation_dates, "KI"
            )

    def _validate_payoff_parameters(self) -> None:
        """Validate payoff configuration."""
        if not isinstance(self.accrual_config.coupon_pay_type, CouponPayType):
            raise ValidationError(
                f"Invalid coupon pay type: {self.accrual_config.coupon_pay_type}"
            )

        if self.payoff_config.call_rebate_enabled:
            if self.payoff_config.call_strike is None:
                raise ValidationError(
                    "Call strike required when call_rebate_enabled is True"
                )
            if self.payoff_config.call_strike <= 0:
                raise ValidationError(
                    f"Call strike must be positive, got {self.payoff_config.call_strike}"
                )

        if not isinstance(self.payoff_config.protection_type, ProtectionType):
            raise ValidationError(
                f"Invalid protection type: {self.payoff_config.protection_type}"
            )

        if self.payoff_config.participation_rate <= 0:
            raise ValidationError(
                f"Participation rate must be positive, "
                f"got {self.payoff_config.participation_rate}"
            )

    def _validate_accrual_parameters(self) -> None:
        """Validate accrual configuration flags and external factors."""
        for flag_name, flag_value in [
            ("is_annualized", self.accrual_config.is_annualized),
            ("is_annualized_ko", self.accrual_config.is_annualized_ko),
            ("is_annualized_ki", self.accrual_config.is_annualized_ki),
            ("is_annualized_rebate", self.accrual_config.is_annualized_rebate),
        ]:
            if flag_value is not None and not isinstance(flag_value, bool):
                raise ValidationError(f"{flag_name} must be boolean, got {flag_value}")

        accrual_factors = self.accrual_config.accrual_factors
        if accrual_factors is not None:
            ko_obs_len = self.num_ko_observations
            if ko_obs_len and len(accrual_factors) != ko_obs_len:
                raise ValidationError(
                    "accrual_factors length "
                    f"({len(accrual_factors)}) must match KO observation length "
                    f"({ko_obs_len})"
                )

    def _build_observation_schedules(self) -> None:
        """Build ObservationSchedules from observation dates if needed (Legacy)."""
        # Check if we need to build schedules
        needs_ko_schedule = (
            self.barrier_config.ko_observation_schedule is None
            and self.barrier_config.ko_observation_dates is not None
        )

        if not needs_ko_schedule:
            return

        # Build KO schedule (simplified version matching Snowball pattern)
        ko_barriers = (
            self.barrier_config.ko_barrier
            if isinstance(self.barrier_config.ko_barrier, list)
            else None
        )
        ko_rates = (
            self.barrier_config.ko_rate
            if isinstance(self.barrier_config.ko_rate, list)
            else None
        )

        records = []
        for i, t in enumerate(self.barrier_config.ko_observation_dates):
            barrier_val = (
                ko_barriers[i] if ko_barriers else self.barrier_config.ko_barrier
            )
            rate_val = ko_rates[i] if ko_rates else self.barrier_config.ko_rate
            records.append(
                ObservationRecord(
                    observation_time=t,
                    barrier=barrier_val,
                    return_rate=rate_val,
                    is_rate_annualized=False,
                )
            )
        ko_schedule = ObservationSchedule(
            records=records,
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        )

        # Update barrier_config with built schedule
        self.barrier_config = replace(
            self.barrier_config,
            ko_observation_schedule=ko_schedule,
        )

    def time_shift(self, time_bump: float, bumped_date: datetime, pricing_env) -> bool:
        """Shift barrier schedules for theta bumping."""
        dropped_all = super().time_shift(time_bump, bumped_date, pricing_env)
        if dropped_all:
            return True

        if self.barrier_config is not None:
            new_config, dropped_all = self.barrier_config.time_shift(
                time_bump, bumped_date, pricing_env
            )
            self.barrier_config = new_config

        return dropped_all

    def _validate_barrier_array(
        self, barrier: Union[float, List[float]], name: str
    ) -> None:
        """Validate barrier level(s) are positive."""
        if isinstance(barrier, list):
            if not barrier:
                raise ValidationError(f"{name} list cannot be empty")
            for i, b in enumerate(barrier):
                if b <= 0:
                    raise ValidationError(f"{name}[{i}] must be positive, got {b}")
        else:
            if barrier <= 0:
                raise ValidationError(f"{name} must be positive, got {barrier}")

    def _validate_rate_array(self, rate: Union[float, List[float]], name: str) -> None:
        """Validate rate(s) - can be negative for some structures."""
        if isinstance(rate, list):
            for i, r in enumerate(rate):
                if not isinstance(r, (int, float)):
                    raise ValidationError(f"{name}[{i}] must be numeric, got {r}")
        else:
            if not isinstance(rate, (int, float)):
                raise ValidationError(f"{name} must be numeric, got {rate}")

    def _validate_observation_dates(
        self, dates: Optional[List[float]], name: str
    ) -> None:
        """Validate observation dates are non-negative and ordered."""
        if dates is None:
            return
        for i, t in enumerate(dates):
            if t < 0:
                raise ValidationError(
                    f"{name} observation date[{i}] must be non-negative, got {t}"
                )
        # Check ordering
        for i in range(1, len(dates)):
            if dates[i] <= dates[i - 1]:
                raise ValidationError(
                    f"{name} observation dates must be strictly increasing"
                )

    def resolve_ko_observations(self, pricing_env) -> List[ResolvedObservationRecord]:
        """
        Resolve KO observation schedule to concrete times, barriers, and settlement times.

        KO payoffs exclude Phoenix coupon payments, which are handled by pricing engines.
        """
        if self.barrier_config.ko_observation_type != ObservationType.DISCRETE:
            raise ValidationError(
                "resolve_ko_observations currently supports discrete KO monitoring."
            )

        schedule = self.barrier_config.ko_observation_schedule
        if schedule is None:
            raise ValidationError(
                "KO observation schedule is required to resolve KO observations."
            )

        default_barrier = (
            None
            if isinstance(self.barrier_config.ko_barrier, list)
            else self.barrier_config.ko_barrier
        )
        resolved_schedule = schedule.resolve(
            pricing_env=pricing_env,
            default_barrier=default_barrier,
            default_payoff=0.0,
            require_single=True,
        )

        annualized_ko = self._effective_annualized_flag(
            self.accrual_config.is_annualized_ko
        )
        principal_component = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )
        maturity_time: Optional[float] = None
        bus_days_in_year = pricing_env.bus_days_in_year if pricing_env else 252

        ko_records: List[ResolvedObservationRecord] = []
        accrual_factors = self.accrual_config.accrual_factors
        for idx, rec in enumerate(resolved_schedule):
            rate = schedule.records[idx].return_rate
            if rate is None:
                rate = self.get_ko_rate_at(idx)

            if accrual_factors is not None:
                accrual_factor = float(accrual_factors[idx])
            elif annualized_ko:
                schedule_record = schedule.records[idx]
                accrual_start_date = self.initial_date
                if schedule_record.observation_date is not None:
                    if accrual_start_date is None:
                        if pricing_env is None:
                            raise ValidationError(
                                "PricingEnvironment required to resolve KO accrual from observation_date."
                            )
                        accrual_start_date = pricing_env.valuation_date
                    accrual_factor = calculate_year_fraction(
                        accrual_start_date,
                        schedule_record.observation_date,
                        self.annualization_day_count,
                        bus_days_in_year,
                        calendar=getattr(pricing_env, "calendar", None),
                    )
                else:
                    if accrual_start_date is None:
                        accrual_factor = rec.observation_time
                    else:
                        if pricing_env is None:
                            raise ValidationError(
                                "PricingEnvironment required to resolve KO accrual without observation_date."
                            )
                        if pricing_env.valuation_date < accrual_start_date:
                            raise ValidationError(
                                "valuation_date must be on or after initial_date to resolve KO accrual."
                            )
                        if pricing_env.valuation_date == accrual_start_date:
                            initial_to_valuation = 0.0
                        else:
                            initial_to_valuation = calculate_year_fraction(
                                accrual_start_date,
                                pricing_env.valuation_date,
                                self.annualization_day_count,
                                pricing_env.bus_days_in_year,
                                calendar=getattr(pricing_env, "calendar", None),
                            )
                        accrual_factor = initial_to_valuation + rec.observation_time
            else:
                accrual_factor = 1.0

            ko_coupon = (
                self.initial_price * self.contract_multiplier * rate * accrual_factor
            )
            payoff = principal_component + ko_coupon

            settlement_time = rec.settlement_time
            if self.accrual_config.coupon_pay_type == CouponPayType.EXPIRY:
                maturity_time = (
                    maturity_time
                    if maturity_time is not None
                    else self.get_maturity(pricing_env)
                )
                settlement_time = maturity_time

            ko_records.append(
                ResolvedObservationRecord(
                    observation_time=rec.observation_time,
                    barrier=rec.barrier,
                    payoff=payoff,
                    settlement_time=settlement_time,
                )
            )
        return ko_records

    def resolve_ki_observations(self, pricing_env) -> List[ResolvedObservationRecord]:
        """
        Resolve KI observation schedule to times and barrier levels.
        """
        if self.barrier_config.ki_barrier is None:
            raise ValidationError("KI barrier configuration is missing.")
        if (
            self.barrier_config.ki_observation_type != ObservationType.DISCRETE
            or self.barrier_config.ki_continuous
        ):
            raise ValidationError(
                "resolve_ki_observations currently supports discrete KI monitoring."
            )

        schedule = self.barrier_config.ki_observation_schedule
        if schedule is None:
            raise ValidationError(
                "KI observation schedule is required to resolve KI observations."
            )

        default_barrier = (
            None
            if isinstance(self.barrier_config.ki_barrier, list)
            else self.barrier_config.ki_barrier
        )
        resolved_schedule = schedule.resolve(
            pricing_env=pricing_env,
            default_barrier=default_barrier,
            default_payoff=0.0,
            require_single=True,
        )

        return [
            ResolvedObservationRecord(
                observation_time=rec.observation_time,
                barrier=rec.barrier,
                payoff=0.0,
                settlement_time=rec.settlement_time,
            )
            for rec in resolved_schedule
        ]

    def get_ko_observation_profile(self, pricing_env) -> Dict[str, List[Optional[float]]]:
        """Return KO observation attributes for engine consumption."""
        records = self.resolve_ko_observations(pricing_env)
        return {
            "observation_times": [rec.observation_time for rec in records],
            "barriers": [rec.barrier for rec in records],
            "payoffs": [rec.payoff for rec in records],
            "settlement_times": [rec.settlement_time for rec in records],
        }

    def get_ki_observation_profile(self, pricing_env) -> Dict[str, List[Optional[float]]]:
        """Return KI observation attributes for engine consumption."""
        records = self.resolve_ki_observations(pricing_env)
        return {
            "observation_times": [rec.observation_time for rec in records],
            "barriers": [rec.barrier for rec in records],
        }

    # ==================== Coupon Barrier Methods ====================

    def is_coupon_triggered(self, spot: float, observation_idx: int = 0) -> bool:
        """
        Check if coupon barrier is triggered at given spot.

        For standard Phoenix: pays coupon when spot >= coupon_barrier
        For reverse Phoenix: pays coupon when spot <= coupon_barrier

        Args:
            spot: Current spot price
            observation_idx: Index of observation date (for time-varying barriers)

        Returns:
            True if coupon barrier is triggered
        """
        barrier = self._get_barrier_at(
            self.coupon_config.coupon_barrier, observation_idx, "Coupon barrier"
        )
        if self.is_reverse:
            return spot <= barrier
        return spot >= barrier

    def get_coupon_barrier_at(self, observation_idx: int) -> float:
        """Get coupon barrier level at given observation index."""
        return self._get_barrier_at(
            self.coupon_config.coupon_barrier, observation_idx, "Coupon barrier"
        )

    def get_coupon_payoff(
        self,
        observation_idx: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        year_fraction: Optional[float] = None,
    ) -> float:
        """
        Calculate coupon payoff for a single observation period.

        The coupon is calculated as:
            coupon = initial_price × contract_multiplier × coupon_rate × year_fraction

        Args:
            observation_idx: Index of observation date
            start_date: Start date for year fraction calculation
            end_date: End date for year fraction calculation
            year_fraction: Pre-calculated year fraction (overrides date calculation)

        Returns:
            Coupon payoff amount
        """
        if year_fraction is not None:
            dcf = year_fraction
        elif start_date is not None and end_date is not None:
            dcf = calculate_day_count_fraction(
                start_date, end_date, self.coupon_config.day_count_convention
            )
        else:
            # Default to per-period rate without annualization
            dcf = 1.0

        principal = self.initial_price * self.contract_multiplier
        return principal * self.coupon_config.coupon_rate * dcf

    def get_coupon_year_fraction(
        self, start_date: datetime, end_date: datetime
    ) -> float:
        """
        Calculate year fraction for coupon accrual using configured day count convention.

        Args:
            start_date: Period start date
            end_date: Period end date

        Returns:
            Year fraction according to day count convention
        """
        return calculate_day_count_fraction(
            start_date, end_date, self.coupon_config.day_count_convention
        )

    def get_coupon_period_year_fractions(
        self, observation_times: List[float]
    ) -> List[float]:
        """
        Resolve coupon period accrual fractions for each observation.

        If fixed_coupon_year_fraction is configured, use that value for every
        coupon period (e.g., 1/12 for equal monthly coupons). Otherwise, derive
        period fractions from successive observation times.
        """
        if not observation_times:
            return []

        accrual_factors = self.accrual_config.accrual_factors
        if accrual_factors is not None:
            if len(accrual_factors) != len(observation_times):
                raise ValidationError(
                    "accrual_factors length "
                    f"({len(accrual_factors)}) must match observation_times length "
                    f"({len(observation_times)})"
                )
            return [float(factor) for factor in accrual_factors]

        fixed = self.coupon_config.fixed_coupon_year_fraction
        if fixed is not None:
            return [float(fixed) for _ in observation_times]

        yfs: List[float] = [float(observation_times[0])]
        for idx in range(1, len(observation_times)):
            yfs.append(float(observation_times[idx] - observation_times[idx - 1]))
        return yfs

    # ==================== KO/KI Barrier Methods ====================

    def is_ko_triggered(self, spot: float, observation_idx: int = 0) -> bool:
        """
        Check if KO barrier is triggered at given spot.

        For standard Phoenix: KO triggers when spot >= ko_barrier (up barrier)
        For reverse Phoenix: KO triggers when spot <= ko_barrier (down barrier)

        Args:
            spot: Current spot price
            observation_idx: Index of observation date (for time-varying barriers)

        Returns:
            True if KO barrier is triggered
        """
        barrier = self._get_barrier_at(
            self.barrier_config.ko_barrier, observation_idx, "KO barrier"
        )
        if self.is_reverse:
            return spot <= barrier
        return spot >= barrier

    def is_ki_triggered(self, spot: float, observation_idx: int = 0) -> bool:
        """
        Check if KI barrier is triggered at given spot.

        For standard Phoenix: KI triggers when spot <= ki_barrier (down barrier)
        For reverse Phoenix: KI triggers when spot >= ki_barrier (up barrier)

        Args:
            spot: Current spot price
            observation_idx: Index of observation date (for time-varying barriers)

        Returns:
            True if KI barrier is triggered
        """
        if self.barrier_config.ki_barrier is None:
            return False

        barrier = self._get_barrier_at(
            self.barrier_config.ki_barrier, observation_idx, "KI barrier"
        )
        if self.is_reverse:
            return spot >= barrier
        return spot <= barrier

    def _get_barrier_at(
        self, barrier_value: Union[float, List[float]], index: int, barrier_type: str
    ) -> float:
        """Extract barrier value at given observation index."""
        if isinstance(barrier_value, list):
            if index < 0 or index >= len(barrier_value):
                raise ValidationError(
                    f"{barrier_type} observation index {index} out of range"
                )
            return barrier_value[index]
        return barrier_value

    def get_ko_barrier_at(self, observation_idx: int) -> float:
        """Get KO barrier level at given observation index."""
        return self._get_barrier_at(
            self.barrier_config.ko_barrier, observation_idx, "KO barrier"
        )

    def get_ko_rate_at(self, observation_idx: int) -> float:
        """Get KO rate at given observation index."""
        return self._get_barrier_at(
            self.barrier_config.ko_rate, observation_idx, "KO rate"
        )

    def get_ki_barrier_at(self, observation_idx: int) -> Optional[float]:
        """Get KI barrier level at given observation index."""
        if self.barrier_config.ki_barrier is None:
            return None
        return self._get_barrier_at(
            self.barrier_config.ki_barrier, observation_idx, "KI barrier"
        )

    # ==================== Payoff Methods ====================

    def get_ko_payoff(
        self,
        spot: float,
        observation_idx: int,
        accumulated_coupons: float = 0.0,
        pricing_env=None,
    ) -> float:
        """
        Calculate KO payoff including accumulated coupons.

        Args:
            spot: Current spot price
            observation_idx: Index of KO observation
            accumulated_coupons: Total accumulated coupon payments
            pricing_env: PricingEnvironment for time calculations

        Returns:
            KO payoff = principal + ko_coupon + accumulated_coupons
        """
        principal = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )

        # KO coupon based on ko_rate
        ko_rate = self.get_ko_rate_at(observation_idx)

        # Use specific flag or fall back to general is_annualized
        annualized_ko = self._effective_annualized_flag(
            self.accrual_config.is_annualized_ko
        )

        accrual_factors = self.accrual_config.accrual_factors
        if accrual_factors is not None:
            accrual_factor = float(accrual_factors[observation_idx])
        elif annualized_ko:
            # Calculate proper accrual from initial_date
            if self.barrier_config.ko_observation_schedule is not None:
                schedule = self.barrier_config.ko_observation_schedule
                schedule_record = schedule.records[observation_idx]

                if schedule_record.observation_date is not None:
                    # Use date-based calculation
                    accrual_start_date = self.initial_date
                    if accrual_start_date is None:
                        if pricing_env is None:
                            raise ValidationError(
                                "PricingEnvironment required to resolve KO accrual from observation_date."
                            )
                        accrual_start_date = pricing_env.valuation_date

                    bus_days_in_year = (
                        pricing_env.bus_days_in_year if pricing_env else 252
                    )
                    accrual_factor = calculate_year_fraction(
                        accrual_start_date,
                        schedule_record.observation_date,
                        self.annualization_day_count,
                        bus_days_in_year,
                        calendar=getattr(pricing_env, "calendar", None),
                    )
                else:
                    # Fallback to observation_time
                    accrual_factor = schedule_record.observation_time
            elif self.barrier_config.ko_observation_dates is not None:
                # Legacy path: use year fraction directly
                accrual_factor = self.barrier_config.ko_observation_dates[observation_idx]
            else:
                accrual_factor = 1.0
        else:
            accrual_factor = 1.0

        ko_coupon = self.initial_price * self.contract_multiplier * ko_rate * accrual_factor

        # Check if current period coupon is triggered
        current_coupon = 0.0
        if self.is_coupon_triggered(spot, observation_idx):
            coupon_year_fraction = (
                float(accrual_factors[observation_idx])
                if accrual_factors is not None
                else None
            )
            current_coupon = self.get_coupon_payoff(
                observation_idx, year_fraction=coupon_year_fraction
            )

        return principal + ko_coupon + accumulated_coupons + current_coupon

    def get_maturity_payoff_v0(
        self,
        spot: float,
        accumulated_coupons: float = 0.0,
        pricing_env=None,
    ) -> float:
        """
        Calculate maturity payoff for V0 state (not knocked-in, no KO).

        V0 payoff = principal + rebate + accumulated_coupons

        Args:
            spot: Spot price at maturity
            accumulated_coupons: Total accumulated coupon payments
            pricing_env: PricingEnvironment for calculations

        Returns:
            V0 maturity payoff
        """
        principal = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )
        contract_tenor: Optional[float] = None

        if (
            self.payoff_config.call_rebate_enabled
            and self.payoff_config.call_strike is not None
        ):
            # Call-style rebate
            call_strike = self.payoff_config.call_strike
            if self.is_reverse:
                call_payoff = max(call_strike - spot, 0.0)
            else:
                call_payoff = max(spot - call_strike, 0.0)
            rebate = (
                self.payoff_config.call_participation_rate
                * self.contract_multiplier
                * call_payoff
            )
            if self.accrual_config.is_annualized_rebate:
                contract_tenor = contract_tenor or self.get_contract_tenor(pricing_env)
                rebate *= contract_tenor
        else:
            # Fixed rebate
            contract_tenor = contract_tenor or self.get_contract_tenor(pricing_env)
            if self.accrual_config.is_annualized_rebate:
                rebate = (
                    self.payoff_config.rebate_rate
                    * self.initial_price
                    * self.contract_multiplier
                    * contract_tenor
                )
            else:
                rebate = (
                    self.payoff_config.rebate_rate
                    * self.initial_price
                    * self.contract_multiplier
                )

        return principal + rebate + accumulated_coupons

    def get_maturity_payoff_v1(self, spot: float, pricing_env=None) -> float:
        """
        Calculate maturity payoff for V1 state (knocked-in, no KO).

        V1 payoff = principal + participation × downside (floored by protection)

        Args:
            spot: Spot price at maturity
            pricing_env: PricingEnvironment for calculations

        Returns:
            V1 maturity payoff
        """
        principal = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )
        participation_rate = self.payoff_config.participation_rate
        effective_strike = self.strike

        # Check airbag
        airbag_barrier = self.airbag_config.airbag_barrier
        if airbag_barrier is not None:
            if self.is_reverse:
                is_unsafe = spot > airbag_barrier
            else:
                is_unsafe = spot < airbag_barrier

            if is_unsafe:
                participation_rate = self.airbag_config.airbag_participation_rate
                if self.airbag_config.airbag_strike is not None:
                    effective_strike = self.airbag_config.airbag_strike

        # Downside calculation
        if self.is_reverse:
            raw_diff = effective_strike - spot
        else:
            raw_diff = spot - effective_strike

        downside = participation_rate * min(raw_diff, 0.0) * self.contract_multiplier

        # Apply protection floor
        if self.payoff_config.protection_type == ProtectionType.FULL:
            downside = max(downside, 0.0)
        elif self.payoff_config.protection_type == ProtectionType.PARTIAL:
            floor = (
                self.payoff_config.protection_rate
                * self.initial_price
                * self.contract_multiplier
            )
            downside = max(downside, -floor)

        return principal + downside

    def get_payoff(
        self,
        spot: float,
        knocked_in: bool = False,
        accumulated_coupons: float = 0.0,
        pricing_env=None,
    ) -> float:
        """
        Get maturity payoff based on knock-in state.

        Args:
            spot: Spot price at maturity
            knocked_in: Whether KI barrier was triggered
            accumulated_coupons: Total accumulated coupon payments
            pricing_env: PricingEnvironment for calculations

        Returns:
            Maturity payoff (V0 or V1)
        """
        if knocked_in:
            return self.get_maturity_payoff_v1(spot, pricing_env)
        return self.get_maturity_payoff_v0(spot, accumulated_coupons, pricing_env)

    # ==================== Properties ====================

    @property
    def has_ki_barrier(self) -> bool:
        """Check if product has a knock-in barrier."""
        return self.barrier_config.ki_barrier is not None

    @property
    def has_memory_coupon(self) -> bool:
        """Check if memory coupon is enabled."""
        return self.coupon_config.memory_coupon

    @property
    def num_ko_observations(self) -> int:
        """Get number of KO observation dates."""
        if self.barrier_config.ko_observation_schedule is not None:
            return len(self.barrier_config.ko_observation_schedule.records)
        if self.barrier_config.ko_observation_dates is not None:
            return len(self.barrier_config.ko_observation_dates)
        return 0

    @property
    def is_standard(self) -> bool:
        """Check if this is a standard phoenix (not reverse)."""
        return not self.is_reverse

    def get_ko_direction(self) -> BarrierType:
        """Get the direction of the KO barrier."""
        return BarrierType.DOWN_OUT if self.is_reverse else BarrierType.UP_OUT

    def get_ki_direction(self) -> BarrierType:
        """Get the direction of the KI barrier."""
        return BarrierType.UP_IN if self.is_reverse else BarrierType.DOWN_IN

    def _effective_annualized_flag(self, flag: Optional[bool]) -> bool:
        """Resolve specific annualized flag with product-level default."""
        if flag is None:
            return bool(self.accrual_config.is_annualized)
        return flag

    def get_contract_tenor(self, pricing_env=None) -> float:
        """
        Get contract tenor in years.

        Contract tenor is the time from initial date to exercise/maturity,
        used for annualized coupon and rebate calculations.

        Args:
            pricing_env: Optional pricing environment for date-based tenor calculation

        Returns:
            Contract tenor in years
        """
        # Use get_tenor from base class which handles various scenarios
        return self.get_tenor(pricing_env)

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value of the embedded option.

        For standard phoenix (PUT): max(strike - spot, 0)
        For reverse phoenix (CALL): max(spot - strike, 0)
        """
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")

        if self.is_reverse:
            intrinsic = max(spot - self.strike, 0.0)
        else:
            intrinsic = max(self.strike - spot, 0.0)
        return intrinsic * self.contract_multiplier
