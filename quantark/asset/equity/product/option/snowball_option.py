"""
Snowball (autocallable) option implementation.

Snowball options are structured products with knock-in and knock-out barriers,
coupon payments, and various protection/participation features.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

from quantark.asset.equity.product.option.base_equity_option import BaseEquityOption
from quantark.util.calendar import calculate_year_fraction
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import (
    BarrierType,
    CouponPayType,
    ExerciseType,
    ObservationAggregation,
    ObservationType,
    OptionType,
    ProtectionType,
    TenorEnd,
)
from quantark.util.exceptions import ValidationError

from .observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
    PricingEnv,
    ResolvedObservationRecord,
)
from .snowball_config import AccrualConfig, AirbagConfig, BarrierConfig, PayoffConfig


@dataclass
class SnowballOption(BaseEquityOption):
    """
    Snowball (autocallable) structured product with knock-in and knock-out barriers.

    A snowball option is an autocallable product that:
    - Pays coupons if knock-out (KO) barrier is triggered (product terminates)
    - Switches to knock-in (KI) state if KI barrier is breached
    - Has different payoffs at maturity depending on KO/KI status

    Product Types:
        Standard Snowball (is_reverse=False):
            - KO barrier: UP (above initial price)
            - KI barrier: DOWN (below initial price)
            - Embedded option: PUT (investor is short put on KI)
            - option_type: PUT

        Reverse Snowball (is_reverse=True):
            - KO barrier: DOWN (below initial price)
            - KI barrier: UP (above initial price)
            - Embedded option: CALL (investor is short call on KI)
            - option_type: CALL

    Payoff Scenarios:
        1. KO triggered: Principal (if included) + KO rate × accrued time
        2. At maturity, never KO and never KI (V0):
           Principal (if included) + fixed rebate or call-style rebate
        3. At maturity, never KO but KI happened (V1):
           Principal (if included) + participation × (Spot - strike), floored by protection

    Core Attributes:
        initial_price: Reference price for payoff calculations
        strike: Strike for the embedded option (put for standard, call for reverse)
        contract_multiplier: Underlying units represented by one contract
        is_reverse: If True, reverse snowball; if False (default), standard snowball
        option_type: CALL for reverse, PUT for standard (auto-set based on is_reverse)
        exercise_type: EUROPEAN (autocallables are European-style)

    Barrier Attributes (via barrier_config):
        ko_barrier: Knock-out barrier level(s)
        ko_rate: Knock-out return rate(s)
        ko_observation_type: DISCRETE or CONTINUOUS monitoring for KO
        ki_barrier: Knock-in barrier level(s)
        ki_observation_type: DISCRETE or CONTINUOUS monitoring for KI
        disable_ko_after_ki: If True, disable KO after KI is triggered

    Payoff Attributes (via payoff_config):
        rebate_rate: Fixed rebate rate for V0 maturity payoff
        call_rebate_enabled: If True, use call-style rebate instead of fixed
        include_principal: Whether principal is part of payouts
        participation_rate: Downside participation after KI
        protection_type: NONE, PARTIAL, or FULL protection

    Accrual Attributes (via accrual_config):
        coupon_pay_type: INSTANT (at KO date) or EXPIRY (discounted to maturity)
        is_annualized: Flag for annualized accruals

    Airbag Attributes (via airbag_config):
        airbag_barrier: Barrier level for airbag protection
    """

    # Core parameters
    initial_price: float = 0.0
    strike: float = 0.0
    is_reverse: bool = False

    # Option type parameters (inherited from BaseEquityOption)
    # For standard snowball: embedded option is PUT (short put exposure on KI)
    # For reverse snowball: embedded option is CALL (short call exposure on KI)
    option_type: OptionType = OptionType.PUT
    exercise_type: ExerciseType = ExerciseType.EUROPEAN

    # Date-based maturity (inherited from base class concept, defined here as dataclass fields)
    initial_date: Optional[datetime] = None
    exercise_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    tenor: Optional[float] = None
    maturity: Optional[float] = None
    tenor_end: TenorEnd = TenorEnd.EXERCISE
    annualization_day_count: DayCountConvention = DayCountConvention.ACT_365

    # Configuration objects (clean API)
    barrier_config: BarrierConfig = field(
        default_factory=lambda: BarrierConfig(ko_barrier=0.0, ko_rate=0.0)
    )
    payoff_config: PayoffConfig = field(default_factory=PayoffConfig)
    accrual_config: AccrualConfig = field(default_factory=AccrualConfig)
    airbag_config: AirbagConfig = field(default_factory=AirbagConfig)

    def __init__(
        self,
        initial_price: float,
        strike: float,
        barrier_config: BarrierConfig,
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
        Initialize Snowball option.

        Args:
            initial_price: Reference price for payoff calculations
            strike: Strike for the embedded option (put for standard, call for reverse)
            barrier_config: BarrierConfig with KO/KI barrier settings (required)
            payoff_config: PayoffConfig with rebate/protection/participation settings
            accrual_config: AccrualConfig with annualization flags
            airbag_config: AirbagConfig with airbag barrier settings
            contract_multiplier: Underlying units represented by one contract
            is_reverse: If True, creates a reverse snowball with embedded call option;
                       if False (default), creates standard snowball with embedded put
            maturity: Time to maturity from valuation (years)
            tenor: Contract tenor in years (issue to expiry)
            initial_date: Product start/issue date
            exercise_date: Expiration date
            settlement_date: Settlement date
            maturity_date: Explicit maturity date
            tenor_end: Tenor end-point selection
            annualization_day_count: Day count basis

        Raises:
            ValidationError: If parameters are invalid

        Note:
            Standard Snowball (is_reverse=False):
                - KO barrier is UP (above initial price)
                - KI barrier is DOWN (below initial price)
                - Embedded option is PUT (investor is short put on KI)
                - V1 payoff: participation × (Spot - Strike), typically negative when spot < strike

            Reverse Snowball (is_reverse=True):
                - KO barrier is DOWN (below initial price)
                - KI barrier is UP (above initial price)
                - Embedded option is CALL (investor is short call on KI)
                - V1 payoff: participation × (Strike - Spot), typically negative when spot > strike
        """
        # Set base class attributes
        self.initial_date = initial_date
        self.exercise_date = exercise_date
        self.settlement_date = settlement_date
        self.maturity_date = maturity_date
        self.tenor = tenor
        self.maturity = maturity
        self.tenor_end = tenor_end
        self.annualization_day_count = annualization_day_count

        # Set core attributes
        self.initial_price = initial_price
        self.strike = strike
        self.contract_multiplier = contract_multiplier
        self.is_reverse = is_reverse

        # Set option type based on standard vs reverse snowball
        # Standard snowball: embedded PUT (short put exposure on KI)
        # Reverse snowball: embedded CALL (short call exposure on KI)
        self.option_type = OptionType.CALL if is_reverse else OptionType.PUT
        self.exercise_type = ExerciseType.EUROPEAN

        # Set configuration objects
        self.barrier_config = barrier_config
        self.payoff_config = (
            payoff_config if payoff_config is not None else PayoffConfig()
        )
        self.accrual_config = (
            accrual_config if accrual_config is not None else AccrualConfig()
        )
        self.airbag_config = (
            airbag_config if airbag_config is not None else AirbagConfig()
        )

        self.validate()

    def validate(self) -> None:
        """
        Validate Snowball option parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        self._validate_core_parameters()
        self._validate_maturity_parameters()
        super().validate()
        self._validate_barrier_parameters()
        self._validate_observation_parameters()
        self._validate_payoff_parameters()
        self._validate_accrual_parameters()
        self._build_observation_schedules()

    def _validate_core_parameters(self) -> None:
        """Validate core product parameters (initial_price, strike)."""
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
                f"annualization_day_count must be DayCountConvention, got {self.annualization_day_count}"
            )
        if self.tenor_end == TenorEnd.SETTLEMENT and self.settlement_date is None:
            raise ValidationError(
                "settlement_date required when tenor_end is SETTLEMENT"
            )
        if (
            self.tenor_end == TenorEnd.MATURITY
            and self.maturity_date is None
            and self.exercise_date is None
        ):
            raise ValidationError(
                "maturity_date or exercise_date required when tenor_end is MATURITY"
            )
        if (
            self.tenor is None
            and any(
                [
                    self.accrual_config.is_annualized_ko,
                    self.accrual_config.is_annualized_ki,
                    self.accrual_config.is_annualized_rebate,
                ]
            )
            and self.initial_date is None
            and self.settlement_date is None
        ):
            raise ValidationError(
                "initial_date or settlement_date required for annualized accruals when tenor is not provided"
            )

    def _validate_barrier_parameters(self) -> None:
        """Validate barrier configurations (KO/KI barriers and rates)."""
        # Validate barrier config
        self._validate_barrier_array(self.barrier_config.ko_barrier, "KO barrier")
        self._validate_rate_array(self.barrier_config.ko_rate, "KO rate")

        # Validate KO barrier/rate array lengths match observation dates
        ko_obs_len = self._get_observation_length(BarrierType.UP_OUT)
        if ko_obs_len is not None:
            self._validate_array_length(
                self.barrier_config.ko_barrier, ko_obs_len, "KO barrier"
            )
            self._validate_array_length(
                self.barrier_config.ko_rate, ko_obs_len, "KO rate"
            )

        # Validate KI barrier if provided
        if self.barrier_config.ki_barrier is not None:
            self._validate_barrier_array(self.barrier_config.ki_barrier, "KI barrier")
            if (
                self.barrier_config.ki_continuous
                or self.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            ) and isinstance(self.barrier_config.ki_barrier, list):
                raise ValidationError("Continuous KI requires scalar ki_barrier")
            ki_obs_len = self._get_observation_length(BarrierType.DOWN_IN)
            if ki_obs_len is not None:
                self._validate_array_length(
                    self.barrier_config.ki_barrier, ki_obs_len, "KI barrier"
                )

    def _validate_observation_parameters(self) -> None:
        """Validate observation types and discrete observation requirements."""
        # Validate observation types
        if not isinstance(self.barrier_config.ko_observation_type, ObservationType):
            raise ValidationError(
                f"Invalid KO observation type: {self.barrier_config.ko_observation_type}"
            )
        if not isinstance(self.barrier_config.ki_observation_type, ObservationType):
            raise ValidationError(
                f"Invalid KI observation type: {self.barrier_config.ki_observation_type}"
            )

        # Validate discrete KO observations
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
        """Validate payoff configuration (rebate, protection, participation)."""
        # Validate accrual config
        if not isinstance(self.accrual_config.coupon_pay_type, CouponPayType):
            raise ValidationError(
                f"Invalid coupon pay type: {self.accrual_config.coupon_pay_type}"
            )

        # Validate call rebate parameters
        if self.payoff_config.call_rebate_enabled:
            if self.payoff_config.call_strike is None:
                raise ValidationError(
                    "Call strike required when call_rebate_enabled is True"
                )
            if self.payoff_config.call_strike <= 0:
                raise ValidationError(
                    f"Call strike must be positive, got {self.payoff_config.call_strike}"
                )
            if self.payoff_config.call_participation_rate <= 0:
                raise ValidationError(
                    f"Call participation rate must be positive, got {self.payoff_config.call_participation_rate}"
                )

        # Validate protection parameters
        if not isinstance(self.payoff_config.protection_type, ProtectionType):
            raise ValidationError(
                f"Invalid protection type: {self.payoff_config.protection_type}"
            )
        if self.payoff_config.protection_type == ProtectionType.PARTIAL:
            if (
                self.payoff_config.protection_rate < 0
                or self.payoff_config.protection_rate > 1
            ):
                raise ValidationError(
                    f"Protection rate must be in [0, 1], got {self.payoff_config.protection_rate}"
                )

        # Validate participation rate
        if self.payoff_config.participation_rate <= 0:
            raise ValidationError(
                f"Participation rate must be positive, got {self.payoff_config.participation_rate}"
            )

    def _validate_accrual_parameters(self) -> None:
        """Validate accrual configuration flags."""
        # Validate accrual config flags
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
            ko_obs_len = self._get_observation_length(BarrierType.UP_OUT)
            if ko_obs_len is not None and len(accrual_factors) != ko_obs_len:
                raise ValidationError(
                    "accrual_factors length "
                    f"({len(accrual_factors)}) must match KO observation length "
                    f"({ko_obs_len})"
                )

    def _validate_barrier_array(
        self, barrier: Union[float, List[float]], name: str
    ) -> None:
        """Validate barrier level(s) are positive."""
        if isinstance(barrier, list):
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

    def _validate_array_length(
        self, value: Union[float, List[float]], expected_len: int, name: str
    ) -> None:
        """Validate array length matches observation dates if it's an array."""
        if isinstance(value, list) and len(value) != expected_len:
            raise ValidationError(
                f"{name} array length ({len(value)}) must match "
                f"observation dates length ({expected_len})"
            )

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

    def _get_observation_length(self, barrier_type: BarrierType) -> Optional[int]:
        """Get the number of observation dates for a barrier type.

        Args:
            barrier_type: BarrierType enum (e.g., UP_OUT, DOWN_IN)

        Returns:
            Number of observation dates or None if not available
        """
        if barrier_type.is_knock_out:
            if self.barrier_config.ko_observation_schedule is not None:
                return len(self.barrier_config.ko_observation_schedule.records)
            if self.barrier_config.ko_observation_dates is not None:
                return len(self.barrier_config.ko_observation_dates)
        elif barrier_type.is_knock_in:
            if self.barrier_config.ki_observation_schedule is not None:
                return len(self.barrier_config.ki_observation_schedule.records)
            if self.barrier_config.ki_observation_dates is not None:
                return len(self.barrier_config.ki_observation_dates)
        return None

    def _build_observation_schedules(self) -> None:
        """Build ObservationSchedules from observation dates if needed (Legacy).

        Note: This method works around frozen config classes by recreating them with schedules.
        """
        # Check if we need to build schedules
        needs_ko_schedule = (
            self.barrier_config.ko_observation_schedule is None
            and self.barrier_config.ko_observation_dates is not None
        )
        needs_ki_schedule = (
            self.barrier_config.ki_barrier is not None
            and self.barrier_config.ki_observation_schedule is None
            and self.barrier_config.ki_observation_dates is not None
        )

        if not needs_ko_schedule and not needs_ki_schedule:
            return  # Nothing to build

        # Build KO schedule
        ko_schedule = self.barrier_config.ko_observation_schedule
        if needs_ko_schedule:
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

        # Build KI schedule
        ki_schedule = self.barrier_config.ki_observation_schedule
        if needs_ki_schedule:
            ki_barriers = (
                self.barrier_config.ki_barrier
                if isinstance(self.barrier_config.ki_barrier, list)
                else None
            )

            records = []
            for i, t in enumerate(self.barrier_config.ki_observation_dates):
                barrier_val = (
                    ki_barriers[i] if ki_barriers else self.barrier_config.ki_barrier
                )
                records.append(
                    ObservationRecord(
                        observation_time=t,
                        barrier=barrier_val,
                    )
                )
            ki_schedule = ObservationSchedule(
                records=records,
                aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
            )

        # Create new barrier config with schedules (workaround for frozen dataclass)
        from dataclasses import replace

        self.barrier_config = replace(
            self.barrier_config,
            ko_observation_schedule=ko_schedule,
            ki_observation_schedule=(
                ki_schedule
                if needs_ki_schedule
                else self.barrier_config.ki_observation_schedule
            ),
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

    def get_maturity(self, pricing_env: PricingEnv = None) -> float:
        """
        Get time to maturity in years.

        Args:
            pricing_env: Optional pricing environment for date-based maturity

        Returns:
            Time to maturity in years

        Raises:
            ValidationError: If maturity cannot be determined
        """
        return super().get_maturity(pricing_env)

    def get_contract_tenor(self, pricing_env: PricingEnv = None) -> float:
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

    def get_payoff(
        self, spot: float, pricing_env: PricingEnv = None, **kwargs
    ) -> float:
        """
        Calculate payoff at maturity (V0 or V1 state).

        This method calculates the payoff assuming no KO has occurred.
        The actual payoff depends on the path history (KO triggered, KI triggered).

        For full path-dependent payoff calculation, use
        resolve_ko_observations, get_maturity_payoff_v0, or
        get_maturity_payoff_v1 methods.

        Args:
            spot: Spot price at maturity
            pricing_env: Optional pricing environment for date-based maturity
            **kwargs:
                knocked_in: Whether KI was triggered (default: False)

        Returns:
            Payoff at maturity
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        knocked_in = kwargs.get("knocked_in", False)

        if knocked_in:
            return self.get_maturity_payoff_v1(spot, pricing_env=pricing_env)
        else:
            return self.get_maturity_payoff_v0(spot, pricing_env=pricing_env)

    def get_maturity_payoff_v0(
        self, spot: float, pricing_env: PricingEnv = None
    ) -> float:
        """
        Calculate payoff at maturity when never KO and never KI (V0 state).

        Args:
            spot: Spot price at maturity
            pricing_env: Optional pricing environment for resolving maturity

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
            call_payoff = max(spot - self.payoff_config.call_strike, 0.0)
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

        return principal + rebate

    def get_maturity_payoff_v1(
        self, spot: float, pricing_env: PricingEnv = None
    ) -> float:
        """
        Calculate payoff at maturity when never KO but KI happened (V1 state).

        Args:
            spot: Spot price at maturity
            pricing_env: Optional pricing environment for resolving maturity when annualizing KI return

        Returns:
            V1 maturity payoff
        """
        principal = (
            self.initial_price * self.contract_multiplier
            if self.payoff_config.include_principal
            else 0.0
        )

        # Determine if airbag logic applies
        airbag_barrier = self.airbag_config.airbag_barrier

        # Default to standard payoff configuration
        participation_rate = self.payoff_config.participation_rate
        effective_strike = self.strike

        if airbag_barrier is not None:
            # Standard snowball: airbag applies (unsafe) when spot < airbag_barrier
            # Reverse snowball: airbag applies (unsafe) when spot > airbag_barrier
            if self.is_reverse:
                is_unsafe = spot > airbag_barrier
            else:
                is_unsafe = spot < airbag_barrier

            if is_unsafe:
                # In unsafe zone, use airbag participation and strike
                participation_rate = self.airbag_config.airbag_participation_rate
                effective_strike = (
                    self.airbag_config.airbag_strike
                    if self.airbag_config.airbag_strike is not None
                    else self.strike
                )
            # When not in unsafe zone, use standard participation rate (already set above)

        # Downside participation
        # Standard: Short Put (loss if spot < strike) -> downside = spot - strike
        # Reverse: Short Call (loss if spot > strike) -> downside = strike - spot
        if self.is_reverse:
            raw_diff = effective_strike - spot
        else:
            raw_diff = spot - effective_strike

        downside = (
            participation_rate * min(raw_diff, 0.0) * self.contract_multiplier
        )
        if self.accrual_config.is_annualized_ki:
            contract_tenor = self.get_contract_tenor(pricing_env)
            downside *= contract_tenor

        # Apply protection floor
        if self.payoff_config.protection_type == ProtectionType.FULL:
            # Full protection: can't lose more than principal (if included)
            floor = 0.0
            downside = max(downside, -floor)
        elif self.payoff_config.protection_type == ProtectionType.PARTIAL:
            # Partial protection: floor at -protection_rate × N
            floor = (
                self.payoff_config.protection_rate
                * self.initial_price
                * self.contract_multiplier
            )
            downside = max(downside, -floor)

        return principal + downside

    def is_ko_triggered(self, spot: float, observation_idx: int = 0) -> bool:
        """
        Check if KO barrier would be triggered at given spot.

        Args:
            spot: Current spot price
            observation_idx: Index of observation date (for time-varying barriers)

        Returns:
            True if KO barrier is triggered (spot >= KO barrier for up barrier)
        """
        barrier = self._get_barrier_at(
            self.barrier_config.ko_barrier, observation_idx, "KO barrier"
        )
        return spot >= barrier

    def is_ki_triggered(self, spot: float, observation_idx: int = 0) -> bool:
        """
        Check if KI barrier would be triggered at given spot.

        Args:
            spot: Current spot price
            observation_idx: Index of observation date (for time-varying barriers)

        Returns:
            True if KI barrier is triggered (spot <= KI barrier for down barrier)
        """
        if self.barrier_config.ki_barrier is None:
            return False

        barrier = self._get_barrier_at(
            self.barrier_config.ki_barrier, observation_idx, "KI barrier"
        )
        return spot <= barrier

    def _get_barrier_at(
        self, barrier_value: Union[float, List[float]], index: int, barrier_type: str
    ) -> float:
        """Extract barrier value at given observation index.

        Args:
            barrier_value: Single barrier or list of barriers
            index: Observation index
            barrier_type: Type description for error messages (e.g., "KO", "KI")

        Returns:
            Barrier level at the specified index
        """
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
            self.barrier_config.ko_barrier, observation_idx, "KO"
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
            self.barrier_config.ki_barrier, observation_idx, "KI"
        )

    def get_ko_direction(self) -> BarrierType:
        """
        Get the direction of the KO barrier.

        Returns:
            BarrierType enum indicating the direction (UP_OUT or DOWN_OUT).
        """
        return BarrierType.DOWN_OUT if self.is_reverse else BarrierType.UP_OUT

    def get_ki_direction(self) -> BarrierType:
        """
        Get the direction of the KI barrier.

        Returns:
            BarrierType enum indicating the direction (UP_IN or DOWN_IN).
        """
        return BarrierType.UP_IN if self.is_reverse else BarrierType.DOWN_IN

    @property
    def has_ki_barrier(self) -> bool:
        """Check if product has a knock-in barrier."""
        return self.barrier_config.ki_barrier is not None

    @property
    def num_ko_observations(self) -> int:
        """Get number of KO observation dates."""
        if self.barrier_config.ko_observation_schedule is not None:
            return len(self.barrier_config.ko_observation_schedule.records)
        if self.barrier_config.ko_observation_dates is not None:
            return len(self.barrier_config.ko_observation_dates)
        return 0

    @property
    def num_ki_observations(self) -> int:
        """Get number of KI observation dates."""
        if self.barrier_config.ki_observation_schedule is not None:
            return len(self.barrier_config.ki_observation_schedule.records)
        if self.barrier_config.ki_observation_dates is not None:
            return len(self.barrier_config.ki_observation_dates)
        return 0

    @property
    def is_standard(self) -> bool:
        """
        Check if this is a standard snowball (not reverse).

        Returns:
            True if standard snowball (embedded put), False if reverse (embedded call)
        """
        return not self.is_reverse

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value of the embedded option.

        For standard snowball (PUT): max(strike - spot, 0)
        For reverse snowball (CALL): max(spot - strike, 0)

        Note: This represents the intrinsic value of the embedded option component,
        not the full V1 payoff which includes participation and protection.

        Args:
            spot: Current spot price

        Returns:
            Intrinsic value (non-negative)

        Raises:
            ValidationError: If spot is negative
        """
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")

        if self.is_reverse:
            intrinsic = max(spot - self.strike, 0.0)
        else:
            intrinsic = max(self.strike - spot, 0.0)
        return intrinsic * self.contract_multiplier

    def _effective_annualized_flag(self, flag: Optional[bool]) -> bool:
        """Resolve specific annualized flag with product-level default."""
        if flag is None:
            return bool(self.accrual_config.is_annualized)
        return flag

    def resolve_ko_observations(self, pricing_env) -> List[ResolvedObservationRecord]:
        """
        Resolve KO observation schedule to concrete times, barriers, payoffs, and settlement times.

        The payoff includes principal (when configured) plus KO coupon scaled by annualization settings.
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
        bus_days_in_year = (
            pricing_env.bus_days_in_year if pricing_env is not None else 252
        )

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
            coupon_payoff = (
                self.initial_price * self.contract_multiplier * rate * accrual_factor
            )
            payoff = principal_component + coupon_payoff

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
        Resolve KI observation schedule to times and barrier levels (no immediate payoff).
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

    def get_ko_observation_profile(
        self, pricing_env
    ) -> Dict[str, List[Optional[float]]]:
        """
        Convenience helper returning KO observation attributes for engine consumption.
        """
        records = self.resolve_ko_observations(pricing_env)
        return {
            "observation_times": [rec.observation_time for rec in records],
            "barriers": [rec.barrier for rec in records],
            "payoffs": [rec.payoff for rec in records],
            "settlement_times": [rec.settlement_time for rec in records],
        }

    def get_ki_observation_profile(
        self, pricing_env
    ) -> Dict[str, List[Optional[float]]]:
        """
        Convenience helper returning KI observation attributes for engine consumption.
        """
        ki_continuous = (
            self.barrier_config.ki_continuous
            or self.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        if ki_continuous:
            if self.barrier_config.ki_barrier is None:
                raise ValidationError("KI barrier configuration is missing.")
            if isinstance(self.barrier_config.ki_barrier, list):
                raise ValidationError("Continuous KI requires scalar ki_barrier")
            # For continuous KI, the engine generates its own time grid.
            # We return the base KI barrier as a scalar (in a list for consistency)
            # and empty lists for other attributes.
            return {
                "observation_times": [],
                "barriers": [self.barrier_config.ki_barrier],
                "payoffs": [],
                "settlement_times": [],
            }
        records = self.resolve_ki_observations(pricing_env)
        return {
            "observation_times": [rec.observation_time for rec in records],
            "barriers": [rec.barrier for rec in records],
            "payoffs": [rec.payoff for rec in records],
            "settlement_times": [rec.settlement_time for rec in records],
        }

    def cache_key(self) -> Dict[str, object]:
        def _serialize_dt(value: Optional[datetime]) -> Optional[str]:
            return value.isoformat() if isinstance(value, datetime) else None

        def _serialize_enum(value) -> Optional[str]:
            return value.name if hasattr(value, "name") else None

        def _serialize_schedule(
            schedule: Optional[ObservationSchedule],
        ) -> Optional[Dict[str, object]]:
            if schedule is None:
                return None
            frequency = schedule.frequency
            if hasattr(frequency, "name"):
                frequency_value = frequency.name
            else:
                frequency_value = frequency
            return {
                "aggregation_mode": _serialize_enum(schedule.aggregation_mode),
                "frequency": frequency_value,
                "records": [_serialize_value(rec) for rec in schedule.records],
            }

        def _serialize_value(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, datetime):
                return value.isoformat()
            if hasattr(value, "name"):
                return value.name
            if isinstance(value, ObservationSchedule):
                return _serialize_schedule(value)
            if is_dataclass(value):
                return {
                    f.name: _serialize_value(getattr(value, f.name))
                    for f in fields(value)
                }
            if isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_serialize_value(v) for v in value]
            return repr(value)

        key = {f.name: _serialize_value(getattr(self, f.name)) for f in fields(self)}
        key["contract_multiplier"] = _serialize_value(self.contract_multiplier)
        return key

    def __repr__(self) -> str:
        ko_barrier_str = (
            f"{self.barrier_config.ko_barrier[0]:.4f}..."
            if isinstance(self.barrier_config.ko_barrier, list)
            else f"{self.barrier_config.ko_barrier:.4f}"
        )
        ko_rate_str = (
            f"{self.barrier_config.ko_rate[0]:.4f}..."
            if isinstance(self.barrier_config.ko_rate, list)
            else f"{self.barrier_config.ko_rate:.4f}"
        )
        ki_barrier_str = "None"
        if self.barrier_config.ki_barrier is not None:
            ki_barrier_str = (
                f"{self.barrier_config.ki_barrier[0]:.4f}..."
                if isinstance(self.barrier_config.ki_barrier, list)
                else f"{self.barrier_config.ki_barrier:.4f}"
            )

        ko_obs_desc = (
            f"{self.barrier_config.ko_observation_type.name.lower()}-{self.num_ko_observations}obs"
            if self.num_ko_observations
            else self.barrier_config.ko_observation_type.name.lower()
        )
        ki_obs_desc = (
            f"{self.barrier_config.ki_observation_type.name.lower()}-{self.num_ki_observations}obs"
            if self.num_ki_observations
            else self.barrier_config.ki_observation_type.name.lower()
        )

        protection = self.payoff_config.protection_type.name.lower()
        pay_timing = self.accrual_config.coupon_pay_type.name.lower()
        principal_flag = "inclN" if self.payoff_config.include_principal else "exN"

        return (
            f"SnowballOption("
            f"S0={self.initial_price:.4f}, K={self.strike:.4f}, "
            f"mult={self.contract_multiplier:.4f}, "
            f"KO={ko_barrier_str} [{ko_obs_desc}] @rate={ko_rate_str}, "
            f"KI={ki_barrier_str} [{ki_obs_desc}], "
            f"pay={pay_timing}, protection={protection}, {principal_flag})"
        )
