"""
Phoenix option implementation.

Phoenix options are autocallable structured products with periodic coupon payments
when spot exceeds a coupon barrier at observation dates. Unlike Snowball options
which only pay coupons on knock-out events, Phoenix options pay coupons at each
observation where the coupon barrier condition is met.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import List, Optional, Union

from asset.equity.product.option.base_equity_option import BaseEquityOption
from util.calendar import calculate_year_fraction
from util.calendar.day_counter import DayCountConvention, calculate_day_count_fraction
from util.enum import (
    BarrierType,
    CouponPayType,
    ExerciseType,
    ObservationType,
    OptionType,
    ProtectionType,
    TenorEnd,
)
from util.exceptions import ValidationError

from .observation_schedule import ObservationSchedule, ObservationRecord, ObservationAggregation
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
        notional: Notional principal (N)
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
        notional: float = 1.0,
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
            notional: Notional principal (default: 1.0)
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
        self.notional = notional
        self.is_reverse = is_reverse

        # Set option type based on standard vs reverse
        self.option_type = OptionType.CALL if is_reverse else OptionType.PUT
        self.exercise_type = ExerciseType.EUROPEAN

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

        self.validate()

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
        self._build_observation_schedules()

    def _validate_core_parameters(self) -> None:
        """Validate core product parameters."""
        if self.initial_price <= 0:
            raise ValidationError(
                f"Initial price must be positive, got {self.initial_price}"
            )
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        if self.notional <= 0:
            raise ValidationError(f"Notional must be positive, got {self.notional}")

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
            coupon = notional × coupon_rate × year_fraction

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

        return self.notional * self.coupon_config.coupon_rate * dcf

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
        principal = self.notional if self.payoff_config.include_principal else 0.0

        # KO coupon based on ko_rate
        ko_rate = self.get_ko_rate_at(observation_idx)

        # Use specific flag or fall back to general is_annualized
        annualized_ko = self._effective_annualized_flag(
            self.accrual_config.is_annualized_ko
        )

        if annualized_ko:
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

        ko_coupon = self.notional * ko_rate * accrual_factor

        return principal + ko_coupon + accumulated_coupons

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
        principal = self.notional if self.payoff_config.include_principal else 0.0
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
                * (self.notional / self.initial_price)
                * call_payoff
            )
            if self.accrual_config.is_annualized_rebate:
                contract_tenor = contract_tenor or self.get_contract_tenor(pricing_env)
                rebate *= contract_tenor
        else:
            # Fixed rebate
            contract_tenor = contract_tenor or self.get_contract_tenor(pricing_env)
            if self.accrual_config.is_annualized_rebate:
                rebate = self.payoff_config.rebate_rate * self.notional * contract_tenor
            else:
                rebate = self.payoff_config.rebate_rate * self.notional

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
        principal = self.notional if self.payoff_config.include_principal else 0.0
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

        downside = (
            participation_rate * min(raw_diff, 0.0) * self.notional / self.initial_price
        )

        # Apply protection floor
        if self.payoff_config.protection_type == ProtectionType.FULL:
            downside = max(downside, 0.0)
        elif self.payoff_config.protection_type == ProtectionType.PARTIAL:
            floor = self.payoff_config.protection_rate * self.notional
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
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)
