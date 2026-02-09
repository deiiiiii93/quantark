"""
Configuration classes for Range Accrual options.

Range Accrual options pay based on the proportion of observations where the
underlying stays within a defined price range, with support for weighted
observations and historical tracking.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union, TYPE_CHECKING

from util.enum import CouponPayType
from util.calendar.day_counter import DayCountConvention
from util.exceptions import ValidationError

if TYPE_CHECKING:
    from priceenv import PricingEnvironment

try:
    from util.calendar import calculate_year_fraction
except ImportError:
    calculate_year_fraction = None


@dataclass
class RangeAccrualObservationRecord:
    """
    Single observation entry for Range Accrual option.

    For past observations: set observed_in_range to True/False
    For future observations: leave observed_in_range as None

    Attributes:
        observation_time: Time as year fraction (optional if observation_date provided)
        observation_date: Calendar date (optional if observation_time provided)
        upper_barrier: Per-observation upper barrier (optional, uses default if None)
        lower_barrier: Per-observation lower barrier (optional, uses default if None)
        weight: Observation weight (default 1.0, use 3.0 for Friday in calendar day convention)
        observed_in_range: Historical observation result (None for future observations)
    """

    observation_time: Optional[float] = None
    observation_date: Optional[datetime] = None
    upper_barrier: Optional[float] = None
    lower_barrier: Optional[float] = None
    weight: float = 1.0
    observed_in_range: Optional[bool] = None

    def resolve_time(self, pricing_env: "PricingEnvironment") -> float:
        """
        Resolve observation time relative to valuation date.

        Args:
            pricing_env: Pricing environment with valuation date

        Returns:
            Year fraction from valuation date to observation date
            (negative if observation is in the past)

        Raises:
            ValidationError: If observation time/date cannot be resolved
        """
        if self.observation_time is not None:
            return self.observation_time
        if self.observation_date is not None:
            if pricing_env is None or calculate_year_fraction is None:
                raise ValidationError(
                    "PricingEnvironment is required to resolve observation_date."
                )

            # Handle past, current and future dates correctly
            if self.observation_date == pricing_env.valuation_date:
                return 0.0
            elif self.observation_date > pricing_env.valuation_date:
                # Future date: positive year fraction
                return calculate_year_fraction(
                    pricing_env.valuation_date,
                    self.observation_date,
                    pricing_env.day_count_convention,
                    pricing_env.bus_days_in_year,
                    calendar=getattr(pricing_env, "calendar", None),
                )
            else:
                # Past date: negative year fraction
                return -calculate_year_fraction(
                    self.observation_date,
                    pricing_env.valuation_date,
                    pricing_env.day_count_convention,
                    pricing_env.bus_days_in_year,
                    calendar=getattr(pricing_env, "calendar", None),
                )
        raise ValidationError(
            "RangeAccrualObservationRecord requires observation_time or observation_date."
        )

    def is_observed(self) -> bool:
        """Check if this observation has historical data."""
        return self.observed_in_range is not None

    def validate(self) -> None:
        """Validate the observation record."""
        if self.observation_time is None and self.observation_date is None:
            raise ValidationError(
                "RangeAccrualObservationRecord must provide observation_time or observation_date."
            )
        if self.weight <= 0:
            raise ValidationError(f"weight must be positive, got {self.weight}")

        # Validate barriers if provided
        if self.upper_barrier is not None and self.upper_barrier <= 0:
            raise ValidationError(
                f"upper_barrier must be positive, got {self.upper_barrier}"
            )
        if self.lower_barrier is not None and self.lower_barrier <= 0:
            raise ValidationError(
                f"lower_barrier must be positive, got {self.lower_barrier}"
            )
        if (
            self.upper_barrier is not None
            and self.lower_barrier is not None
            and self.lower_barrier >= self.upper_barrier
        ):
            raise ValidationError(
                f"lower_barrier ({self.lower_barrier}) must be less than "
                f"upper_barrier ({self.upper_barrier})"
            )


@dataclass(frozen=True)
class RangeAccrualConfig:
    """
    Configuration for Range Accrual option settings.

    Attributes:
        upper_barrier: Upper barrier level(s). Can be scalar or time-varying (list).
        lower_barrier: Lower barrier level(s). Can be scalar or time-varying (list).
        accrual_rate: Per-period accrual rate (e.g., 0.01 for 1% per period).
        is_rate_annualized: Whether accrual_rate is annualized (affects year_fraction calc).
        day_count_convention: Day count convention for year fraction calculation.
        accrual_pay_type: INSTANT (pay per observation) or EXPIRY (pay all at maturity).
        is_reverse: If True, pay when OUTSIDE range instead of inside.
        use_calendar_day_weights: If True, auto-assign Friday=3 weights when generating schedule.
    """

    upper_barrier: Union[float, List[float]]
    lower_barrier: Union[float, List[float]]
    accrual_rate: float = 0.01
    is_rate_annualized: bool = False
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365
    accrual_pay_type: CouponPayType = CouponPayType.EXPIRY
    is_reverse: bool = False
    use_calendar_day_weights: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate barriers are positive
        self._validate_barrier_positive(self.upper_barrier, "upper_barrier")
        self._validate_barrier_positive(self.lower_barrier, "lower_barrier")

        # Validate upper > lower at each observation
        self._validate_barrier_ordering()

        # Validate accrual_rate is non-negative
        if not isinstance(self.accrual_rate, (int, float)):
            raise ValidationError(
                f"accrual_rate must be a number, got {type(self.accrual_rate)}"
            )
        if self.accrual_rate < 0:
            raise ValidationError(
                f"accrual_rate must be non-negative, got {self.accrual_rate}"
            )

        # Validate enums
        if not isinstance(self.accrual_pay_type, CouponPayType):
            raise ValidationError(
                f"accrual_pay_type must be CouponPayType, got {type(self.accrual_pay_type)}"
            )
        if not isinstance(self.day_count_convention, DayCountConvention):
            raise ValidationError(
                f"day_count_convention must be DayCountConvention, "
                f"got {type(self.day_count_convention)}"
            )

        # Validate booleans
        if not isinstance(self.is_rate_annualized, bool):
            raise ValidationError(
                f"is_rate_annualized must be boolean, got {type(self.is_rate_annualized)}"
            )
        if not isinstance(self.is_reverse, bool):
            raise ValidationError(
                f"is_reverse must be boolean, got {type(self.is_reverse)}"
            )
        if not isinstance(self.use_calendar_day_weights, bool):
            raise ValidationError(
                f"use_calendar_day_weights must be boolean, "
                f"got {type(self.use_calendar_day_weights)}"
            )

    @staticmethod
    def _validate_barrier_positive(
        barrier: Union[float, List[float]], name: str
    ) -> None:
        """Validate that barrier level(s) are positive."""
        if isinstance(barrier, list):
            if not barrier:
                raise ValidationError(f"{name} list cannot be empty")
            for i, b in enumerate(barrier):
                if not isinstance(b, (int, float)) or b <= 0:
                    raise ValidationError(
                        f"{name}[{i}] must be positive number, got {b}"
                    )
        else:
            if not isinstance(barrier, (int, float)) or barrier <= 0:
                raise ValidationError(f"{name} must be positive number, got {barrier}")

    def _validate_barrier_ordering(self) -> None:
        """Validate that upper_barrier > lower_barrier at each observation."""
        upper_is_list = isinstance(self.upper_barrier, list)
        lower_is_list = isinstance(self.lower_barrier, list)

        if upper_is_list and lower_is_list:
            # Both are lists - must have same length
            # Cast to list since we checked isinstance above
            upper_list = list(self.upper_barrier)  # type: ignore[arg-type]
            lower_list = list(self.lower_barrier)  # type: ignore[arg-type]
            if len(upper_list) != len(lower_list):
                raise ValidationError(
                    f"upper_barrier and lower_barrier lists must have same length, "
                    f"got {len(upper_list)} and {len(lower_list)}"
                )
            for i, (u, l) in enumerate(zip(upper_list, lower_list)):
                if l >= u:
                    raise ValidationError(
                        f"lower_barrier[{i}] ({l}) must be less than "
                        f"upper_barrier[{i}] ({u})"
                    )
        elif upper_is_list:
            # Upper is list, lower is scalar
            upper_list = list(self.upper_barrier)  # type: ignore[arg-type]
            lower_scalar: float = float(self.lower_barrier)  # type: ignore[arg-type]
            for i, u in enumerate(upper_list):
                if lower_scalar >= u:
                    raise ValidationError(
                        f"lower_barrier ({lower_scalar}) must be less than "
                        f"upper_barrier[{i}] ({u})"
                    )
        elif lower_is_list:
            # Lower is list, upper is scalar
            lower_list = list(self.lower_barrier)  # type: ignore[arg-type]
            upper_scalar: float = float(self.upper_barrier)  # type: ignore[arg-type]
            for i, l in enumerate(lower_list):
                if l >= upper_scalar:
                    raise ValidationError(
                        f"lower_barrier[{i}] ({l}) must be less than "
                        f"upper_barrier ({upper_scalar})"
                    )
        else:
            # Both are scalars
            upper_scalar = self.upper_barrier  # type: ignore[assignment]
            lower_scalar = self.lower_barrier  # type: ignore[assignment]
            if lower_scalar >= upper_scalar:
                raise ValidationError(
                    f"lower_barrier ({lower_scalar}) must be less than "
                    f"upper_barrier ({upper_scalar})"
                )

    def get_upper_barrier(self, obs_idx: int) -> float:
        """Get upper barrier for a specific observation index."""
        if isinstance(self.upper_barrier, list):
            if obs_idx < 0 or obs_idx >= len(self.upper_barrier):
                raise ValidationError(
                    f"obs_idx {obs_idx} out of range for upper_barrier "
                    f"with length {len(self.upper_barrier)}"
                )
            return self.upper_barrier[obs_idx]
        return self.upper_barrier

    def get_lower_barrier(self, obs_idx: int) -> float:
        """Get lower barrier for a specific observation index."""
        if isinstance(self.lower_barrier, list):
            if obs_idx < 0 or obs_idx >= len(self.lower_barrier):
                raise ValidationError(
                    f"obs_idx {obs_idx} out of range for lower_barrier "
                    f"with length {len(self.lower_barrier)}"
                )
            return self.lower_barrier[obs_idx]
        return self.lower_barrier
