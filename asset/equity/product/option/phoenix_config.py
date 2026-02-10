"""
Configuration classes for Phoenix options.

Phoenix options are autocallable structured products with periodic coupon payments
when spot exceeds a coupon barrier at observation dates.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from util.enum import CouponPayType
from util.calendar.day_counter import DayCountConvention
from util.exceptions import ValidationError


@dataclass(frozen=True)
class CouponBarrierConfig:
    """
    Configuration for coupon barrier settings (Phoenix-specific).

    The coupon barrier determines when periodic coupons are paid. At each observation
    date, if the spot price is above (or below for reverse) the coupon barrier, a
    coupon is paid based on the coupon_rate and day count convention.

    Attributes:
        coupon_barrier: Coupon barrier level(s). Can be scalar or time-varying (list).
                       For standard Phoenix: pays coupon when spot >= coupon_barrier.
                       For reverse Phoenix: pays coupon when spot <= coupon_barrier.
        coupon_rate: Per-period coupon rate (e.g., 0.01 for 1% per period).
        coupon_pay_type: INSTANT (pay at observation) or EXPIRY (accumulate to maturity).
        day_count_convention: Day count convention for year fraction calculation.
                             Supported: ACT_365, THIRTY_360_US, THIRTY_360_EUROPEAN, etc.
        memory_coupon: If True, missed coupons accumulate and are paid when barrier
                      is hit later. If False, only current period coupon is paid.
        fixed_coupon_year_fraction: Optional fixed year fraction used for each coupon
                      period (e.g., 1/12 for equal monthly coupons). If None, engines
                      use observation-time differences.
    """

    # Coupon barrier level(s)
    coupon_barrier: Union[float, List[float]]

    # Coupon rate per period
    coupon_rate: float = 0.01

    # Payment timing
    coupon_pay_type: CouponPayType = CouponPayType.INSTANT

    # Day count convention for year fraction calculation
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365

    # Memory coupon feature
    memory_coupon: bool = True

    # Optional fixed accrual per coupon period (e.g. 1/12)
    fixed_coupon_year_fraction: Optional[float] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate coupon_barrier is positive
        self._validate_barrier_positive(self.coupon_barrier, "coupon_barrier")

        # Validate coupon_rate is non-negative
        if not isinstance(self.coupon_rate, (int, float)):
            raise ValidationError(
                f"coupon_rate must be a number, got {type(self.coupon_rate)}"
            )
        if self.coupon_rate < 0:
            raise ValidationError(f"coupon_rate must be non-negative, got {self.coupon_rate}")

        # Validate coupon_pay_type is enum
        if not isinstance(self.coupon_pay_type, CouponPayType):
            raise ValidationError(
                f"coupon_pay_type must be CouponPayType, got {type(self.coupon_pay_type)}"
            )

        # Validate day_count_convention is enum
        if not isinstance(self.day_count_convention, DayCountConvention):
            raise ValidationError(
                f"day_count_convention must be DayCountConvention, "
                f"got {type(self.day_count_convention)}"
            )

        # Validate memory_coupon is boolean
        if not isinstance(self.memory_coupon, bool):
            raise ValidationError(
                f"memory_coupon must be boolean, got {type(self.memory_coupon)}"
            )

        # Validate fixed coupon accrual if provided
        if self.fixed_coupon_year_fraction is not None:
            if not isinstance(self.fixed_coupon_year_fraction, (int, float)):
                raise ValidationError(
                    "fixed_coupon_year_fraction must be numeric when provided, "
                    f"got {type(self.fixed_coupon_year_fraction)}"
                )
            if self.fixed_coupon_year_fraction <= 0:
                raise ValidationError(
                    "fixed_coupon_year_fraction must be positive, "
                    f"got {self.fixed_coupon_year_fraction}"
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
                    raise ValidationError(f"{name}[{i}] must be positive number, got {b}")
        else:
            if not isinstance(barrier, (int, float)) or barrier <= 0:
                raise ValidationError(f"{name} must be positive number, got {barrier}")
