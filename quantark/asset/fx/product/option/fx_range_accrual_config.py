"""
Configuration classes for FX range accrual products.

An FX range accrual pays a coupon proportional to the fraction of fixings on
which the FX spot ``X_t`` stays within a ``[lower, upper]`` band. Barriers are
absolute rate levels (e.g. 1.05 and 1.15 for EUR/USD), consistent with the way
FX ranges are quoted in the market. The coupon is a fixed cash amount in the
domestic (quote) currency, so the contingent payoff is a *cash-or-nothing*
range digital (see :class:`FxRangeAccrualOption`).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union, TYPE_CHECKING

from quantark.util.calendar import calculate_year_fraction
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType
from quantark.util.exceptions import ValidationError

if TYPE_CHECKING:
    from quantark.priceenv import FxPricingEnvironment


@dataclass
class FxRangeAccrualObservationRecord:
    """
    A single fixing of an FX range accrual.

    For past fixings set ``observed_in_range`` to the realised outcome; for
    future fixings leave it ``None``. Per-fixing barriers override the config
    barriers when supplied (used for stepped/ratchet ranges).

    Attributes:
        observation_time: Fixing time as a year fraction from valuation
            (mutually exclusive with observation_date).
        observation_date: Fixing calendar date (resolved against the env).
        upper_barrier: Per-fixing upper barrier (optional).
        lower_barrier: Per-fixing lower barrier (optional).
        weight: Fixing weight (e.g. 3.0 for a Friday carrying the weekend in a
            calendar-day accrual). Defaults to 1.0.
        observed_in_range: Realised outcome for past fixings; None for future.
    """

    observation_time: Optional[float] = None
    observation_date: Optional[datetime] = None
    upper_barrier: Optional[float] = None
    lower_barrier: Optional[float] = None
    weight: float = 1.0
    observed_in_range: Optional[bool] = None

    def resolve_time(self, fx_env: Optional["FxPricingEnvironment"]) -> float:
        """
        Year fraction from the valuation date to this fixing.

        Returns a negative value for fixings strictly in the past. Date-based
        records require an ``FxPricingEnvironment`` to resolve.
        """
        if self.observation_time is not None:
            return self.observation_time
        if self.observation_date is not None:
            if fx_env is None:
                raise ValidationError(
                    "FxPricingEnvironment is required to resolve observation_date."
                )
            if self.observation_date == fx_env.valuation_date:
                return 0.0
            if self.observation_date > fx_env.valuation_date:
                return calculate_year_fraction(
                    fx_env.valuation_date,
                    self.observation_date,
                    fx_env.day_count_convention,
                    fx_env.bus_days_in_year,
                    calendar=fx_env.calendar,
                )
            return -calculate_year_fraction(
                self.observation_date,
                fx_env.valuation_date,
                fx_env.day_count_convention,
                fx_env.bus_days_in_year,
                calendar=fx_env.calendar,
            )
        raise ValidationError(
            "FxRangeAccrualObservationRecord requires observation_time or "
            "observation_date."
        )

    def is_observed(self) -> bool:
        """True when this fixing carries a realised outcome."""
        return self.observed_in_range is not None

    def validate(self) -> None:
        """Validate a single observation record."""
        if self.observation_time is None and self.observation_date is None:
            raise ValidationError(
                "FxRangeAccrualObservationRecord must provide observation_time "
                "or observation_date."
            )
        if self.weight <= 0:
            raise ValidationError(f"weight must be positive, got {self.weight}")
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
                f"lower_barrier ({self.lower_barrier}) must be below "
                f"upper_barrier ({self.upper_barrier})"
            )


@dataclass(frozen=True)
class FxRangeAccrualConfig:
    """
    Settings shared across an FX range accrual's fixings.

    Attributes:
        upper_barrier: Upper rate barrier; scalar or per-fixing list.
        lower_barrier: Lower rate barrier; scalar or per-fixing list.
        accrual_rate: Coupon rate applied to the notional when 100% in range
            (e.g. 0.04 for a 4% range-accrual coupon).
        is_rate_annualized: When True the coupon is scaled by the accrual-period
            year fraction; when False ``accrual_rate`` is the full-period coupon.
        day_count_convention: Day count used for the year fraction.
        accrual_pay_type: EXPIRY (single coupon discounted from maturity) or
            INSTANT (each fixing's contribution discounted from its own date).
        is_reverse: When True the coupon accrues OUTSIDE the band instead of
            inside.
    """

    upper_barrier: Union[float, List[float]]
    lower_barrier: Union[float, List[float]]
    accrual_rate: float = 0.04
    is_rate_annualized: bool = False
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365
    accrual_pay_type: CouponPayType = CouponPayType.EXPIRY
    is_reverse: bool = False

    def __post_init__(self):
        self._validate_barrier_positive(self.upper_barrier, "upper_barrier")
        self._validate_barrier_positive(self.lower_barrier, "lower_barrier")
        self._validate_barrier_ordering()

        if not isinstance(self.accrual_rate, (int, float)):
            raise ValidationError(
                f"accrual_rate must be a number, got {type(self.accrual_rate)}"
            )
        if self.accrual_rate < 0:
            raise ValidationError(
                f"accrual_rate must be non-negative, got {self.accrual_rate}"
            )
        if not isinstance(self.accrual_pay_type, CouponPayType):
            raise ValidationError(
                f"accrual_pay_type must be CouponPayType, got "
                f"{type(self.accrual_pay_type)}"
            )
        if not isinstance(self.day_count_convention, DayCountConvention):
            raise ValidationError(
                f"day_count_convention must be DayCountConvention, got "
                f"{type(self.day_count_convention)}"
            )
        if not isinstance(self.is_rate_annualized, bool):
            raise ValidationError(
                f"is_rate_annualized must be boolean, got "
                f"{type(self.is_rate_annualized)}"
            )
        if not isinstance(self.is_reverse, bool):
            raise ValidationError(
                f"is_reverse must be boolean, got {type(self.is_reverse)}"
            )

    @staticmethod
    def _validate_barrier_positive(
        barrier: Union[float, List[float]], name: str
    ) -> None:
        if isinstance(barrier, list):
            if not barrier:
                raise ValidationError(f"{name} list cannot be empty")
            for i, b in enumerate(barrier):
                if not isinstance(b, (int, float)) or b <= 0:
                    raise ValidationError(
                        f"{name}[{i}] must be a positive number, got {b}"
                    )
        else:
            if not isinstance(barrier, (int, float)) or barrier <= 0:
                raise ValidationError(
                    f"{name} must be a positive number, got {barrier}"
                )

    def _validate_barrier_ordering(self) -> None:
        upper_is_list = isinstance(self.upper_barrier, list)
        lower_is_list = isinstance(self.lower_barrier, list)

        if upper_is_list and lower_is_list:
            upper_list = list(self.upper_barrier)  # type: ignore[arg-type]
            lower_list = list(self.lower_barrier)  # type: ignore[arg-type]
            if len(upper_list) != len(lower_list):
                raise ValidationError(
                    f"upper_barrier and lower_barrier lists must have the same "
                    f"length, got {len(upper_list)} and {len(lower_list)}"
                )
            for i, (u, l) in enumerate(zip(upper_list, lower_list)):
                if l >= u:
                    raise ValidationError(
                        f"lower_barrier[{i}] ({l}) must be below "
                        f"upper_barrier[{i}] ({u})"
                    )
        elif upper_is_list:
            lower_scalar = float(self.lower_barrier)  # type: ignore[arg-type]
            for i, u in enumerate(self.upper_barrier):  # type: ignore[arg-type]
                if lower_scalar >= u:
                    raise ValidationError(
                        f"lower_barrier ({lower_scalar}) must be below "
                        f"upper_barrier[{i}] ({u})"
                    )
        elif lower_is_list:
            upper_scalar = float(self.upper_barrier)  # type: ignore[arg-type]
            for i, l in enumerate(self.lower_barrier):  # type: ignore[arg-type]
                if l >= upper_scalar:
                    raise ValidationError(
                        f"lower_barrier[{i}] ({l}) must be below "
                        f"upper_barrier ({upper_scalar})"
                    )
        else:
            if self.lower_barrier >= self.upper_barrier:  # type: ignore[operator]
                raise ValidationError(
                    f"lower_barrier ({self.lower_barrier}) must be below "
                    f"upper_barrier ({self.upper_barrier})"
                )

    def get_upper_barrier(self, obs_idx: int) -> float:
        """Upper barrier for a fixing index (scalar config returns the scalar)."""
        if isinstance(self.upper_barrier, list):
            self._check_index(obs_idx, len(self.upper_barrier), "upper_barrier")
            return self.upper_barrier[obs_idx]
        return self.upper_barrier

    def get_lower_barrier(self, obs_idx: int) -> float:
        """Lower barrier for a fixing index (scalar config returns the scalar)."""
        if isinstance(self.lower_barrier, list):
            self._check_index(obs_idx, len(self.lower_barrier), "lower_barrier")
            return self.lower_barrier[obs_idx]
        return self.lower_barrier

    @staticmethod
    def _check_index(obs_idx: int, length: int, name: str) -> None:
        if obs_idx < 0 or obs_idx >= length:
            raise ValidationError(
                f"obs_idx {obs_idx} out of range for {name} with length {length}"
            )
