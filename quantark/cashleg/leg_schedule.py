"""Cash-leg accrual schedule."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical.constants import Tolerance


@dataclass(frozen=True)
class LegSchedule:
    """Independent accrual period and payment schedule for cash legs."""

    period_starts: np.ndarray
    period_ends: np.ndarray
    payment_times: np.ndarray
    period_start_dates: Optional[List[datetime]] = None
    period_end_dates: Optional[List[datetime]] = None
    payment_dates: Optional[List[datetime]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "period_starts", np.asarray(self.period_starts, dtype=float)
        )
        object.__setattr__(self, "period_ends", np.asarray(self.period_ends, dtype=float))
        object.__setattr__(
            self, "payment_times", np.asarray(self.payment_times, dtype=float)
        )

        n = len(self.period_starts)
        if n == 0:
            raise ValidationError("LegSchedule must contain at least one period")
        if len(self.period_ends) != n or len(self.payment_times) != n:
            raise ValidationError(
                "LegSchedule arrays must have matching length; got "
                f"period_starts={n}, period_ends={len(self.period_ends)}, "
                f"payment_times={len(self.payment_times)}"
            )
        if (
            self.period_starts.ndim != 1
            or self.period_ends.ndim != 1
            or self.payment_times.ndim != 1
        ):
            raise ValidationError("LegSchedule arrays must be one-dimensional")
        if (
            np.any(~np.isfinite(self.period_starts))
            or np.any(~np.isfinite(self.period_ends))
            or np.any(~np.isfinite(self.payment_times))
        ):
            raise ValidationError("LegSchedule arrays must contain finite values")
        if np.any(self.period_ends < self.period_starts - Tolerance.PROBABILITY):
            bad = np.where(self.period_ends < self.period_starts)[0]
            raise ValidationError(
                f"LegSchedule period_ends < period_starts at indices {bad.tolist()}"
            )
        if np.any(self.payment_times < 0.0):
            raise ValidationError("LegSchedule payment_times must be non-negative")

        self._validate_optional_dates("period_start_dates", self.period_start_dates, n)
        self._validate_optional_dates("period_end_dates", self.period_end_dates, n)
        self._validate_optional_dates("payment_dates", self.payment_dates, n)

    def last_period_end(self) -> float:
        """Return the final accrual period end time."""

        return float(self.period_ends[-1])

    def validate_within_maturity(self, maturity: float) -> None:
        """Validate no period or payment extends beyond product maturity."""

        maturity = float(maturity)
        if self.last_period_end() > maturity + Tolerance.PROBABILITY:
            raise ValidationError(
                f"LegSchedule last period end {self.last_period_end()} "
                f"exceeds product maturity {maturity}"
            )
        max_payment = float(np.max(self.payment_times))
        if max_payment > maturity + Tolerance.PROBABILITY:
            raise ValidationError(
                f"LegSchedule last payment time {max_payment} exceeds product maturity {maturity}"
            )

    @staticmethod
    def _validate_optional_dates(
        field_name: str, values: Optional[List[datetime]], expected_len: int
    ) -> None:
        if values is not None and len(values) != expected_len:
            raise ValidationError(
                f"{field_name} must have length {expected_len}, got {len(values)}"
            )
