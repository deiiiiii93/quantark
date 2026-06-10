"""Base amount for notional- and margin-linked cash legs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from util.exceptions import ValidationError


class BaseAmountMode(Enum):
    """How a cash-leg base amount is resolved at valuation time."""

    ABSOLUTE = "absolute"
    NOTIONAL_FRACTION = "notional_fraction"
    MARGIN_FRACTION = "margin_fraction"


@dataclass(frozen=True)
class BaseAmount:
    """Base amount for accrual computations."""

    value: float
    mode: BaseAmountMode
    margin_rate: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.value)):
            raise ValidationError(f"BaseAmount.value must be finite, got {self.value}")
        if not isinstance(self.mode, BaseAmountMode):
            raise ValidationError(f"Invalid BaseAmountMode: {self.mode}")

        if self.mode is BaseAmountMode.ABSOLUTE:
            if self.value < 0.0:
                raise ValidationError(
                    "BaseAmount.value must be non-negative for ABSOLUTE mode"
                )
        elif not (0.0 <= self.value <= 1.0):
            raise ValidationError(
                f"BaseAmount.value must be in [0, 1] for {self.mode.value} mode, "
                f"got {self.value}"
            )

        if self.mode is BaseAmountMode.MARGIN_FRACTION:
            if not math.isfinite(float(self.margin_rate)) or self.margin_rate <= 0.0:
                raise ValidationError(
                    f"margin_rate must be > 0 for MARGIN_FRACTION mode, got {self.margin_rate}"
                )

    def resolve(self, position_notional: float) -> float:
        """Resolve this base amount against a position notional."""

        if self.mode is BaseAmountMode.ABSOLUTE:
            return float(self.value)
        if self.mode is BaseAmountMode.NOTIONAL_FRACTION:
            return float(self.value) * float(position_notional)
        if self.mode is BaseAmountMode.MARGIN_FRACTION:
            return float(self.value) * float(self.margin_rate) * float(position_notional)
        raise ValidationError(f"Unknown BaseAmountMode: {self.mode}")
