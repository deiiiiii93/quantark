"""Deterministic fixed cash flow legs."""

from __future__ import annotations

from dataclasses import dataclass
import math

from cashleg.base import CashLeg
from util.exceptions import ValidationError


@dataclass(frozen=True)
class DeterministicLeg(CashLeg):
    """Fixed amount paid at a fixed time."""

    amount: float = 0.0
    payment_time: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.amount)) or self.amount < 0.0:
            raise ValidationError(
                f"DeterministicLeg.amount must be non-negative and finite, got {self.amount}"
            )
        if not math.isfinite(float(self.payment_time)) or self.payment_time < 0.0:
            raise ValidationError(
                f"DeterministicLeg.payment_time must be non-negative and finite, got {self.payment_time}"
            )

    def value(self, event_dist, env, position_notional: float) -> float:
        """Return signed discounted PV of the fixed cash flow."""

        return self.sign() * float(self.amount) * env.get_discount_factor(
            float(self.payment_time)
        )

    def requires_event_distribution(self) -> bool:
        """A deterministic leg can be valued without event timing."""

        return False
