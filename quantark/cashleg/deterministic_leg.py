"""Deterministic fixed cash flow legs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional

from quantark.cashleg.base import CashLeg, roll_payment_time
from quantark.util.exceptions import ValidationError


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

    def time_shift(self, dt: float, *, keep_landing: bool = False) -> "Optional[DeterministicLeg]":
        """The leg seen ``dt`` years later (spec 2026-10-23 D10): the payment moves by ``-dt``; ``None`` once it
        is no longer ahead (a landing payment kept at 0 only with ``keep_landing``). ``dt == 0`` drops a payment
        due at 0 (paid on that day) and otherwise returns the leg itself."""
        if float(dt) < 0.0:
            return self
        payment_time = roll_payment_time(self.payment_time, dt, keep_landing=keep_landing)
        if payment_time is None:
            return None
        if payment_time == self.payment_time:
            return self
        return replace(self, payment_time=payment_time)

    def requires_event_distribution(self) -> bool:
        """A deterministic leg can be valued without event timing."""

        return False

    def required_event_types(self) -> frozenset:
        """A deterministic cash flow reads no event streams [§11.1]."""

        return frozenset()
