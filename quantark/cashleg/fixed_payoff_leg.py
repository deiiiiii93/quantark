"""Event-conditional fixed payoff legs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from quantark.cashleg.base import CashLeg
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.util.exceptions import ValidationError


class PaymentTrigger(Enum):
    """Event that triggers a fixed payoff."""

    AT_KO = "at_ko"
    AT_KI = "at_ki"
    AT_MATURITY_NO_KO = "at_maturity_no_ko"
    AT_MATURITY_WITH_KI = "at_maturity_with_ki"
    AT_MATURITY_ANY = "at_maturity_any"


_TRIGGER_TO_EVENT = {
    PaymentTrigger.AT_KO: EventType.KO,
    PaymentTrigger.AT_KI: EventType.KI,
    PaymentTrigger.AT_MATURITY_NO_KO: EventType.MATURITY_NO_KO,
    PaymentTrigger.AT_MATURITY_WITH_KI: EventType.MATURITY_WITH_KI,
}


@dataclass(frozen=True)
class FixedPayoffLeg(CashLeg):
    """Fixed amount paid when a specified event occurs."""

    amount: float = 0.0
    trigger: PaymentTrigger = PaymentTrigger.AT_MATURITY_NO_KO
    payment_offset_days: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.amount)) or self.amount < 0.0:
            raise ValidationError(
                f"FixedPayoffLeg.amount must be non-negative and finite, got {self.amount}"
            )
        if self.payment_offset_days < 0:
            raise ValidationError(
                "FixedPayoffLeg.payment_offset_days must be >= 0, "
                f"got {self.payment_offset_days}"
            )
        if not isinstance(self.trigger, PaymentTrigger):
            raise ValidationError(f"Invalid PaymentTrigger: {self.trigger}")

    def required_event_types(self) -> frozenset:
        """The single trigger stream (or both maturity buckets for ANY) [§11.1]."""

        if self.trigger is PaymentTrigger.AT_MATURITY_ANY:
            return frozenset(
                {EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI}
            )
        return frozenset({_TRIGGER_TO_EVENT[self.trigger]})

    def value(
        self, event_dist: EventDistribution, env, position_notional: float
    ) -> float:
        """Return signed PV of this event-triggered amount."""

        offset_yf = float(self.payment_offset_days) / 365.0
        if self.trigger is PaymentTrigger.AT_MATURITY_ANY:
            probability = float(
                event_dist.probabilities.get(EventType.MATURITY_NO_KO, 0.0)
            ) + float(event_dist.probabilities.get(EventType.MATURITY_WITH_KI, 0.0))
            pay_time = float(event_dist.event_times[-1]) + offset_yf
            return (
                self.sign()
                * float(self.amount)
                * probability
                * env.get_discount_factor(pay_time)
            )

        event_type = _TRIGGER_TO_EVENT[self.trigger]
        if event_type not in event_dist.probabilities:
            raise ValidationError(
                f"FixedPayoffLeg trigger {self.trigger.value} requires "
                f"{event_type.value} in EventDistribution.probabilities"
            )

        probability = event_dist.probabilities[event_type]
        if isinstance(probability, np.ndarray):
            pay_times = event_dist.event_times + offset_yf
            dfs = np.array([env.get_discount_factor(float(t)) for t in pay_times])
            return self.sign() * float(self.amount) * float(np.sum(probability * dfs))

        pay_time = float(event_dist.event_times[-1]) + offset_yf
        return (
            self.sign()
            * float(self.amount)
            * float(probability)
            * env.get_discount_factor(pay_time)
        )
