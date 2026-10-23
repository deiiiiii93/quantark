"""CashLeg ABC and shared sign convention."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from quantark.util.numerical import is_zero

if TYPE_CHECKING:
    from quantark.cashleg.event_distribution import EventDistribution, EventType
    from quantark.priceenv import PricingEnvironment


class LegDirection(Enum):
    """Cash-flow direction for the position holder.

    Legs are absolute [§11.8]: they are not scaled by the position quantity or its
    sign, so a seller books its legs with the seller's own direction. The BUYER_*
    names read as "the holder receives" / "the holder pays".
    """

    BUYER_RECEIVES = +1
    BUYER_PAYS = -1


def roll_payment_time(payment_time: float, dt: float, *, keep_landing: bool = False) -> Optional[float]:
    """A payment time seen from an origin ``dt`` later (spec 2026-10-23 portfolio-trade-value D10).

    ``payment_time - dt`` when that is still ahead; ``0.0`` when it lands on the new origin (``is_zero``) and
    ``keep_landing`` is set; ``None`` when the payment is no longer ahead (strictly past, or landing without
    ``keep_landing``). The leg rolls and the multi-day booking share this predicate: a flow is paid exactly when
    its roll without ``keep_landing`` returns ``None``.
    """
    remaining = float(payment_time) - float(dt)
    if is_zero(remaining):
        return 0.0 if keep_landing else None
    if remaining < 0.0:
        return None
    return remaining


@dataclass(frozen=True)
class CashLeg(ABC):
    """Abstract cash leg attached to an equity position."""

    direction: LegDirection
    name: Optional[str] = None
    leg_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @abstractmethod
    def value(
        self,
        event_dist: "EventDistribution",
        env: "PricingEnvironment",
        position_notional: float,
    ) -> float:
        """Return signed present value from the buyer's perspective."""

    def requires_event_distribution(self) -> bool:
        """Whether this leg needs non-trivial event timing to value accurately."""

        return True

    def required_event_types(self) -> "frozenset[EventType]":
        """Event-distribution streams this leg reads at valuation time [§11.1].

        The engine takes the union across all attached legs to decide which
        indicator columns to propagate, pruning the expensive KI split when no
        leg needs it. Default: no streams (a deterministic leg reads none).
        """

        return frozenset()

    def sign(self) -> int:
        """Return +1 for buyer receives, -1 for buyer pays."""

        return int(self.direction.value)
