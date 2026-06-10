"""CashLeg ABC and shared sign convention."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cashleg.event_distribution import EventDistribution
    from priceenv import PricingEnvironment


class LegDirection(Enum):
    """Cash-flow direction from the buyer's perspective."""

    BUYER_RECEIVES = +1
    BUYER_PAYS = -1


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

    def sign(self) -> int:
        """Return +1 for buyer receives, -1 for buyer pays."""

        return int(self.direction.value)
