"""Cash-leg valuator and trade attribution structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from cashleg.base import CashLeg, LegDirection
from cashleg.event_distribution import EventDistribution


def value_leg(
    leg: CashLeg, event_dist: EventDistribution, env, position_notional: float
) -> float:
    """Compute signed PV of a cash leg from the buyer's perspective."""

    return leg.value(event_dist, env, position_notional)


@dataclass(frozen=True)
class LegPV:
    """PV attribution for one cash leg."""

    name: Optional[str]
    direction: LegDirection
    pv: float


@dataclass(frozen=True)
class TradeValueBreakdown:
    """Product and cash-leg PV attribution."""

    product_npv: float
    leg_pvs: Dict[str, LegPV]

    @property
    def total(self) -> float:
        """Total trade PV."""

        return self.product_npv + sum(leg_pv.pv for leg_pv in self.leg_pvs.values())
