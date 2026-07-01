"""Cash-leg valuator and trade attribution structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from quantark.cashleg.base import CashLeg, LegDirection
from quantark.cashleg.event_distribution import EventDistribution


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


@dataclass(frozen=True)
class TradeGreeks:
    """Product-only and full-trade (product + cash legs) greeks.

    ``product`` holds ``quantity * d(npv)/dx``; ``total`` holds
    ``quantity * d(npv)/dx + d(Σ leg_pv)/dx`` — the legs carry their own
    direction and are NOT scaled by the note quantity sign [§11.8].
    """

    product: Dict[str, float]
    total: Dict[str, float]
    leg_pvs: Dict[str, LegPV]
