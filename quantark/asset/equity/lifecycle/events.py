"""
Lifecycle event primitives shared by historical replay (quantark.backtest.otc)
and dynamic scenario simulation (quantark.dynamicscenario).

A ``LifecycleEvent`` is an immutable record of one realized contract event
(knock-out, knock-in, coupon, maturity, expiry) produced by a lifecycle
tracker. Trackers return events; consumers decide how to record or act on
them (append to a backtest action log, remove a contingent position, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cashflows import RealizedCashflow


class LifecycleEventType(Enum):
    """Types of realized lifecycle events."""

    KNOCK_OUT = "KO"
    KNOCK_IN = "KI"
    COUPON = "COUPON"
    MATURITY = "MATURITY"
    EXPIRY = "EXPIRY"


@dataclass(frozen=True)
class LifecycleEvent:
    """
    One realized lifecycle event.

    Attributes:
        event_type: Kind of event.
        date: Observation/settlement date (datetime or pandas Timestamp).
        spot: Underlying close used for the observation.
        observation_index: Index in the product's observation schedule, if any.
        barrier: Barrier level checked, if any.
        payoff: Per-position-unit amount fixed by the event (0 for state-only
            events).
        cashflow: Quantity-scaled contractual amount fixed by the event.
        realized_cashflow: Immutable ledger entry registered when the event
            determines a cashflow. State-only events leave this as ``None``.
        terminates_position: True if the position ceases to exist after this
            event (KO, one-touch hit, maturity, expiry).
        state_before: Lifecycle-state snapshot before the event.
        state_after: Lifecycle-state snapshot after the event.
        metadata: Extra recorder fields (e.g. {"monitoring": "daily_close"}).
    """

    event_type: LifecycleEventType
    date: datetime
    spot: float
    observation_index: Optional[int] = None
    barrier: Optional[float] = None
    payoff: float = 0.0
    cashflow: float = 0.0
    realized_cashflow: Optional["RealizedCashflow"] = field(
        default=None,
        hash=False,
        compare=False,
    )
    terminates_position: bool = False
    state_before: Dict[str, bool] = field(default_factory=dict, hash=False, compare=False)
    state_after: Dict[str, bool] = field(default_factory=dict, hash=False, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
