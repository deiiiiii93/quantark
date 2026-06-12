"""
Shared lifecycle core for equity structured products.

Consumed by ``quantark.backtest.otc`` (historical replay) and
``quantark.dynamicscenario`` (hypothetical path simulation).
"""

from .autocallable import AutocallableLifecycleTracker
from .barrier import TRACKED_BARRIER_PRODUCTS, BarrierLifecycleTracker
from .events import LifecycleEvent, LifecycleEventType
from .state import AutocallableLifecycleState, BarrierLifecycleState

__all__ = [
    "AutocallableLifecycleTracker",
    "BarrierLifecycleTracker",
    "TRACKED_BARRIER_PRODUCTS",
    "LifecycleEvent",
    "LifecycleEventType",
    "AutocallableLifecycleState",
    "BarrierLifecycleState",
]
