"""
Shared lifecycle core for equity structured products.

Consumed by ``quantark.backtest.otc`` (historical replay) and
``quantark.dynamicscenario`` (hypothetical path simulation).
"""

from .events import LifecycleEvent, LifecycleEventType
from .state import AutocallableLifecycleState, BarrierLifecycleState

__all__ = [
    "LifecycleEvent",
    "LifecycleEventType",
    "AutocallableLifecycleState",
    "BarrierLifecycleState",
]
