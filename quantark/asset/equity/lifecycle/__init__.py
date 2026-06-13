"""
Shared lifecycle core for equity structured products.

Consumed by ``quantark.backtest.otc`` (single-product historical replay),
``quantark.backtest.equity`` (portfolio backtests) and
``quantark.dynamicscenario`` (hypothetical path simulation). The
portfolio-driving ``PortfolioLifecycleManager`` is shared by the latter two.
"""

from .autocallable import AutocallableLifecycleTracker
from .barrier import TRACKED_BARRIER_PRODUCTS, BarrierLifecycleTracker
from .events import LifecycleEvent, LifecycleEventType
from .manager import PortfolioLifecycleManager, ProcessedLifecycleEvent
from .state import AutocallableLifecycleState, BarrierLifecycleState

__all__ = [
    "AutocallableLifecycleTracker",
    "BarrierLifecycleTracker",
    "TRACKED_BARRIER_PRODUCTS",
    "LifecycleEvent",
    "LifecycleEventType",
    "PortfolioLifecycleManager",
    "ProcessedLifecycleEvent",
    "AutocallableLifecycleState",
    "BarrierLifecycleState",
]
