"""
Fixed Income backtest module.

Provides FI-specific backtest components for simulating DV01/convexity-neutral
hedging strategies on Fixed Income portfolios.
"""
from .config import FIBacktestConfig
from .state import FIBacktestState, FIStateTracker, FITradeRecord
from .results import FIBacktestResults
from .metrics import FIPerformanceMetrics
from .engine import FIBacktestEngine
from .hedge_executor import FIHedgeExecutor

__all__ = [
    # Config
    'FIBacktestConfig',
    # State
    'FIBacktestState',
    'FIStateTracker',
    'FITradeRecord',
    # Results
    'FIBacktestResults',
    # Metrics
    'FIPerformanceMetrics',
    # Engine
    'FIBacktestEngine',
    # Hedge Executor
    'FIHedgeExecutor',
]

