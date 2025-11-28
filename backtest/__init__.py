"""
QuantArk Backtest Module

A comprehensive backtesting framework for hedging strategies with:
- Delta-neutral and custom strategy support (equity)
- DV01-neutral and convexity-neutral strategies (fixed income)
- Transaction cost modeling
- Comprehensive logging and metrics
- Static and interactive visualizations

Supports multiple asset classes:
- Equity: Delta/gamma hedging with spot/futures
- Fixed Income: DV01/convexity hedging with bond futures

Backward Compatibility:
- BacktestEngine, BacktestConfig, etc. are aliases to equity implementations
"""

# Base protocols
from backtest.base import (
    BaseTradeRecord,
    BaseHedgeExecutor,
    BaseBacktestEngine,
    BaseBacktestResults,
    BaseBacktestConfig,
)

# Equity implementations (with backward-compatible aliases)
from backtest.equity import (
    # Config
    BacktestConfig,
    EquityBacktestConfig,
    # State
    BacktestState,
    StateTracker,
    TradeRecord,
    # Results
    BacktestResults,
    EquityBacktestResults,
    # Metrics
    PerformanceMetrics,
    EquityPerformanceMetrics,
    # Engine
    BacktestEngine,
    EquityBacktestEngine,
    # Hedge Executor
    HedgeExecutor,
    EquityHedgeExecutor,
)

# Transaction costs (shared)
from backtest.transaction_costs import (
    TransactionCostModel,
    ZeroCostModel,
    FixedCostModel,
    ProportionalCostModel,
    CompleteCostModel,
)

# Logging (shared)
from backtest.logger import BacktestLogger

# Visualization (shared)
from backtest.visualizer import StaticVisualizer
from backtest.dashboard import InteractiveDashboard
from backtest.report_generator import ReportGenerator

__all__ = [
    # Base protocols
    "BaseTradeRecord",
    "BaseHedgeExecutor",
    "BaseBacktestEngine",
    "BaseBacktestResults",
    "BaseBacktestConfig",
    # Equity (explicit names)
    "EquityBacktestEngine",
    "EquityBacktestConfig",
    "EquityBacktestResults",
    "EquityPerformanceMetrics",
    "EquityHedgeExecutor",
    # Backward compatibility aliases
    "BacktestEngine",
    "BacktestConfig",
    "BacktestState",
    "StateTracker",
    "TradeRecord",
    "BacktestResults",
    "PerformanceMetrics",
    "HedgeExecutor",
    # Transaction costs
    "TransactionCostModel",
    "ZeroCostModel",
    "FixedCostModel",
    "ProportionalCostModel",
    "CompleteCostModel",
    # Shared utilities
    "BacktestLogger",
    "StaticVisualizer",
    "InteractiveDashboard",
    "ReportGenerator",
]


# Lazy imports for FI module (avoid circular imports)
def __getattr__(name):
    fi_names = (
        "FIBacktestEngine",
        "FIBacktestConfig",
        "FIBacktestResults",
        "FIPerformanceMetrics",
        "FIHedgeExecutor",
        "FIBacktestState",
    )
    if name in fi_names:
        from backtest import fi

        return getattr(fi, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
