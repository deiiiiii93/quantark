"""
OTC autocallable backtesting tools.
"""

from .config import (
    AutocallableBacktestConfig,
    AutocallableEngineConfig,
    FuturesRollPolicy,
    SurfaceGridConfig,
)
from .dashboard import AutocallableBacktestDashboard, AutocallableDashboardConfig
from .engine import AutocallableBacktestEngine
from .engine_factory import (
    create_autocallable_engine,
    create_event_stats_engine,
    create_mc_event_stats_engine,
    create_pricing_engine,
    create_surface_engine,
)
from .market import (
    AKShareAutocallableDataAdapter,
    AutocallableMarketDataSet,
    ImpliedBasisYield,
    SignedDividendYield,
    calculate_basis_yield,
    derive_implied_dividend_yield,
    normalize_futures_chain,
    normalize_time_series,
)
from .results import AutocallableBacktestResults
from .state import (
    AutocallableDeltaHedgeStrategy,
    AutocallableLifecycleState,
    FuturesHedgePosition,
)

__all__ = [
    "AKShareAutocallableDataAdapter",
    "AutocallableBacktestConfig",
    "AutocallableBacktestDashboard",
    "AutocallableDashboardConfig",
    "AutocallableBacktestEngine",
    "AutocallableBacktestResults",
    "AutocallableDeltaHedgeStrategy",
    "AutocallableEngineConfig",
    "AutocallableLifecycleState",
    "AutocallableMarketDataSet",
    "FuturesHedgePosition",
    "FuturesRollPolicy",
    "ImpliedBasisYield",
    "SignedDividendYield",
    "SurfaceGridConfig",
    "calculate_basis_yield",
    "create_autocallable_engine",
    "create_event_stats_engine",
    "create_mc_event_stats_engine",
    "create_pricing_engine",
    "create_surface_engine",
    "derive_implied_dividend_yield",
    "normalize_futures_chain",
    "normalize_time_series",
]
