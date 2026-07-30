"""
Product-replay backtesting: multi-product daily replay with futures hedging,
autocallable lifecycle, and per-day vol-model calibration.

Canonical names carry the ``Replay*`` prefix; the ``Book*``/``Autocallable*``
names remain as compatible aliases (single-product API in ``single.py``).
"""

from .config import (
    AutocallableEngineConfig,
    ReplayBacktestConfig,
    ReplayProduct,
    BookAutocallableBacktestConfig,
    BookProduct,
    HedgeSpec,
    SurfaceGridConfig,
    VolModelCalibrationConfig,
)
from .dashboard import AutocallableBacktestDashboard, AutocallableDashboardConfig
from .engine import BookAutocallableBacktestEngine, ReplayBacktestEngine
from .engine_factory import (
    create_autocallable_engine,
    create_event_stats_engine,
    create_mc_event_stats_engine,
    create_pricing_engine,
    create_surface_engine,
    create_vol_model_engine,
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
from .config import AutocallableBacktestConfig
from .results import (
    AutocallableBacktestResults,
    BookBacktestResults,
    ReplayBacktestResults,
)
from .single import AutocallableBacktestEngine
from .strategy_state import (
    AutocallableDeltaHedgeStrategy,
    AutocallableLifecycleState,
    FuturesHedgePosition,
)
from quantark.backtest.futures_ledger import FuturesRollPolicy


__all__ = [
    "AKShareAutocallableDataAdapter",
    "AutocallableBacktestConfig",
    "AutocallableBacktestDashboard",
    "AutocallableBacktestEngine",
    "AutocallableBacktestResults",
    "AutocallableDashboardConfig",
    "AutocallableDeltaHedgeStrategy",
    "AutocallableEngineConfig",
    "AutocallableLifecycleState",
    "AutocallableMarketDataSet",
    "BookAutocallableBacktestConfig",
    "BookAutocallableBacktestEngine",
    "BookBacktestResults",
    "BookProduct",
    "FuturesHedgePosition",
    "FuturesRollPolicy",
    "HedgeSpec",
    "ImpliedBasisYield",
    "ReplayBacktestConfig",
    "ReplayBacktestEngine",
    "ReplayBacktestResults",
    "ReplayProduct",
    "SignedDividendYield",
    "SurfaceGridConfig",
    "VolModelCalibrationConfig",
    "calculate_basis_yield",
    "create_autocallable_engine",
    "create_event_stats_engine",
    "create_mc_event_stats_engine",
    "create_pricing_engine",
    "create_surface_engine",
    "create_vol_model_engine",
    "derive_implied_dividend_yield",
    "normalize_futures_chain",
    "normalize_time_series",
]
