"""
OTC autocallable backtesting tools.
"""

from .book_engine import (
    BookAutocallableBacktestConfig,
    BookAutocallableBacktestEngine,
    BookBacktestResults,
    BookProduct,
    HedgeSpec,
)
from .config import (
    AutocallableBacktestConfig,
    AutocallableEngineConfig,
    FuturesRollPolicy,
    SurfaceGridConfig,
    VolModelCalibrationConfig,
)
from .dashboard import AutocallableBacktestDashboard, AutocallableDashboardConfig
from .engine import AutocallableBacktestEngine
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
from .results import AutocallableBacktestResults
from .state import (
    AutocallableDeltaHedgeStrategy,
    AutocallableLifecycleState,
    FuturesHedgePosition,
)
from quantark.volmodels.calibration import CalibratedVolModel, VolModelCalibrator
from quantark.param.vol.surface_history import IvSurfaceArtifact, VolSurfaceHistory

__all__ = [
    "AKShareAutocallableDataAdapter",
    "AutocallableBacktestConfig",
    "BookAutocallableBacktestConfig",
    "BookAutocallableBacktestEngine",
    "BookBacktestResults",
    "BookProduct",
    "HedgeSpec",
    "AutocallableBacktestDashboard",
    "AutocallableDashboardConfig",
    "AutocallableBacktestEngine",
    "AutocallableBacktestResults",
    "AutocallableDeltaHedgeStrategy",
    "AutocallableEngineConfig",
    "AutocallableLifecycleState",
    "AutocallableMarketDataSet",
    "CalibratedVolModel",
    "FuturesHedgePosition",
    "FuturesRollPolicy",
    "ImpliedBasisYield",
    "IvSurfaceArtifact",
    "SignedDividendYield",
    "SurfaceGridConfig",
    "VolModelCalibrationConfig",
    "VolModelCalibrator",
    "VolSurfaceHistory",
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
