"""
Deprecated shim package — OTC autocallable backtesting moved to
``quantark.backtest.replay`` (0.5.0 removes this package).

Relocated infrastructure:
- vol calibrators -> ``quantark.volmodels.calibration``
- IV-surface history -> ``quantark.param.vol.surface_history``
- futures ledger -> ``quantark.backtest.futures_ledger``
"""
import warnings

from quantark.backtest.replay import (
    AKShareAutocallableDataAdapter,
    AutocallableBacktestConfig,
    AutocallableBacktestDashboard,
    AutocallableBacktestEngine,
    AutocallableBacktestResults,
    AutocallableDashboardConfig,
    AutocallableDeltaHedgeStrategy,
    AutocallableEngineConfig,
    AutocallableLifecycleState,
    AutocallableMarketDataSet,
    BookAutocallableBacktestConfig,
    BookAutocallableBacktestEngine,
    BookBacktestResults,
    BookProduct,
    FuturesHedgePosition,
    FuturesRollPolicy,
    HedgeSpec,
    ImpliedBasisYield,
    SignedDividendYield,
    SurfaceGridConfig,
    VolModelCalibrationConfig,
    calculate_basis_yield,
    create_autocallable_engine,
    create_event_stats_engine,
    create_mc_event_stats_engine,
    create_pricing_engine,
    create_surface_engine,
    create_vol_model_engine,
    derive_implied_dividend_yield,
    normalize_futures_chain,
    normalize_time_series,
)
from quantark.param.vol.surface_history import IvSurfaceArtifact, VolSurfaceHistory
from quantark.volmodels.calibration import CalibratedVolModel, VolModelCalibrator

warnings.warn(
    "quantark.backtest.otc moved to quantark.backtest.replay; "
    "this alias package is removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)

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
