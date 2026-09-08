"""Simulated-path backtest for autocallable books (spec 2026-09-08).

Plan 1: the data layer -- ``MarketPath`` batches, carry-curve to
listed-chain conversion, the shared dividend rule, the history builder and
the three path generators.  Plan 2: the ensemble engine -- exact repricing
with a state cache, the vectorised lifecycle and futures ledger, the daily
loop and the conformance oracle against the replay engine.  Plan 3: the
approximate providers -- the spot ladder, the PDE life surface -- the
on-disk cache tier, the sampling accuracy gate and path batching over a
spawn pool.  See README.md in this package.
"""
from __future__ import annotations

from .carry import DayChain, carry_at, curve_from_chain, day_chain, listed_im_contracts
from .config import CacheConfig, EnsembleConfig, GateConfig, PricingProviderConfig
from .conformance import OracleReport, run_oracle
from .dataset import to_market_dataset
from .dividends import dividend_yield_for_day
from .engine import EnsembleBacktestEngine, EnsembleResults, StateCube
from .hedge import VectorHedgeLedger
from .lifecycle import VectorLifecycle
from .paths import (
    DEFAULT_TENOR_GRID, ConstantVol, GBMPaths, MarketPath, PathHistory, SnowballStressLibrary,
    StartState, StationaryBlockBootstrap, StickyRealisedVol, market_path_from_day_path, stress_set,
    trading_calendar,
)
from .pricing import DayStates, GateReport, StateKey
from .pricing.base import GateFailure, GateScale, bucket_centre, bucket_key
from .pricing.cache import DiskTier, StateCache
from .pricing.repricing import RepricingPricer
from .pricing.surface import LifeSurfacePricer, SurfaceCache
from .runner import batch_ranges, concat_results, run_ensemble

__all__ = [
    "DEFAULT_TENOR_GRID", "MarketPath", "StartState", "trading_calendar", "PathHistory",
    "StationaryBlockBootstrap", "GBMPaths", "ConstantVol", "StickyRealisedVol",
    "market_path_from_day_path", "SnowballStressLibrary", "stress_set", "DayChain", "day_chain",
    "carry_at", "curve_from_chain", "listed_im_contracts", "dividend_yield_for_day", "to_market_dataset",
    "EnsembleConfig", "PricingProviderConfig", "CacheConfig", "GateConfig",
    "EnsembleBacktestEngine", "EnsembleResults", "StateCube", "RepricingPricer",
    "StateCache", "DayStates", "StateKey", "GateReport", "VectorLifecycle",
    "VectorHedgeLedger", "OracleReport", "run_oracle",
    "LifeSurfacePricer", "SurfaceCache", "GateScale", "GateFailure", "DiskTier",
    "bucket_key", "bucket_centre", "run_ensemble", "concat_results", "batch_ranges",
]
