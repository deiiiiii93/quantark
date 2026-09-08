"""Simulated-path backtest for autocallable books (spec 2026-09-08).

Plan 1: the data layer -- ``MarketPath`` batches, carry-curve to
listed-chain conversion, the shared dividend rule, the history builder and
the three path generators.  See README.md in this package.
"""
from __future__ import annotations

from .carry import DayChain, carry_at, curve_from_chain, day_chain, listed_im_contracts
from .dataset import to_market_dataset
from .dividends import dividend_yield_for_day
from .paths import (
    DEFAULT_TENOR_GRID, ConstantVol, GBMPaths, MarketPath, PathHistory, SnowballStressLibrary,
    StartState, StationaryBlockBootstrap, StickyRealisedVol, market_path_from_day_path, stress_set,
    trading_calendar,
)

__all__ = [
    "DEFAULT_TENOR_GRID", "MarketPath", "StartState", "trading_calendar", "PathHistory",
    "StationaryBlockBootstrap", "GBMPaths", "ConstantVol", "StickyRealisedVol",
    "market_path_from_day_path", "SnowballStressLibrary", "stress_set", "DayChain", "day_chain",
    "carry_at", "curve_from_chain", "listed_im_contracts", "dividend_yield_for_day", "to_market_dataset",
]
