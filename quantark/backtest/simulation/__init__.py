"""Simulated-path backtest for autocallable books (spec 2026-09-08).

Plan 1 ships the data layer: ``MarketPath`` batches, carry-curve to
listed-chain conversion, the shared dividend rule, the history builder and
the three path generators.
"""
from __future__ import annotations

from .paths.market_path import DEFAULT_TENOR_GRID, MarketPath, StartState, trading_calendar

__all__ = ["DEFAULT_TENOR_GRID", "MarketPath", "StartState", "trading_calendar"]
