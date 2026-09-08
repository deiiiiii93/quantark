"""Shared synthetic fixtures for the simulation tests."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.paths.market_path import (
    DEFAULT_TENOR_GRID,
    MarketPath,
    StartState,
    trading_calendar,
)

SPOT = 6000.0
VOL = 0.22
RATE = 0.02


def flat_carry(tenor_grid: np.ndarray, annual_carry: float = -0.10) -> np.ndarray:
    """B(T) = annual_carry * T: a flat 10% discount curve."""
    return annual_carry * np.asarray(tenor_grid, dtype=float)


@pytest.fixture()
def start_state() -> StartState:
    return StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))


@pytest.fixture()
def calendar() -> pd.DatetimeIndex:
    return trading_calendar(date(2024, 1, 2), 30)


def make_market_path(n_paths: int = 3, n_days: int = 30, start: date = date(2024, 1, 2)) -> MarketPath:
    """Deterministic flat-vol, drifting-spot batch for structural tests."""
    dates = trading_calendar(start, n_days)
    days = np.arange(n_days, dtype=float)
    spot = SPOT * (1.0 + 0.001 * days)[None, :] * (1.0 + 0.01 * np.arange(n_paths))[:, None]
    atm_vol = np.full((n_paths, n_days), VOL)
    rate = np.full((n_paths, n_days), RATE)
    carry = np.broadcast_to(flat_carry(DEFAULT_TENOR_GRID), (n_paths, n_days, DEFAULT_TENOR_GRID.size)).copy()
    return MarketPath(dates=dates, spot=spot, atm_vol=atm_vol, rate=rate, carry=carry,
                      tenor_grid=DEFAULT_TENOR_GRID.copy(), meta={"generator": "fixture"})
