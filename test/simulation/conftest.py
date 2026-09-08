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


# --- plan 2: a short snowball book and its ensemble configuration ----------

from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.asset.equity.product.option import create_standard_snowball  # noqa: E402
from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: E402
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct  # noqa: E402
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy  # noqa: E402
from quantark.backtest.transaction_costs import ZeroCostModel  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402
from quantark.util.enum.engine_enums import EngineType  # noqa: E402

KO_BARRIER = 1.03 * SPOT
KI_BARRIER = 0.75 * SPOT


def short_snowball(maturity_days: int = 6, *, ko_days=(2, 5), ki_days=(1, 3), continuous_ki: bool = False,
                   settlement_lag_days: int = 0):
    """A snowball short enough to run a whole life inside a test.

    ``settlement_lag_days`` puts a YEAR_FRACTION settlement lag on the
    contract (the numeric clock: the schedule is in year fractions, so the
    tracker keeps a time-based valuation point).  The factory does not take
    a convention, so it is set after construction and re-validated.
    """
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit

    product = create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=maturity_days / 365.0, contract_multiplier=1.0,
        ko_barrier=KO_BARRIER, ki_barrier=KI_BARRIER, ko_rate=0.20, num_observations=len(ko_days),
        ko_observation_dates=[d / 365.0 for d in ko_days],
        ki_observation_type=ObservationType.CONTINUOUS if continuous_ki else ObservationType.DISCRETE,
        ki_continuous=continuous_ki,
        ki_observation_dates=None if continuous_ki else [d / 365.0 for d in ki_days],
        include_principal=True,
    )
    if settlement_lag_days:
        product.settlement_convention = SettlementConvention(
            lag=settlement_lag_days / 365.0, lag_unit=SettlementLagUnit.YEAR_FRACTION,
        )
        product._validate_settlement_convention()
    return product


def pde_engine_config(**overrides) -> AutocallableEngineConfig:
    """PDE with stock parameters, like the replay goldens: the QUAD recursion
    cannot age the short fixture's KO grid day by day."""
    kwargs = dict(pricing_engine_type=EngineType.PDE, pde_params=PDEParams())
    kwargs.update(overrides)
    return AutocallableEngineConfig(**kwargs)


def ensemble_config(products=None, **overrides):
    """A one-snowball seller book with a zero-cost front-month futures hedge.

    The book is 1000 units short: one unit of a 6000-spot snowball carries a
    delta well under one 200-multiplier hand, so a one-unit book would round
    to zero contracts every day and never exercise the hedge.
    """
    from quantark.backtest.simulation.config import CacheConfig, EnsembleConfig, GateConfig, PricingProviderConfig

    if products is None:
        products = [ReplayProduct(product=short_snowball(), quantity=-1000.0, position_id=1, has_lifecycle=True)]
    kwargs = dict(
        products=products,
        engine_config=pde_engine_config(),
        hedge=HedgeSpec(kind="futures", multiplier=200.0, roll_policy=FuturesRollPolicy()),
        strategy=AutocallableDeltaHedgeStrategy(delta_threshold=0.0, hedge_ratio=1.0, target_delta=0.0),
        transaction_cost_model=ZeroCostModel(),
        pricing=PricingProviderConfig(
            provider="repricing",
            cache=CacheConfig(memory_bytes=8_000_000),
            gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0),
        ),
        underlying="CSI1000",
    )
    kwargs.update(overrides)
    return EnsembleConfig(**kwargs)
