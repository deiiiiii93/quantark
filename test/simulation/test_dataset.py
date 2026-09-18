from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.futures_ledger import FuturesRollPolicy
from quantark.backtest.replay import AutocallableMarketDataSet
from quantark.backtest.simulation.carry import day_chain
from quantark.backtest.simulation.dataset import to_market_dataset

from .conftest import make_market_path


def test_dataset_frames_follow_the_path():
    mp = make_market_path(n_paths=2, n_days=30, start=date(2024, 1, 2))
    ds = to_market_dataset(mp, 1)
    assert isinstance(ds, AutocallableMarketDataSet)
    assert list(ds.dates) == list(mp.dates)
    row = ds.get_market_row(mp.dates[5])
    assert row["spot"] == pytest.approx(mp.spot[1, 5])
    assert row["volatility"] == pytest.approx(mp.atm_vol[1, 5])
    assert row["rate"] == pytest.approx(mp.rate[1, 5])
    chain = ds.get_futures_slice(mp.dates[5])
    assert list(chain["contract"]) == list(day_chain(mp, 5).contracts)
    assert chain["futures_price"].to_numpy() == pytest.approx(day_chain(mp, 5).prices[1])
    selected = FuturesRollPolicy().select_contract(chain, mp.dates[5], None)
    assert selected["contract"] == "IM2401"


def test_the_carry_tail_history_serves_the_paths_own_curve_and_nothing_else():
    from quantark.backtest.simulation.dividends import CurveTailPillars
    from quantark.util.exceptions import ValidationError

    mp = make_market_path(n_paths=2, n_days=30, start=date(2024, 1, 2))
    mp.carry[1, 7, :] += np.linspace(0.0, 0.01, mp.carry.shape[2])     # make path 1, day 7 distinct
    assert to_market_dataset(mp, 1).surface_history is None            # only on request
    history = to_market_dataset(mp, 1, carry_tail=True).surface_history
    day = mp.dates[7]
    artifact = history.surface_for(day)
    expected = CurveTailPillars(tenors=mp.tenor_grid, carry=mp.carry[1, 7, :])
    assert artifact.implied_q_pillars(0.02) == expected.implied_q_pillars(0.02)
    assert artifact.trade_date == day.date() and artifact.max_listed_T == float(mp.tenor_grid[-1])
    assert artifact.sha256 != history.surface_for(mp.dates[6]).sha256
    with pytest.raises(ValidationError, match="no simulated carry curve"):
        history.surface_for(pd.Timestamp("2023-12-29"))
