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
