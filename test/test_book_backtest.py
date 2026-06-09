import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asset.equity.product.option import create_standard_snowball
from backtest.otc import (
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
)
from util.enum.engine_enums import EngineType


def _synthetic_market(start="2024-01-02", end="2024-04-30", spot0=6000.0):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0, 0.012, len(dates))
    spot = spot0 * np.exp(np.cumsum(rets))
    spot_data = pd.DataFrame({"date": dates, "spot": spot})
    vol_data = pd.DataFrame({"date": dates, "volatility": np.full(len(dates), 0.18)})
    rate_data = pd.DataFrame({"date": dates, "rate": np.full(len(dates), 0.02)})
    expiry = pd.Timestamp("2024-06-21")
    futures_data = pd.DataFrame({
        "date": dates,
        "contract": ["IC2406"] * len(dates),
        "futures_price": spot * 0.99,
        "expiry_date": [expiry] * len(dates),
        "multiplier": [200.0] * len(dates),
    })
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=spot_data, vol_data=vol_data, rate_data=rate_data, futures_data=futures_data
    )


def _single_config(market):
    snowball = create_standard_snowball(
        initial_price=6000.0,
        strike=6000.0,
        maturity=1.0,
        ko_barrier=6180.0,   # 103% of initial_price
        ki_barrier=4500.0,   # 75% of initial_price
        ko_rate=0.15,
    )
    return AutocallableBacktestConfig(
        product=snowball, market_data=market,
        engine_config=AutocallableEngineConfig(pricing_engine_type=EngineType.QUADRATURE),
        product_quantity=-1.0, underlying="CSI500",
        start_date=datetime(2024, 1, 2), end_date=datetime(2024, 4, 30),
        calculate_surfaces=False, calculate_event_probabilities=False,
    )


@pytest.fixture(scope="module")
def single_summary():
    market = _synthetic_market()
    results = AutocallableBacktestEngine(_single_config(market)).run()
    return results.get_summary()


def test_single_product_summary_is_stable(single_summary):
    # Pinned values recorded from the deterministic synthetic-market run.
    # These are the characterization anchors for the Task 1.2 refactor.
    assert single_summary["num_days"] == 86
    assert single_summary["num_trades"] == 0
    assert single_summary["total_pnl"] == pytest.approx(1.0482609702399657, rel=1e-9)
