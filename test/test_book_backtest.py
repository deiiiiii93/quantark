import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantark.asset.equity.product.option import EuropeanVanillaOption, create_standard_snowball
from quantark.backtest.otc import (
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
)
from quantark.util.enum.engine_enums import EngineType


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
        product_quantity=-350.0, underlying="CSI500",
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
    # product_quantity=-350.0 ensures the rounded futures target is nonzero,
    # exercising the hedge/rebalance/roll path.
    # These are the characterization anchors for the Task 1.2 refactor.
    # Re-pinned 2026-07-24: hedge deltas now use BumpConfig's documented 1%
    # spot bump (deprecated EngineParams.bump_size=1e-4 shim retired).
    assert single_summary["num_days"] == 86
    assert single_summary["num_trades"] == 5
    assert single_summary["num_trades"] > 0, "hedging must produce at least one trade"
    assert single_summary["total_pnl"] == pytest.approx(-3417.8369450558384, rel=1e-9)


def test_book_config_rejects_empty_products():
    from quantark.backtest.otc.book_engine import BookAutocallableBacktestConfig
    market = _synthetic_market()
    with pytest.raises(Exception):
        BookAutocallableBacktestConfig(products=[], market_data=market)


def test_results_summary_empty_states():
    from quantark.backtest.otc.book_engine import BookBacktestResults
    r = BookBacktestResults(config=None, states=[], greeks=[], rebalances=[], trades=[],
                            actions=[], daily_event_summary=[], event_probabilities=[],
                            surfaces=[], products_meta=[{"position_id": 1}])
    s = r.get_summary()
    assert s["num_days"] == 0 and s["num_products"] == 1


def test_book_of_one_matches_single_product(single_summary):
    from quantark.backtest.otc.book_engine import (
        BookAutocallableBacktestConfig,
        BookAutocallableBacktestEngine,
        BookProduct,
        HedgeSpec,
    )

    market = _synthetic_market()
    # Build the SAME snowball as _single_config (exact constructor call).
    snowball = create_standard_snowball(
        initial_price=6000.0,
        strike=6000.0,
        maturity=1.0,
        ko_barrier=6180.0,   # 103% of initial_price
        ki_barrier=4500.0,   # 75% of initial_price
        ko_rate=0.15,
    )
    cfg = BookAutocallableBacktestConfig(
        products=[BookProduct(product=snowball, quantity=-350.0, position_id=1, has_lifecycle=True)],
        market_data=market,
        hedge=HedgeSpec(kind="futures", multiplier=200.0),
        engine_config=AutocallableEngineConfig(pricing_engine_type=EngineType.QUADRATURE),
        underlying="CSI500",
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 4, 30),
        calculate_surfaces=False,
        calculate_event_probabilities=False,
    )
    book = BookAutocallableBacktestEngine(cfg).run().get_summary()
    assert book["num_days"] == single_summary["num_days"]      # 86
    assert book["num_trades"] == single_summary["num_trades"]  # 7
    assert book["total_pnl"] == pytest.approx(single_summary["total_pnl"], rel=1e-9)  # -9580.96...


def test_offsetting_products_net_to_near_zero_hedge():
    """Two equal-and-opposite positions on the same underlying net to ~zero delta,
    producing materially fewer hedge trades than the single -350 book (7 trades)."""
    from quantark.backtest.otc.book_engine import (
        BookAutocallableBacktestConfig,
        BookAutocallableBacktestEngine,
        BookProduct,
        HedgeSpec,
    )

    market = _synthetic_market()
    sb = create_standard_snowball(
        initial_price=6000.0,
        strike=6000.0,
        maturity=1.0,
        ko_barrier=6180.0,
        ki_barrier=4500.0,
        ko_rate=0.15,
    )
    cfg = BookAutocallableBacktestConfig(
        products=[
            BookProduct(product=sb, quantity=-350.0, position_id=1, has_lifecycle=True),
            BookProduct(product=sb, quantity=350.0, position_id=2, has_lifecycle=True),
        ],
        market_data=market,
        hedge=HedgeSpec(kind="futures", multiplier=200.0),
        engine_config=AutocallableEngineConfig(pricing_engine_type=EngineType.QUADRATURE),
        underlying="CSI500",
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 4, 30),
        calculate_event_probabilities=False,
    )
    summary = BookAutocallableBacktestEngine(cfg).run().get_summary()
    assert summary["num_products"] == 2
    # Net delta ~0 after rounding -> essentially no hedge trades.
    # The single -350 book produces 7 trades; offsetting legs must be materially fewer.
    # (A residual ±1 contract on some days is possible from integer rounding.)
    assert summary["num_trades"] <= 3


def test_spot_hedge_mode_runs():
    """Spot-hedged book runs end-to-end (no futures roll), trades are instrument_type='spot',
    and total P&L is finite."""
    from quantark.backtest.otc.book_engine import (
        BookAutocallableBacktestConfig,
        BookAutocallableBacktestEngine,
        BookProduct,
        HedgeSpec,
    )

    market = _synthetic_market()
    sb = create_standard_snowball(
        initial_price=6000.0,
        strike=6000.0,
        maturity=1.0,
        ko_barrier=6180.0,
        ki_barrier=4500.0,
        ko_rate=0.15,
    )
    cfg = BookAutocallableBacktestConfig(
        products=[BookProduct(product=sb, quantity=-350.0, position_id=1, has_lifecycle=True)],
        market_data=market,
        hedge=HedgeSpec(kind="spot", multiplier=1.0),
        engine_config=AutocallableEngineConfig(pricing_engine_type=EngineType.QUADRATURE),
        underlying="CSI500",
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 4, 30),
        calculate_event_probabilities=False,
    )
    results = BookAutocallableBacktestEngine(cfg).run()
    summary = results.get_summary()
    assert summary["num_days"] > 0
    assert np.isfinite(summary["total_pnl"])
    trades = results.trades_df()
    if not trades.empty:
        assert (trades["instrument_type"] == "spot").any()


def test_book_backtest_supports_european_quad_engine():
    from quantark.backtest.otc.book_engine import (
        BookAutocallableBacktestConfig,
        BookAutocallableBacktestEngine,
        BookProduct,
        HedgeSpec,
    )
    from quantark.util.enum.option_enums import OptionType

    market = _synthetic_market(start="2024-01-02", end="2024-01-31")
    option = EuropeanVanillaOption(6000.0, OptionType.CALL, maturity=1.0)
    cfg = BookAutocallableBacktestConfig(
        products=[BookProduct(product=option, quantity=100.0, position_id=1, has_lifecycle=False)],
        market_data=market,
        hedge=HedgeSpec(kind="spot", multiplier=1.0),
        engine_config=AutocallableEngineConfig(pricing_engine_type=EngineType.QUADRATURE),
        underlying="CSI500",
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 1, 31),
        calculate_surfaces=False,
        calculate_event_probabilities=False,
    )

    summary = BookAutocallableBacktestEngine(cfg).run().get_summary()

    assert summary["num_days"] > 0
    assert np.isfinite(summary["total_pnl"])
