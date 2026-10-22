"""Gate 4 tests: equity TRS in the backtest engine.

A swap position is registered via add_swap_position and revalued through the
BasePosition interface as the engine steps a spot path. Covers an unhedged
delta-one run and a delta-hedged run.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.util.calendar.business_calendar import Calendar
from quantark.asset.equity.product.swap import (
    AssetParams,
    FixLegParams,
    FloatLegParams,
    PricingParams,
    TRSParams,
    OneAssetTotalReturnSwap,
)
from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
from quantark.backtest.strategy import DeltaNeutralStrategy
from quantark.portfolio import EquitySwapPosition
from quantark.util.marketdata.models import MarketDataSet, TimeSeriesData

UNDERLYING = "IDX"
TRS_START, TRS_END, VALUATION = "2024-01-02", "2024-01-31", "2024-01-30"
INITIAL, VAL_SPOT, NOTIONAL = 100.0, 100.0, 1_000_000.0
QTY_SHARES = NOTIONAL / INITIAL
BT_START = datetime(2024, 1, 1)


class RampAdapter:
    """Deterministic adapter: spot ramps linearly from start to end."""

    def __init__(self, start_spot, end_spot, num_days, vol=0.25, rate=0.048, div=0.0):
        self.dates = pd.bdate_range(start=BT_START, periods=num_days)
        self._spots = [
            start_spot + (end_spot - start_spot) * i / (num_days - 1)
            for i in range(num_days)
        ]
        self._vol, self._rate, self._div = vol, rate, div

    def get_market_data_set(self, asset_name, start_date, end_date, currency, frequency):
        idx = self.dates
        return MarketDataSet(
            spot_data=TimeSeriesData(
                pd.DataFrame({"spot": self._spots}, index=idx), asset_name, "spot"
            ),
            vol_data=TimeSeriesData(
                pd.DataFrame({"volatility": [self._vol] * len(idx)}, index=idx),
                asset_name, "volatility",
            ),
            rate_data=TimeSeriesData(
                pd.DataFrame({"rate": [self._rate] * len(idx)}, index=idx),
                asset_name, "rate",
            ),
            div_yield_data=TimeSeriesData(
                pd.DataFrame({"div_yield": [self._div] * len(idx)}, index=idx),
                asset_name, "div_yield",
            ),
        )


def _swap_position(quantity=1.0) -> EquitySwapPosition:
    days = pd.date_range(TRS_START, TRS_END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    path = pd.Series([INITIAL] * len(idx), index=idx)
    calendar = Calendar(name="DemoCalendar")
    swap = OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="TRS_BT",
            asset=AssetParams("IDX", INITIAL, path),
            fix_leg=FixLegParams(
                rate=0.048, notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=TRS_START, end_date=TRS_END,
                payment_calendar=calendar, direction=-1,
            ),
            float_leg=FloatLegParams(
                notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=TRS_START, end_date=TRS_END,
                payment_calendar=calendar, direction=1,
            ),
            pricing=PricingParams(valuation_date=VALUATION, output_mode="spot"),
        )
    )
    return EquitySwapPosition(
        product=swap, quantity=quantity, underlying=UNDERLYING, entry_timestamp=BT_START
    )


def _config(strategy, start_spot, end_spot, num_days=20, handle_lifecycle=True):
    adapter = RampAdapter(start_spot=start_spot, end_spot=end_spot, num_days=num_days)
    return BacktestConfig(
        strategy=strategy,
        start_date=BT_START,
        end_date=adapter.dates[-1].to_pydatetime(),
        underlying=UNDERLYING,
        initial_positions=[_swap_position()],
        market_data_adapter=adapter,
        transaction_cost_model=ZeroCostModel(),
        handle_lifecycle_events=handle_lifecycle,
    )


def test_unhedged_swap_tracks_spot_ramp():
    """Unhedged long TRS gains ~ qty_shares * (end - start) over a rising ramp."""
    never_hedge = DeltaNeutralStrategy(delta_threshold=1e12)
    results = BacktestEngine(
        _config(never_hedge, start_spot=100.0, end_spot=120.0)
    ).run()

    pnl = results.get_total_pnl()
    expected = QTY_SHARES * (120.0 - 100.0)
    assert pnl == pytest.approx(expected, rel=0.05)
    assert results.num_hedges == 0


def test_swap_delta_drives_hedging():
    """The swap's delta-one exposure flows into the hedging loop.

    A tight delta threshold against the swap's large (qty_shares) delta makes the
    DeltaNeutralStrategy trade the underlying — proving get_portfolio_greeks
    surfaces the swap's delta and the executor hedges it. (PnL levels are not
    asserted: this engine marks the spot hedge at market value without booking
    the offsetting financing cash, a pre-existing accounting characteristic
    independent of the swap.)
    """
    hedged = BacktestEngine(
        _config(DeltaNeutralStrategy(delta_threshold=100.0),
                start_spot=100.0, end_spot=120.0)
    ).run()
    assert hedged.num_hedges > 0

    trades = hedged.state_tracker.get_trades_dataframe()
    assert not trades.empty
    assert (trades["underlying"] == UNDERLYING).any()
