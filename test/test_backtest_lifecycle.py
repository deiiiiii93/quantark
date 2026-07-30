"""
Lifecycle event handling in the portfolio equity backtest engine.

These tests exercise the integration of the shared
``PortfolioLifecycleManager`` into ``BacktestEngine`` via the
``handle_lifecycle_events`` flag: a barrier-family position whose barrier is
breached during the historical path settles to cash, is removed from the
portfolio, and is reported through ``results.get_lifecycle_events()``.

A deterministic ramp adapter is used so the knock-out is guaranteed (the mock
adapter's stochastic path is not suitable for an exact event assertion).
"""

from datetime import datetime

import pandas as pd

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
from quantark.backtest.strategy import DeltaNeutralStrategy
from quantark.portfolio import Position
from quantark.util.enum import OptionType
from quantark.util.enum.option_enums import BarrierType
from quantark.util.calendar import BusinessDayConvention
from quantark.util.marketdata.models import MarketDataSet, TimeSeriesData
from quantark.util.numerical import is_close, safe_exp

UNDERLYING = "EQUITY_INDEX"
START = datetime(2024, 1, 1)


class ConstantEngine:
    def __init__(self, value=7.0):
        self.value = float(value)

    def price(self, product, pricing_env):
        return self.value


class RampAdapter:
    """Deterministic adapter: spot ramps linearly from start to end."""

    def __init__(self, start_spot, end_spot, num_days, vol=0.25, rate=0.05, div=0.02):
        self.dates = pd.bdate_range(start=START, periods=num_days)
        spots = [
            start_spot + (end_spot - start_spot) * i / (num_days - 1)
            for i in range(num_days)
        ]
        self._spots = spots
        self._vol, self._rate, self._div = vol, rate, div

    def get_market_data_set(self, asset_name, start_date, end_date, currency, frequency):
        idx = self.dates
        spot = TimeSeriesData(
            pd.DataFrame({"spot": self._spots}, index=idx), asset_name, "spot"
        )
        vol = TimeSeriesData(
            pd.DataFrame({"volatility": [self._vol] * len(idx)}, index=idx),
            asset_name,
            "volatility",
        )
        rate = TimeSeriesData(
            pd.DataFrame({"rate": [self._rate] * len(idx)}, index=idx),
            asset_name,
            "rate",
        )
        div = TimeSeriesData(
            pd.DataFrame({"div_yield": [self._div] * len(idx)}, index=idx),
            asset_name,
            "div_yield",
        )
        return MarketDataSet(
            spot_data=spot, vol_data=vol, rate_data=rate, div_yield_data=div
        )


def _down_out_put_position(
    rebate=2.0,
    quantity=10.0,
    *,
    settlement_convention=None,
    engine=None,
):
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.PUT,
        barrier=85.0,
        barrier_type=BarrierType.DOWN_OUT,
        maturity=1.0,
        rebate=rebate,
        pay_at_hit=True,
        contract_multiplier=1.0,
        settlement_convention=settlement_convention,
    )
    return Position(
        product=option,
        quantity=quantity,
        entry_price=5.0,
        underlying=UNDERLYING,
        engine=engine or BarrierAnalyticalEngine(),
        entry_timestamp=START,
    )


def _make_config(
    handle_lifecycle_events=True,
    rebate=2.0,
    quantity=10.0,
    *,
    settlement_convention=None,
    engine=None,
    calculate_greeks=True,
):
    # Spot ramps 100 -> 70 over 40 business days; crosses the 85 KO barrier.
    adapter = RampAdapter(start_spot=100.0, end_spot=70.0, num_days=40)
    return BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12),  # never hedges
        start_date=START,
        end_date=adapter.dates[-1].to_pydatetime(),
        underlying=UNDERLYING,
        initial_positions=[
            _down_out_put_position(
                rebate,
                quantity,
                settlement_convention=settlement_convention,
                engine=engine,
            )
        ],
        market_data_adapter=adapter,
        transaction_cost_model=ZeroCostModel(),
        handle_lifecycle_events=handle_lifecycle_events,
        calculate_greeks=calculate_greeks,
    )


class TestBacktestLifecycle:
    def test_knock_out_recorded_and_settled(self):
        rebate, quantity = 2.0, 10.0
        engine = BacktestEngine(_make_config(rebate=rebate, quantity=quantity))
        results = engine.run()

        events = results.get_lifecycle_events()
        assert isinstance(events, pd.DataFrame)
        assert len(events) == 1
        row = events.iloc[0]
        assert row["event_type"] == "KO"
        assert row["terminates_position"] is True or bool(row["terminates_position"])
        assert row["product_type"] == "BarrierOption"

        # pay_at_hit rebate: cashflow = quantity * rebate * multiplier
        expected_cash = quantity * rebate * 1.0
        assert is_close(row["cashflow"], expected_cash)

    def test_settled_cash_keeps_value_continuous(self):
        rebate, quantity = 2.0, 10.0
        engine = BacktestEngine(_make_config(rebate=rebate, quantity=quantity))
        results = engine.run()

        states = results.states_df
        # After the only position knocks out, recorded portfolio value equals
        # the booked settlement cash (no other positions, no hedging).
        expected_cash = quantity * rebate * 1.0
        final_value = states["portfolio_value"].iloc[-1]
        assert is_close(final_value, expected_cash)
        assert is_close(states["pending_receivable_pv"].iloc[-1], 0.0)
        assert is_close(states["paid_cash"].iloc[-1], expected_cash)
        assert is_close(states["realized_cash"].iloc[-1], expected_cash)

        # The position is gone by the end of the run.
        assert states["num_positions"].iloc[-1] == 0

    def test_disabled_flag_leaves_position_alive(self):
        # With lifecycle handling off, the barrier option is never knocked out;
        # it is repriced as a live barrier option every day and survives.
        engine = BacktestEngine(_make_config(handle_lifecycle_events=False))
        results = engine.run()

        events = results.get_lifecycle_events()
        assert len(events) == 0
        assert results.states_df["num_positions"].iloc[-1] == 1

    def test_delayed_settlement_is_pending_before_it_becomes_cash(self):
        convention = SettlementConvention(
            lag=2,
            lag_unit=SettlementLagUnit.CALENDAR_DAYS,
            business_day_convention=BusinessDayConvention.UNADJUSTED,
        )
        engine = BacktestEngine(
            _make_config(
                settlement_convention=convention,
                engine=ConstantEngine(),
                calculate_greeks=False,
            )
        )
        results = engine.run()
        states = results.states_df
        event_date = results.get_lifecycle_events().index[0]
        payment_date = event_date + pd.Timedelta(days=2)
        amount = 20.0

        assert is_close(states.loc[event_date, "paid_cash"], 0.0)
        assert is_close(
            states.loc[event_date, "pending_receivable_pv"],
            amount * safe_exp(-0.05 * 2.0 / 365.0),
        )
        assert is_close(
            states.loc[payment_date, "pending_receivable_pv"],
            0.0,
        )
        assert is_close(states.loc[payment_date, "paid_cash"], amount)
        assert is_close(states["paid_cash"].iloc[-1], amount)
