"""
Unit tests for the multi-instrument hedge executor.
"""

from datetime import datetime, timedelta

import pytest

from quantark.backtest.equity.multi_hedge_executor import MultiInstrumentHedgeExecutor
from quantark.backtest.strategy.hedge_instruments import (
    OptionHedgeInstrument,
    SpotHedgeInstrument,
)
from quantark.backtest.transaction_costs import ProportionalCostModel, ZeroCostModel
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

UNDERLYING = "TEST"
T0 = datetime(2026, 1, 2)


def make_env(spot=100.0, valuation_date=T0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name=UNDERLYING),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=valuation_date,
    )


@pytest.fixture
def env():
    return make_env()


@pytest.fixture
def portfolio(env):
    return Portfolio(
        portfolio_name="ExecutorTest",
        pricing_environments={UNDERLYING: env},
        creation_date=T0,
    )


def make_executor(portfolio, cost_model=None):
    return MultiInstrumentHedgeExecutor(
        portfolio=portfolio,
        transaction_cost_model=cost_model or ZeroCostModel(),
        instruments=[
            OptionHedgeInstrument(tenor=0.25, name="gamma_option"),
            SpotHedgeInstrument(),
        ],
    )


class TestInstrumentGreeks:
    def test_greeks_for_all_instruments(self, portfolio, env):
        executor = make_executor(portfolio)
        greeks = executor.get_instrument_greeks(UNDERLYING, env, T0)

        assert set(greeks) == {"gamma_option", "spot"}
        assert greeks["gamma_option"]["gamma"] > 0.0
        assert greeks["gamma_option"]["vega"] > 0.0
        assert greeks["spot"]["delta"] == pytest.approx(1.0, abs=1e-6)
        assert greeks["spot"]["gamma"] == pytest.approx(0.0, abs=1e-6)


class TestExecution:
    def test_open_positions(self, portfolio, env):
        executor = make_executor(portfolio)
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        records = executor.execute_hedges(
            UNDERLYING, {"gamma_option": 30.0, "spot": -25.0}, env, T0
        )

        assert len(records) == 2
        assert all(r.trade_type == "open" for r in records)
        assert len(portfolio.positions) == 2
        # The traded option is the contract the optimizer saw: ATM at spot 100
        option_position = portfolio.positions[
            executor.get_position_id("gamma_option")
        ]
        assert option_position.product.strike == pytest.approx(100.0)
        assert option_position.quantity == pytest.approx(30.0)

    def test_zero_quantity_skipped(self, portfolio, env):
        executor = make_executor(portfolio)
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        records = executor.execute_hedges(
            UNDERLYING, {"gamma_option": 0.0, "spot": -25.0}, env, T0
        )
        assert len(records) == 1
        assert len(portfolio.positions) == 1

    def test_unknown_instrument_raises(self, portfolio, env):
        executor = make_executor(portfolio)
        with pytest.raises(ValidationError):
            executor.execute_hedges(UNDERLYING, {"mystery": 1.0}, env, T0)

    def test_increase_blends_entry_price(self, portfolio):
        env = make_env(spot=100.0)
        portfolio.pricing_environments[UNDERLYING] = env
        executor = make_executor(portfolio)
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        executor.execute_hedges(UNDERLYING, {"spot": 10.0}, env, T0)

        env.spot_quote = SpotQuote(spot=110.0, asset_name=UNDERLYING)
        executor.execute_hedges(UNDERLYING, {"spot": 10.0}, env, T0)

        position = portfolio.positions[executor.get_position_id("spot")]
        assert position.quantity == pytest.approx(20.0)
        assert position.entry_price == pytest.approx(105.0)  # (100+110)/2
        assert executor.realized_pnl == pytest.approx(0.0)

    def test_reduce_realizes_pnl(self, portfolio):
        env = make_env(spot=100.0)
        portfolio.pricing_environments[UNDERLYING] = env
        executor = make_executor(portfolio)
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        executor.execute_hedges(UNDERLYING, {"spot": 10.0}, env, T0)

        env.spot_quote = SpotQuote(spot=110.0, asset_name=UNDERLYING)
        executor.execute_hedges(UNDERLYING, {"spot": -4.0}, env, T0)

        position = portfolio.positions[executor.get_position_id("spot")]
        assert position.quantity == pytest.approx(6.0)
        assert position.entry_price == pytest.approx(100.0)  # unchanged
        assert executor.realized_pnl == pytest.approx(40.0)  # (110-100)*4

    def test_full_close_removes_position(self, portfolio):
        env = make_env(spot=100.0)
        portfolio.pricing_environments[UNDERLYING] = env
        executor = make_executor(portfolio)
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        executor.execute_hedges(UNDERLYING, {"spot": 10.0}, env, T0)

        env.spot_quote = SpotQuote(spot=110.0, asset_name=UNDERLYING)
        records = executor.execute_hedges(UNDERLYING, {"spot": -10.0}, env, T0)

        assert records[0].trade_type == "close"
        assert len(portfolio.positions) == 0
        assert executor.realized_pnl == pytest.approx(100.0)  # (110-100)*10

    def test_sign_flip_realizes_and_reenters(self, portfolio):
        env = make_env(spot=100.0)
        portfolio.pricing_environments[UNDERLYING] = env
        executor = make_executor(portfolio)
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        executor.execute_hedges(UNDERLYING, {"spot": 10.0}, env, T0)

        env.spot_quote = SpotQuote(spot=110.0, asset_name=UNDERLYING)
        executor.execute_hedges(UNDERLYING, {"spot": -16.0}, env, T0)

        position = portfolio.positions[executor.get_position_id("spot")]
        assert position.quantity == pytest.approx(-6.0)
        assert position.entry_price == pytest.approx(110.0)  # re-entered
        assert executor.realized_pnl == pytest.approx(100.0)  # old lot realized

    def test_transaction_costs_recorded(self, portfolio):
        env = make_env(spot=100.0)
        portfolio.pricing_environments[UNDERLYING] = env
        executor = make_executor(
            portfolio, cost_model=ProportionalCostModel(commission_rate=0.001)
        )
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        records = executor.execute_hedges(UNDERLYING, {"spot": 10.0}, env, T0)
        assert records[0].transaction_cost == pytest.approx(10.0 * 100.0 * 0.001)


class TestRolls:
    def test_near_expiry_contract_is_rolled(self, portfolio):
        env = make_env(spot=100.0)
        portfolio.pricing_environments[UNDERLYING] = env
        executor = make_executor(portfolio)
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        executor.execute_hedges(UNDERLYING, {"gamma_option": 30.0}, env, T0)

        option = portfolio.positions[
            executor.get_position_id("gamma_option")
        ].product

        # Nothing to roll while the contract is fresh
        assert executor.process_rolls(UNDERLYING, env, T0) == []

        # 3 days to expiry < min_time_to_expiry (7d): the contract rolls
        roll_date = option.exercise_date - timedelta(days=3)
        env.valuation_date = roll_date
        records = executor.process_rolls(UNDERLYING, env, roll_date)

        assert len(records) == 1
        assert records[0].trade_type == "close"
        assert records[0].reason == "roll"
        assert len(portfolio.positions) == 0
        # Realized P&L picked up the option's value change since entry
        assert executor.realized_pnl != 0.0

    def test_next_greeks_use_fresh_contract_after_roll(self, portfolio):
        env = make_env(spot=100.0)
        portfolio.pricing_environments[UNDERLYING] = env
        executor = make_executor(portfolio)
        executor.get_instrument_greeks(UNDERLYING, env, T0)
        executor.execute_hedges(UNDERLYING, {"gamma_option": 30.0}, env, T0)

        option = portfolio.positions[
            executor.get_position_id("gamma_option")
        ].product
        roll_date = option.exercise_date - timedelta(days=3)
        env.valuation_date = roll_date
        env.spot_quote = SpotQuote(spot=120.0, asset_name=UNDERLYING)
        executor.process_rolls(UNDERLYING, env, roll_date)

        executor.get_instrument_greeks(UNDERLYING, env, roll_date)
        records = executor.execute_hedges(
            UNDERLYING, {"gamma_option": 25.0}, env, roll_date
        )
        new_option = portfolio.positions[
            executor.get_position_id("gamma_option")
        ].product
        # Fresh contract re-struck ATM at the new spot with the full tenor
        assert new_option.strike == pytest.approx(120.0)
        assert records[0].trade_type == "open"


class TestValidation:
    def test_duplicate_instrument_names_raise(self, portfolio):
        with pytest.raises(ValidationError):
            MultiInstrumentHedgeExecutor(
                portfolio=portfolio,
                transaction_cost_model=ZeroCostModel(),
                instruments=[SpotHedgeInstrument(), SpotHedgeInstrument()],
            )
