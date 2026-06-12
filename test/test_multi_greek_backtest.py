"""
Integration tests: multi-Greek strategies running through BacktestEngine.
"""

from datetime import datetime

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
from quantark.backtest.strategy.multi_greek_strategy import (
    DeltaGammaNeutralStrategy,
    DeltaGammaVegaNeutralStrategy,
)
from quantark.portfolio import Position
from quantark.util.enum import OptionType
from quantark.util.marketdata.adapter.mock_adapter import MockMarketDataAdapter

UNDERLYING = "TEST"
START = datetime(2024, 1, 1)
END = datetime(2024, 3, 1)


def make_config(strategy):
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    short_calls = Position(
        product=option,
        quantity=-500.0,
        entry_price=10.0,
        underlying=UNDERLYING,
        engine=BlackScholesEngine(),
        entry_timestamp=START,
    )
    return BacktestConfig(
        strategy=strategy,
        start_date=START,
        end_date=END,
        underlying=UNDERLYING,
        initial_positions=[short_calls],
        market_data_adapter=MockMarketDataAdapter(seed=42),
        transaction_cost_model=ZeroCostModel(),
    )


class TestDeltaGammaVegaBacktest:
    def test_backtest_neutralizes_all_greeks(self):
        strategy = DeltaGammaVegaNeutralStrategy(
            delta_threshold=1.0,
            gamma_threshold=0.5,
            vega_threshold=0.5,
            rebalance_frequency="continuous",
        )
        engine = BacktestEngine(make_config(strategy))
        results = engine.run()

        # The final hedge left the book inside every threshold at the
        # final pricing environment
        greeks = engine.portfolio.get_portfolio_greeks(engine.greeks_calculator)
        assert abs(greeks["delta"]) <= 1.0 + 1e-6
        assert abs(greeks["gamma"]) <= 0.5 + 1e-6
        assert abs(greeks["vega"]) <= 0.5 + 1e-6

        # Hedge book contains the strategy's instruments plus the original
        # short calls
        assert results is not None
        assert engine._num_hedges_executed > 0
        assert len(engine.portfolio.positions) >= 3

    def test_results_pnl_is_finite_and_recorded(self):
        strategy = DeltaGammaVegaNeutralStrategy(
            delta_threshold=1.0,
            gamma_threshold=0.5,
            vega_threshold=0.5,
            rebalance_frequency="continuous",
        )
        engine = BacktestEngine(make_config(strategy))
        results = engine.run()

        pnl_series = results.get_pnl_series()
        assert len(pnl_series) > 0
        assert pnl_series.notna().all()


class TestDeltaGammaBacktest:
    def test_delta_gamma_strategy_runs(self):
        strategy = DeltaGammaNeutralStrategy(
            delta_threshold=1.0,
            gamma_threshold=0.5,
            rebalance_frequency="continuous",
        )
        engine = BacktestEngine(make_config(strategy))
        engine.run()

        greeks = engine.portfolio.get_portfolio_greeks(engine.greeks_calculator)
        assert abs(greeks["delta"]) <= 1.0 + 1e-6
        assert abs(greeks["gamma"]) <= 0.5 + 1e-6


class TestSingleInstrumentStrategiesStillWork:
    def test_whalley_wilmott_runs_on_existing_engine(self):
        from quantark.backtest.strategy.whalley_wilmott_strategy import (
            WhalleyWilmottStrategy,
        )

        strategy = WhalleyWilmottStrategy(risk_aversion=0.1, cost_rate=0.001)
        engine = BacktestEngine(make_config(strategy))
        results = engine.run()
        assert results is not None

    def test_min_variance_delta_runs_on_existing_engine(self):
        from quantark.backtest.strategy.min_variance_delta_strategy import (
            MinimumVarianceDeltaStrategy,
        )

        strategy = MinimumVarianceDeltaStrategy(
            vol_spot_slope=-0.0005, delta_threshold=10.0,
            rebalance_frequency="continuous",
        )
        engine = BacktestEngine(make_config(strategy))
        results = engine.run()
        assert results is not None
