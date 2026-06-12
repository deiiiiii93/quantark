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


class TestSemiStaticBacktest:
    def test_hedges_only_at_events(self):
        from quantark.backtest.strategy import SemiStaticHedgeStrategy

        obs_dates = [datetime(2024, 1, 15), datetime(2024, 2, 15)]
        strategy = SemiStaticHedgeStrategy(rebalance_dates=obs_dates)
        engine = BacktestEngine(make_config(strategy))
        results = engine.run()

        # Trade date + two observation dates = exactly 3 rebalances
        assert engine._num_hedges_executed == 3
        hedge_dates = {
            t.timestamp.date()
            for state in engine.state_tracker.states
            for t in state.trades
        }
        assert hedge_dates == {
            START.date(),
            obs_dates[0].date(),
            obs_dates[1].date(),
        }
        assert results is not None


class TestScenarioBacktest:
    def test_scenario_strategy_reduces_scenario_loss(self):
        from quantark.backtest.strategy import MarketScenario, ScenarioHedgeStrategy
        from quantark.backtest.strategy.scenarios import portfolio_scenario_pnl

        scenarios = [
            MarketScenario("spot_down_10", spot_shift=-0.10),
            MarketScenario("vol_up_8", vol_shift=0.08),
        ]
        strategy = ScenarioHedgeStrategy(
            scenarios=scenarios,
            pnl_threshold=100.0,
            rebalance_frequency="continuous",
        )
        engine = BacktestEngine(make_config(strategy))
        engine.run()

        # After the final rebalance, scenario P&L is inside the threshold
        for scenario in scenarios:
            residual = portfolio_scenario_pnl(
                engine.portfolio, UNDERLYING, engine.pricing_env, scenario
            )
            assert abs(residual) <= 100.0 + 1e-6


class TestTriggeredHedgeBacktest:
    def test_impossible_trigger_never_hedges(self):
        from quantark.backtest.strategy import HedgeTrigger, TriggeredHedgeStrategy

        strategy = TriggeredHedgeStrategy(
            triggers=[HedgeTrigger("deep_crash", spot_drawdown=0.90)],
        )
        engine = BacktestEngine(make_config(strategy))
        engine.run()
        # The short-call delta drifts the whole backtest, but the trigger
        # never fires, so the strategy stays inactive
        assert engine._num_hedges_executed == 0

    def test_fired_trigger_enables_delta_hedging(self):
        from quantark.backtest.strategy import HedgeTrigger, TriggeredHedgeStrategy

        # Any movement away from the start fires one of these
        strategy = TriggeredHedgeStrategy(
            triggers=[
                HedgeTrigger("any_dip", spot_drawdown=1e-9),
                HedgeTrigger("any_rally", spot_rally=1e-9),
            ],
        )
        engine = BacktestEngine(make_config(strategy))
        engine.run()

        assert strategy.armed
        assert engine._num_hedges_executed > 0
        greeks = engine.portfolio.get_portfolio_greeks(engine.greeks_calculator)
        # Default target: delta neutralized (threshold 0 -> solve to zero)
        assert abs(greeks["delta"]) < 1e-6


class TestBarrierTriggerBacktest:
    def test_barrier_trigger_runs_on_existing_engine(self):
        from quantark.backtest.strategy import BarrierTriggerHedgeStrategy

        strategy = BarrierTriggerHedgeStrategy(
            barrier_level=80.0,
            far_delta_threshold=50.0,
            near_delta_threshold=10.0,
        )
        engine = BacktestEngine(make_config(strategy))
        results = engine.run()
        assert results is not None
        assert engine._num_hedges_executed > 0


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
