"""
Unit tests for the scenario-based hedging strategy.
"""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.backtest.equity.multi_hedge_executor import MultiInstrumentHedgeExecutor
from quantark.backtest.strategy.scenario_hedge_strategy import ScenarioHedgeStrategy
from quantark.backtest.strategy.scenarios import (
    MarketScenario,
    instrument_scenario_pnl,
    portfolio_scenario_pnl,
)
from quantark.backtest.transaction_costs import ZeroCostModel
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

UNDERLYING = "TEST"
T0 = datetime(2026, 1, 2)

SCENARIOS = [
    MarketScenario("spot_down_10", spot_shift=-0.10),
    MarketScenario("vol_up_8", vol_shift=0.08),
]


def make_env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name=UNDERLYING),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=T0,
    )


class TestConstruction:
    def test_targets_from_scenarios(self):
        strategy = ScenarioHedgeStrategy(
            scenarios=SCENARIOS, pnl_threshold=500.0,
            thresholds={"vol_up_8": 100.0},
        )
        targets = {t.greek: t for t in strategy.targets}
        assert set(targets) == {"spot_down_10", "vol_up_8"}
        assert targets["spot_down_10"].threshold == 500.0
        assert targets["vol_up_8"].threshold == 100.0
        # Default instrument basket: two options + spot
        assert [i.name for i in strategy.hedge_instruments] == [
            "gamma_option", "vega_option", "spot",
        ]

    def test_validation(self):
        with pytest.raises(ValidationError):
            ScenarioHedgeStrategy(scenarios=[])
        with pytest.raises(ValidationError):
            ScenarioHedgeStrategy(
                scenarios=[SCENARIOS[0], MarketScenario("spot_down_10", spot_shift=-0.2)]
            )
        with pytest.raises(ValidationError):
            ScenarioHedgeStrategy(scenarios=SCENARIOS, pnl_threshold=-1.0)
        with pytest.raises(ValidationError):
            ScenarioHedgeStrategy(
                scenarios=SCENARIOS, thresholds={"mystery": 1.0}
            )


class TestMeasures:
    def test_portfolio_measures_match_helper(self):
        env = make_env()
        portfolio = Portfolio(
            portfolio_name="ScenarioStrategyTest",
            pricing_environments={UNDERLYING: env},
            creation_date=T0,
        )
        option = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )
        portfolio.add_position(
            product=option, quantity=-200.0, entry_price=10.0,
            underlying=UNDERLYING, engine=BlackScholesEngine(),
            entry_timestamp=T0,
        )

        strategy = ScenarioHedgeStrategy(scenarios=SCENARIOS)
        measures = strategy.compute_portfolio_measures(portfolio, UNDERLYING, env)

        for scenario in SCENARIOS:
            expected = portfolio_scenario_pnl(portfolio, UNDERLYING, env, scenario)
            assert measures[scenario.name] == pytest.approx(expected)
        # Short calls: vol spike hurts
        assert measures["vol_up_8"] < 0.0

    def test_instrument_measures_match_helper(self):
        env = make_env()
        strategy = ScenarioHedgeStrategy(scenarios=SCENARIOS)
        instrument = strategy.hedge_instruments[0]
        product = instrument.create_product(UNDERLYING, env, T0)
        engine = instrument.create_engine()

        measures = strategy.instrument_measures(
            instrument, product, engine, env, GreeksCalculator()
        )
        for scenario in SCENARIOS:
            expected = instrument_scenario_pnl(product, engine, env, scenario)
            assert measures[scenario.name] == pytest.approx(expected)

    def test_should_hedge_on_scenario_loss(self):
        strategy = ScenarioHedgeStrategy(
            scenarios=SCENARIOS, pnl_threshold=1000.0,
            rebalance_frequency="continuous",
        )
        inside = {"spot_down_10": -800.0, "vol_up_8": -500.0}
        assert not strategy.should_hedge(T0, inside, {})
        breached = {"spot_down_10": -1500.0, "vol_up_8": -500.0}
        assert strategy.should_hedge(T0, breached, {})


class TestEndToEndSolve:
    def test_hedge_flattens_scenario_pnl(self):
        """Square case: 2 scenarios, 2 instruments -> exact scenario hedge."""
        env = make_env()
        portfolio = Portfolio(
            portfolio_name="ScenarioSolveTest",
            pricing_environments={UNDERLYING: env},
            creation_date=T0,
        )
        book_option = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=0.5
        )
        portfolio.add_position(
            product=book_option, quantity=-300.0, entry_price=7.0,
            underlying=UNDERLYING, engine=BlackScholesEngine(),
            entry_timestamp=T0,
        )

        from quantark.backtest.strategy.hedge_instruments import (
            OptionHedgeInstrument,
            SpotHedgeInstrument,
        )

        strategy = ScenarioHedgeStrategy(
            scenarios=SCENARIOS,
            hedge_instruments=[
                OptionHedgeInstrument(name="hedge_option", tenor=1.0),
                SpotHedgeInstrument(),
            ],
            rebalance_frequency="continuous",
        )
        executor = MultiInstrumentHedgeExecutor(
            portfolio=portfolio,
            transaction_cost_model=ZeroCostModel(),
            instruments=strategy.hedge_instruments,
        )

        portfolio_measures = strategy.compute_portfolio_measures(
            portfolio, UNDERLYING, env
        )
        instrument_measures = executor.get_instrument_measures(
            UNDERLYING, env, T0, measure_provider=strategy
        )
        quantities = strategy.calculate_hedge_quantities(
            T0, portfolio_measures, {}, instrument_measures
        )
        executor.execute_hedges(UNDERLYING, quantities, env, T0)

        # Post-hedge, the full-revaluation scenario P&L is (close to) flat.
        # The solve is linear while scenario P&L is mildly nonlinear in the
        # hedge quantities only through nothing here (P&L is linear in
        # position size), so the result is exact up to solver tolerance.
        post = strategy.compute_portfolio_measures(portfolio, UNDERLYING, env)
        for scenario in SCENARIOS:
            assert abs(post[scenario.name]) < 1e-6 * 300.0 * 100.0
