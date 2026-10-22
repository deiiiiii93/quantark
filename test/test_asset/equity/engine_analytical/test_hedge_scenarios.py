"""
Unit tests for market scenarios and scenario P&L helpers.
"""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.backtest.strategy.scenarios import (
    MarketScenario,
    apply_scenario,
    instrument_scenario_pnl,
    portfolio_scenario_pnl,
)
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


def make_env(spot=100.0, vol=0.2, rate=0.03):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name=UNDERLYING),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=T0,
    )


class TestMarketScenario:
    def test_valid_scenario(self):
        scenario = MarketScenario("crash", spot_shift=-0.10, vol_shift=0.08)
        assert scenario.name == "crash"

    def test_invalid_parameters(self):
        with pytest.raises(ValidationError):
            MarketScenario("", spot_shift=-0.10)
        with pytest.raises(ValidationError):
            MarketScenario("too_far", spot_shift=-1.0)
        with pytest.raises(ValidationError):
            MarketScenario("no_move")  # all shifts zero
        with pytest.raises(ValidationError):
            MarketScenario("bad_weight", spot_shift=0.1, weight=0.0)


class TestApplyScenario:
    def test_joint_shift(self):
        env = make_env(spot=100.0, vol=0.2, rate=0.03)
        scenario = MarketScenario(
            "crash", spot_shift=-0.10, vol_shift=0.08, rate_shift=-0.01
        )
        bumped = apply_scenario(env, scenario)

        assert bumped.spot == pytest.approx(90.0)
        assert bumped.get_vol(100.0, 1.0) == pytest.approx(0.28)
        assert bumped.get_rate(1.0) == pytest.approx(0.02)
        # Original untouched; unaffected fields carried over
        assert env.spot == pytest.approx(100.0)
        assert bumped.valuation_date == env.valuation_date
        assert bumped.get_div_yield(1.0) == pytest.approx(0.01)

    def test_partial_shift_keeps_other_params(self):
        env = make_env()
        bumped = apply_scenario(env, MarketScenario("vol_up", vol_shift=0.05))
        assert bumped.spot == pytest.approx(100.0)
        assert bumped.get_vol(100.0, 1.0) == pytest.approx(0.25)
        assert bumped.get_rate(1.0) == pytest.approx(0.03)


class TestScenarioPnl:
    def test_instrument_pnl_matches_direct_repricing(self):
        env = make_env()
        option = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )
        engine = BlackScholesEngine()
        scenario = MarketScenario("crash", spot_shift=-0.10, vol_shift=0.08)

        pnl = instrument_scenario_pnl(option, engine, env, scenario)
        expected = engine.price(option, apply_scenario(env, scenario)) - engine.price(
            option, env
        )
        assert pnl == pytest.approx(expected)

    def test_long_call_loses_in_crash(self):
        env = make_env()
        option = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )
        # Pure spot crash, no vol cushion: a long ATM call loses
        pnl = instrument_scenario_pnl(
            option, BlackScholesEngine(), env, MarketScenario("down", spot_shift=-0.10)
        )
        assert pnl < 0.0

    def test_portfolio_pnl_scales_and_restores(self):
        env = make_env()
        portfolio = Portfolio(
            portfolio_name="ScenarioTest",
            pricing_environments={UNDERLYING: env},
            creation_date=T0,
        )
        option = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )
        engine = BlackScholesEngine()
        portfolio.add_position(
            product=option,
            quantity=-100.0,
            entry_price=10.0,
            underlying=UNDERLYING,
            engine=engine,
            entry_timestamp=T0,
        )

        scenario = MarketScenario("down", spot_shift=-0.10)
        pnl = portfolio_scenario_pnl(portfolio, UNDERLYING, env, scenario)

        # Short 100 calls gain when spot crashes; equals -100x per-unit P&L
        unit = instrument_scenario_pnl(option, engine, env, scenario)
        assert pnl == pytest.approx(-100.0 * unit)
        assert pnl > 0.0
        # The original environment was restored
        assert portfolio.pricing_environments[UNDERLYING] is env
