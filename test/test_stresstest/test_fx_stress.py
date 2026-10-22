"""
Tests for FX stress testing (FXStressEngine + FX shock applicator).

Exercises the FX-specific shock surface: spot, vol, and the two rate curves
(domestic and foreign), plus portfolio/underlying targeting and the standard
result helpers (worst/best scenario).
"""
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine, GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.util.enum import OptionType


def _env(spot=1.20, r_dom=0.05, r_for=0.03, vol=0.10):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=spot),
        domestic_curve=FlatRateCurve(rate=r_dom),
        foreign_curve=FlatRateCurve(rate=r_for),
        vol_surface=FlatVolSurface(volatility=vol),
    )


def _call():
    return FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1_000_000.0,
    )


def _long_call_portfolio():
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": _env()})
    pf.add_position(
        product=_call(), quantity=1.0, entry_price=0.0,
        underlying="EURUSD", engine=GarmanKohlhagenEngine(),
    )
    return pf


def _spot_scenario(name, pct):
    return Scenario(name=name, stresses=[
        Stress("spot", StressType.PERCENTAGE, pct, StressLevel.PORTFOLIO)
    ])


def test_zero_shock_scenario_has_no_pnl():
    from quantark.stresstest import FXStressConfig, FXStressEngine

    engine = FXStressEngine(FXStressConfig(calculate_greeks=False))
    scenario = Scenario(name="flat", stresses=[
        Stress("spot", StressType.PERCENTAGE, 0.0, StressLevel.PORTFOLIO)
    ])
    results = engine.run_static_scenarios(_long_call_portfolio(), [scenario])
    assert results.scenario_results[0].portfolio_pnl == pytest.approx(0.0, abs=1e-6)


def test_spot_up_helps_long_call_and_down_hurts():
    from quantark.stresstest import FXStressConfig, FXStressEngine

    engine = FXStressEngine(FXStressConfig(calculate_greeks=False))
    results = engine.run_static_scenarios(
        _long_call_portfolio(),
        [_spot_scenario("up", 0.10), _spot_scenario("down", -0.10)],
    )
    assert results.get_scenario_result("up").portfolio_pnl > 0
    assert results.get_scenario_result("down").portfolio_pnl < 0
    assert results.get_worst_scenario().scenario.name == "down"


def test_vol_up_helps_long_option():
    from quantark.stresstest import FXStressConfig, FXStressEngine

    engine = FXStressEngine(FXStressConfig(calculate_greeks=True))
    scenario = Scenario(name="vol+", stresses=[
        Stress("volatility", StressType.PERCENTAGE, 0.50, StressLevel.PORTFOLIO)
    ])
    results = engine.run_static_scenarios(_long_call_portfolio(), [scenario])
    assert results.scenario_results[0].portfolio_pnl > 0
    # Greeks are computed on the stressed book.
    assert results.scenario_results[0].greeks is not None
    assert "rho_for" in results.scenario_results[0].greeks


def test_foreign_rate_hike_lowers_long_call_value():
    """Raising the foreign (EUR) rate lowers the forward, hurting a long EURUSD call."""
    from quantark.stresstest import FXStressConfig, FXStressEngine

    engine = FXStressEngine(FXStressConfig(calculate_greeks=False))
    scenario = Scenario(name="for+200bps", stresses=[
        Stress("foreign_rate", StressType.ABSOLUTE, 0.02, StressLevel.PORTFOLIO)
    ])
    results = engine.run_static_scenarios(_long_call_portfolio(), [scenario])
    assert results.scenario_results[0].portfolio_pnl < 0


def test_domestic_rate_hike_raises_long_call_value():
    """Raising the domestic (USD) rate lifts the forward, helping a long EURUSD call."""
    from quantark.stresstest import FXStressConfig, FXStressEngine

    engine = FXStressEngine(FXStressConfig(calculate_greeks=False))
    scenario = Scenario(name="dom+200bps", stresses=[
        Stress("domestic_rate", StressType.ABSOLUTE, 0.02, StressLevel.PORTFOLIO)
    ])
    results = engine.run_static_scenarios(_long_call_portfolio(), [scenario])
    assert results.scenario_results[0].portfolio_pnl > 0


def test_underlying_targeting_only_shocks_named_pair():
    from quantark.stresstest import FXStressConfig, FXStressEngine

    pf = FXPortfolio(
        portfolio_name="multi",
        pricing_environments={"EURUSD": _env(), "GBPUSD": _env(spot=1.30)},
    )
    pf.add_position(product=_call(), quantity=1.0, entry_price=0.0,
                    underlying="EURUSD", engine=GarmanKohlhagenEngine())
    gbp_fwd = FxForward(currency_pair=CurrencyPair("GBP", "USD"),
                        notional_base=1_000_000.0, contract_rate=1.30,
                        maturity_date=datetime(2027, 6, 14))
    pf.add_position(product=gbp_fwd, quantity=1.0, entry_price=0.0,
                    underlying="GBPUSD", engine=FxDeltaOneEngine())

    engine = FXStressEngine(FXStressConfig(save_detailed_results=True,
                                           calculate_greeks=False))
    scenario = Scenario(name="eur-only", stresses=[
        Stress("spot", StressType.PERCENTAGE, -0.10, StressLevel.UNDERLYING,
               target="EURUSD")
    ])
    results = engine.run_static_scenarios(pf, [scenario])
    per_pair = results.scenario_results[0].underlying_results
    # GBP forward unaffected (~0 pnl); EUR call hurt.
    assert per_pair["GBPUSD"]["total_value"] == pytest.approx(
        pf.get_greeks_by_underlying("GBPUSD")["market_value"], rel=1e-9)


def test_supports_portfolio_accepts_fx_and_rejects_non_portfolio():
    from quantark.stresstest import FXStressEngine

    engine = FXStressEngine()
    assert engine.supports_portfolio(_long_call_portfolio()) is True
    assert engine.supports_portfolio(object()) is False
