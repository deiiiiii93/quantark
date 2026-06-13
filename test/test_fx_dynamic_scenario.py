"""
Tests for FX dynamic (multi-day) scenario analysis.

Walks an FX book day-by-day through a DayPath, applying spot / vol / two-rate
changes and tracking value, P&L and FX greeks each day.
"""
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType


def _portfolio():
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": env})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.25,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=1_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    return pf


def test_flat_path_has_zero_pnl():
    from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
    from quantark.dynamicscenario import FXPathLibrary

    path = FXPathLibrary.spot_trend(days=4, daily_pct=0.0)
    results = FXDynamicScenarioEngine(FXDynamicScenarioConfig()).run(_portfolio(), path)
    assert results.num_days == 4
    assert results.total_pnl == pytest.approx(0.0, abs=1e-6)


def test_spot_rally_helps_long_call_and_tracks_greeks():
    from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
    from quantark.dynamicscenario import FXPathLibrary

    path = FXPathLibrary.spot_trend(days=5, daily_pct=0.01)
    results = FXDynamicScenarioEngine(
        FXDynamicScenarioConfig(calculate_greeks=True)
    ).run(_portfolio(), path)

    assert results.total_pnl > 0
    pnl_df = results.get_pnl_evolution()
    assert len(pnl_df) == 5
    greeks_df = results.get_greeks_evolution()
    # Two-rate FX greeks are carried through the per-day greeks dict.
    assert "rho_dom" in greeks_df.columns
    assert "rho_for" in greeks_df.columns


def test_foreign_rate_rise_hurts_long_call():
    from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
    from quantark.dynamicscenario import FXPathLibrary

    path = FXPathLibrary.rate_divergence(days=5, dom_bps_per_day=0.0,
                                         for_bps_per_day=10.0)
    results = FXDynamicScenarioEngine(FXDynamicScenarioConfig()).run(_portfolio(), path)
    assert results.total_pnl < 0


def test_factory_selects_fx_engine():
    from quantark.dynamicscenario import get_engine_for_portfolio
    from quantark.dynamicscenario.fx.engine import FXDynamicScenarioEngine as FXEng

    engine = get_engine_for_portfolio(_portfolio())
    assert isinstance(engine, FXEng)


def test_path_library_builds_valid_paths():
    from quantark.dynamicscenario import FXPathLibrary

    assert FXPathLibrary.spot_trend(days=3, daily_pct=0.02).num_days == 3
    assert FXPathLibrary.carry_unwind(days=4).num_days == 4
    assert FXPathLibrary.vol_spike_decay(days=6).num_days == 6
    assert FXPathLibrary.rate_divergence(days=5).num_days == 5
