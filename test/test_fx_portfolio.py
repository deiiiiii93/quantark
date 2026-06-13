"""
Tests for the FX portfolio layer (FXPosition, FXPortfolio).

These exercise the foundation that the FX VaR / stress / dynamic-scenario /
backtest subpackages build on: a portfolio of FX positions keyed by currency
pair, each priced through its own FxPricingEnvironment and aggregating the
two-rate FX greeks (delta, gamma, vega, theta, rho_dom, rho_for).
"""
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine, GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

CORE_GREEKS = {"delta", "gamma", "vega", "theta", "rho_dom", "rho_for"}


@pytest.fixture
def eurusd_env():
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),  # USD
        foreign_curve=FlatRateCurve(rate=0.03),  # EUR
        vol_surface=FlatVolSurface(volatility=0.10),
    )


@pytest.fixture
def eurusd_call():
    return FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1_000_000.0,
    )


@pytest.fixture
def eurusd_forward():
    return FxForward(
        currency_pair=CurrencyPair("EUR", "USD"),
        notional_base=1_000_000.0,
        contract_rate=1.2100,
        maturity_date=datetime(2027, 6, 14),
    )


# --------------------------------------------------------------------------- #
# FXPosition
# --------------------------------------------------------------------------- #
def test_fx_position_market_value_scales_with_quantity(eurusd_env, eurusd_call):
    from quantark.portfolio.fx import FXPosition

    engine = GarmanKohlhagenEngine()
    pos = FXPosition(
        product=eurusd_call,
        quantity=3.0,
        entry_price=12_000.0,
        underlying="EURUSD",
        engine=engine,
        entry_timestamp=datetime(2026, 6, 13),
    )
    unit = engine.price(eurusd_call, eurusd_env)
    assert pos.get_current_price(eurusd_env) == pytest.approx(unit)
    assert pos.get_market_value(eurusd_env) == pytest.approx(unit * 3.0)


def test_fx_position_pnl_uses_entry_price(eurusd_env, eurusd_call):
    from quantark.portfolio.fx import FXPosition

    engine = GarmanKohlhagenEngine()
    entry = 10_000.0
    pos = FXPosition(
        product=eurusd_call,
        quantity=2.0,
        entry_price=entry,
        underlying="EURUSD",
        engine=engine,
        entry_timestamp=datetime(2026, 6, 13),
    )
    unit = engine.price(eurusd_call, eurusd_env)
    assert pos.get_pnl(eurusd_env) == pytest.approx((unit - entry) * 2.0)


def test_fx_position_greeks_have_two_rhos_and_scale(eurusd_env, eurusd_call):
    from quantark.portfolio.fx import FXPosition

    engine = GarmanKohlhagenEngine()
    pos = FXPosition(
        product=eurusd_call,
        quantity=2.0,
        entry_price=10_000.0,
        underlying="EURUSD",
        engine=engine,
        entry_timestamp=datetime(2026, 6, 13),
    )
    g = pos.get_greeks(eurusd_env)
    assert CORE_GREEKS <= set(g)
    # Position greeks are per-position (quantity-scaled) vs raw single-unit.
    raw = engine.calculate_greeks(eurusd_call, eurusd_env)
    assert g["delta"] == pytest.approx(raw["delta"] * 2.0)
    assert g["rho_for"] == pytest.approx(raw["rho_for"] * 2.0)


def test_fx_position_long_short_and_zero_quantity(eurusd_call):
    from quantark.portfolio.fx import FXPosition

    engine = GarmanKohlhagenEngine()
    long_pos = FXPosition(
        product=eurusd_call, quantity=1.0, entry_price=1.0,
        underlying="EURUSD", engine=engine, entry_timestamp=datetime(2026, 6, 13),
    )
    short_pos = FXPosition(
        product=eurusd_call, quantity=-1.0, entry_price=1.0,
        underlying="EURUSD", engine=engine, entry_timestamp=datetime(2026, 6, 13),
    )
    assert long_pos.is_long() and not long_pos.is_short()
    assert short_pos.is_short() and not short_pos.is_long()
    with pytest.raises(ValidationError):
        FXPosition(
            product=eurusd_call, quantity=0.0, entry_price=1.0,
            underlying="EURUSD", engine=engine,
            entry_timestamp=datetime(2026, 6, 13),
        )


# --------------------------------------------------------------------------- #
# FXPortfolio
# --------------------------------------------------------------------------- #
def test_fx_portfolio_aggregates_value_and_greeks(eurusd_env, eurusd_call):
    from quantark.portfolio.fx import FXPortfolio

    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": eurusd_env})
    pf.add_position(
        product=eurusd_call, quantity=2.0, entry_price=10_000.0,
        underlying="EURUSD", engine=GarmanKohlhagenEngine(),
    )
    expected = 2.0 * GarmanKohlhagenEngine().price(eurusd_call, eurusd_env)
    assert pf.get_portfolio_value() == pytest.approx(expected)

    g = pf.get_portfolio_risk_measures()
    assert CORE_GREEKS <= set(g)
    assert "market_value" in g
    assert g["market_value"] == pytest.approx(expected)


def test_fx_portfolio_multi_pair_aggregation(eurusd_env, eurusd_call, eurusd_forward):
    from quantark.portfolio.fx import FXPortfolio

    gbpusd_env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.30),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.045),
    )
    pf = FXPortfolio(
        portfolio_name="multi",
        pricing_environments={"EURUSD": eurusd_env, "GBPUSD": gbpusd_env},
    )
    pf.add_position(
        product=eurusd_call, quantity=1.0, entry_price=1.0,
        underlying="EURUSD", engine=GarmanKohlhagenEngine(),
    )
    pf.add_position(
        product=eurusd_forward, quantity=1.0, entry_price=0.0,
        underlying="GBPUSD", engine=FxDeltaOneEngine(),
    )
    assert len(pf) == 2
    assert set(pf.get_summary()["pairs"]) == {"EURUSD", "GBPUSD"}
    # Delta-one forward carries delta but no vega; total delta is finite.
    g = pf.get_portfolio_risk_measures()
    assert g["delta"] != 0.0


def test_fx_portfolio_rejects_unknown_pair(eurusd_env, eurusd_call):
    from quantark.portfolio.fx import FXPortfolio

    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": eurusd_env})
    with pytest.raises(ValidationError):
        pf.add_position(
            product=eurusd_call, quantity=1.0, entry_price=0.01,
            underlying="GBPUSD", engine=GarmanKohlhagenEngine(),
        )


def test_fx_types_satisfy_base_protocols(eurusd_env, eurusd_call):
    from quantark.portfolio.base import BasePortfolio, BasePosition
    from quantark.portfolio.fx import FXPortfolio, FXPosition

    pos = FXPosition(
        product=eurusd_call, quantity=1.0, entry_price=1.0,
        underlying="EURUSD", engine=GarmanKohlhagenEngine(),
        entry_timestamp=datetime(2026, 6, 13),
    )
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": eurusd_env})
    assert isinstance(pos, BasePosition)
    assert isinstance(pf, BasePortfolio)
