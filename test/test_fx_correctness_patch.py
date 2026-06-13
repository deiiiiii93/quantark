"""Regression tests for FX fixing, settlement, and quanto conventions."""

import math
from datetime import datetime

import pytest
from scipy.stats import norm

from quantark.asset.fx.engine.analytical import (
    FxDigitalOptionAnalyticalEngine,
    GarmanKohlhagenEngine,
    GarmanKohlhagenQuantoEngine,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxDigitalOption,
    FxQuantoVanillaOption,
    FxVanillaOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import (
    FxPricingEnvironment,
    FxQuantoMarketData,
    QuantoConversionOrientation,
)
from quantark.portfolio import FXPosition
from quantark.simm import FXDeltaSensitivity, FXVegaSensitivity, SIMMConfig
from quantark.util.enum import FxPayoutCurrency, OptionType


def _env(orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.2),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.1),
        quanto=FxQuantoMarketData(
            settlement_curve=FlatRateCurve(rate=0.02),
            quanto_vol=0.12,
            correlation=0.4,
            conversion_orientation=orientation,
        ),
    )


def test_delayed_foreign_digital_uses_expiry_fixing_and_delivery_discounting():
    option = FxDigitalOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.CALL,
        maturity=1.0,
        delivery=1.25,
        payout=100_000,
        payout_currency=FxPayoutCurrency.FOREIGN,
    )
    fwd = 1.2 * math.exp((0.05 - 0.03) * 1.0)
    d1 = (math.log(fwd / 1.25) + 0.5 * 0.1**2) / 0.1
    expected = 100_000 * fwd * math.exp(-0.05 * 1.25) * norm.cdf(d1)
    assert FxDigitalOptionAnalyticalEngine().price(option, _env()) == pytest.approx(
        expected
    )


def test_fixed_payoff_at_expiry_is_still_discounted_to_delivery():
    option = FxDigitalOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.1,
        option_type=OptionType.CALL,
        maturity=1e-14,
        delivery=0.25,
        payout=100_000,
    )
    expected = 100_000 * math.exp(-0.05 * 0.25)
    assert FxDigitalOptionAnalyticalEngine().price(option, _env()) == pytest.approx(
        expected
    )


def test_quanto_orientation_reverses_covariance_adjustment():
    option = FxQuantoVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.2,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1.0,
        quanto_fx_rate=1.0,
    )
    engine = GarmanKohlhagenQuantoEngine()
    settlement_per_domestic = engine.price(option, _env())
    domestic_per_settlement = engine.price(
        option, _env(QuantoConversionOrientation.DOMESTIC_PER_SETTLEMENT)
    )
    assert domestic_per_settlement > settlement_per_domestic


def test_fx_position_supplies_delta_and_vega_simm_risk_factors():
    product = FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.2,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1_000_000,
    )
    position = FXPosition(product=product, quantity=2.0, engine=GarmanKohlhagenEngine())
    sensitivities = position.get_simm_sensitivities(
        SIMMConfig(), {"EURUSD": _env()}
    )
    assert len(sensitivities.by_risk_class(FXDeltaSensitivity("T", 0).risk_class)) == 2
    assert any(isinstance(s, FXDeltaSensitivity) for s in sensitivities)
    assert any(isinstance(s, FXVegaSensitivity) for s in sensitivities)


def test_market_forward_override_greeks_respect_fixed_forward():
    env = _env()
    env.market_forward = 1.23
    option = FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.2,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1_000_000,
    )
    greeks = GarmanKohlhagenEngine().calculate_greeks(option, env)
    assert greeks["delta"] == pytest.approx(0.0, abs=1e-8)
    assert greeks["rho_for"] == pytest.approx(0.0, abs=1e-8)


def test_delayed_cash_digital_greeks_use_consistent_finite_differences():
    env = _env()
    option = FxDigitalOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.2,
        option_type=OptionType.CALL,
        maturity=1.0,
        delivery=1.25,
        payout=100_000,
    )
    engine = FxDigitalOptionAnalyticalEngine()
    greeks = engine.calculate_greeks(option, env)
    expected_delta, expected_gamma = engine._fdm_delta_gamma(
        option, env, greeks["price"]
    )
    assert greeks["delta"] == pytest.approx(expected_delta, rel=1e-12)
    assert greeks["gamma"] == pytest.approx(expected_gamma, rel=1e-12)
