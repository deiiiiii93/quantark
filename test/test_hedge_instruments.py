"""
Unit tests for hedge instrument specifications.
"""

from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.backtest.strategy.hedge_instruments import (
    FuturesHedgeInstrument,
    OptionHedgeInstrument,
    SpotHedgeInstrument,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

UNDERLYING = "TEST"
VALUATION = datetime(2026, 1, 2)


@pytest.fixture
def pricing_env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name=UNDERLYING),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=VALUATION,
    )


@pytest.fixture
def calculator():
    return GreeksCalculator()


class TestSpotHedgeInstrument:
    def test_product_and_greeks(self, pricing_env, calculator):
        spec = SpotHedgeInstrument()
        product = spec.create_product(UNDERLYING, pricing_env, VALUATION)
        engine = spec.create_engine()

        assert isinstance(product, SpotInstrument)
        assert spec.unit_price(product, engine, pricing_env) == pytest.approx(100.0)

        greeks = spec.unit_greeks(product, engine, pricing_env, calculator)
        assert greeks["delta"] == pytest.approx(1.0, abs=1e-6)
        assert greeks["gamma"] == pytest.approx(0.0, abs=1e-6)
        assert greeks["vega"] == pytest.approx(0.0, abs=1e-6)

    def test_never_rolls(self, pricing_env):
        spec = SpotHedgeInstrument()
        product = spec.create_product(UNDERLYING, pricing_env, VALUATION)
        assert not spec.requires_roll(product, pricing_env)


class TestFuturesHedgeInstrument:
    def test_product_creation(self, pricing_env):
        spec = FuturesHedgeInstrument(maturity=0.25, multiplier=10.0)
        product = spec.create_product(UNDERLYING, pricing_env, VALUATION)
        assert isinstance(product, Futures)
        assert product.multiplier == 10.0

    def test_invalid_parameters(self):
        with pytest.raises(ValidationError):
            FuturesHedgeInstrument(maturity=0.0)
        with pytest.raises(ValidationError):
            FuturesHedgeInstrument(maturity=0.25, multiplier=0.0)


class TestOptionHedgeInstrument:
    def test_creates_aging_atm_option(self, pricing_env, calculator):
        spec = OptionHedgeInstrument(tenor=0.25, moneyness=1.0, name="gamma_option")
        product = spec.create_product(UNDERLYING, pricing_env, VALUATION)

        assert isinstance(product, EuropeanVanillaOption)
        assert product.strike == pytest.approx(100.0)  # ATM at creation
        assert product.exercise_date is not None
        # Date-based expiry: ages with the backtest clock
        assert product.get_maturity(pricing_env) == pytest.approx(0.25, abs=0.01)

        greeks = spec.unit_greeks(
            product, spec.create_engine(), pricing_env, calculator
        )
        assert greeks["gamma"] > 0.0
        assert greeks["vega"] > 0.0
        assert 0.0 < greeks["delta"] < 1.0

    def test_moneyness_sets_strike(self, pricing_env):
        spec = OptionHedgeInstrument(tenor=0.25, moneyness=1.05)
        product = spec.create_product(UNDERLYING, pricing_env, VALUATION)
        assert product.strike == pytest.approx(105.0)

    def test_rolls_near_expiry(self, pricing_env):
        spec = OptionHedgeInstrument(tenor=0.25, min_time_to_expiry=7.0 / 365.0)
        product = spec.create_product(UNDERLYING, pricing_env, VALUATION)

        assert not spec.requires_roll(product, pricing_env)

        pricing_env.valuation_date = product.exercise_date - timedelta(days=3)
        assert spec.requires_roll(product, pricing_env)

        # On/after expiry must also roll (get_maturity would raise)
        pricing_env.valuation_date = product.exercise_date
        assert spec.requires_roll(product, pricing_env)

    def test_restrike_band(self, pricing_env):
        spec = OptionHedgeInstrument(tenor=0.25, moneyness=1.0, restrike_band=0.10)
        product = spec.create_product(UNDERLYING, pricing_env, VALUATION)

        assert not spec.requires_roll(product, pricing_env)

        # Spot rallies 25%: strike/spot = 100/125 = 0.8, drift 0.2 > 0.10
        pricing_env.spot_quote = SpotQuote(spot=125.0, asset_name=UNDERLYING)
        assert spec.requires_roll(product, pricing_env)

    def test_invalid_parameters(self):
        with pytest.raises(ValidationError):
            OptionHedgeInstrument(tenor=0.0)
        with pytest.raises(ValidationError):
            OptionHedgeInstrument(tenor=0.25, moneyness=0.0)
        with pytest.raises(ValidationError):
            OptionHedgeInstrument(tenor=0.25, min_time_to_expiry=0.5)
        with pytest.raises(ValidationError):
            OptionHedgeInstrument(tenor=0.25, restrike_band=0.0)

    def test_put_option_type(self, pricing_env, calculator):
        spec = OptionHedgeInstrument(tenor=0.25, option_type=OptionType.PUT)
        product = spec.create_product(UNDERLYING, pricing_env, VALUATION)
        greeks = spec.unit_greeks(
            product, spec.create_engine(), pricing_env, calculator
        )
        assert -1.0 < greeks["delta"] < 0.0
        assert greeks["gamma"] > 0.0
