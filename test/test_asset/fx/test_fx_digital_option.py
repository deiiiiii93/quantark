"""
Tests for FxDigitalOption priced with the analytical digital engine.
"""

import math
from datetime import datetime

import pytest
from scipy.stats import norm

from quantark.asset.fx.engine.analytical import FxDigitalOptionAnalyticalEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxDigitalOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxPayoutCurrency, OptionType
from quantark.util.exceptions import ValidationError

SPOT = 1.20
STRIKE = 1.25
R_DOM = 0.05
R_FOR = 0.03
VOL = 0.10
T = 1.0
PAYOUT = 100_000.0


def make_env(**overrides):
    kwargs = dict(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=R_DOM),
        foreign_curve=FlatRateCurve(rate=R_FOR),
        vol_surface=FlatVolSurface(volatility=VOL),
    )
    kwargs.update(overrides)
    return FxPricingEnvironment(**kwargs)


def make_option(option_type=OptionType.CALL, **overrides):
    kwargs = dict(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=STRIKE,
        option_type=option_type,
        maturity=T,
        payout=PAYOUT,
    )
    kwargs.update(overrides)
    return FxDigitalOption(**kwargs)


def d1_d2():
    fwd = SPOT * math.exp((R_DOM - R_FOR) * T)
    d1 = (math.log(fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
    return d1, d1 - VOL * math.sqrt(T)


class TestCashOrNothing:
    def test_call_price(self):
        _, d2 = d1_d2()
        expected = PAYOUT * math.exp(-R_DOM * T) * norm.cdf(d2)
        price = FxDigitalOptionAnalyticalEngine().price(make_option(), make_env())
        assert price == pytest.approx(expected, rel=1e-12)

    def test_put_price(self):
        _, d2 = d1_d2()
        expected = PAYOUT * math.exp(-R_DOM * T) * norm.cdf(-d2)
        price = FxDigitalOptionAnalyticalEngine().price(
            make_option(OptionType.PUT), make_env()
        )
        assert price == pytest.approx(expected, rel=1e-12)

    def test_call_put_parity(self):
        engine = FxDigitalOptionAnalyticalEngine()
        env = make_env()
        call = engine.price(make_option(OptionType.CALL), env)
        put = engine.price(make_option(OptionType.PUT), env)
        assert call + put == pytest.approx(
            PAYOUT * math.exp(-R_DOM * T), rel=1e-12
        )


class TestAssetOrNothing:
    def test_call_price(self):
        d1, _ = d1_d2()
        expected = PAYOUT * SPOT * math.exp(-R_FOR * T) * norm.cdf(d1)
        price = FxDigitalOptionAnalyticalEngine().price(
            make_option(payout_currency=FxPayoutCurrency.FOREIGN), make_env()
        )
        assert price == pytest.approx(expected, rel=1e-12)

    def test_call_put_parity(self):
        engine = FxDigitalOptionAnalyticalEngine()
        env = make_env()
        call = engine.price(
            make_option(OptionType.CALL, payout_currency=FxPayoutCurrency.FOREIGN),
            env,
        )
        put = engine.price(
            make_option(OptionType.PUT, payout_currency=FxPayoutCurrency.FOREIGN),
            env,
        )
        assert call + put == pytest.approx(
            PAYOUT * SPOT * math.exp(-R_FOR * T), rel=1e-12
        )


class TestDigitalGreeks:
    @pytest.fixture
    def greeks(self):
        return FxDigitalOptionAnalyticalEngine().calculate_greeks(
            make_option(), make_env()
        )

    def test_delta(self, greeks):
        _, d2 = d1_d2()
        expected = (
            PAYOUT
            * math.exp(-R_DOM * T)
            * norm.pdf(d2)
            / (SPOT * VOL * math.sqrt(T))
        )
        assert greeks["delta"] == pytest.approx(expected, rel=1e-10)

    def test_put_delta_negative(self):
        greeks = FxDigitalOptionAnalyticalEngine().calculate_greeks(
            make_option(OptionType.PUT), make_env()
        )
        assert greeks["delta"] < 0

    def test_gamma(self, greeks):
        d1, d2 = d1_d2()
        expected = (
            -PAYOUT
            * math.exp(-R_DOM * T)
            * norm.pdf(d2)
            * d1
            / (SPOT**2 * VOL**2 * T)
        )
        assert greeks["gamma"] == pytest.approx(expected, rel=1e-10)

    def test_vega(self, greeks):
        d1, d2 = d1_d2()
        expected = -PAYOUT * math.exp(-R_DOM * T) * norm.pdf(d2) * d1 / VOL * 0.01
        assert greeks["vega"] == pytest.approx(expected, rel=1e-10)

    def test_otm_digital_call_has_positive_vega(self):
        # OTM digital call (F < K => d1 < 0 region check): with our params
        # F = 1.224 < 1.25 so vega should be positive (more vol => more
        # chance of finishing ITM).
        greeks = FxDigitalOptionAnalyticalEngine().calculate_greeks(
            make_option(), make_env()
        )
        assert greeks["vega"] > 0


class TestPayoff:
    def test_cash_payoff(self):
        opt = make_option()
        assert opt.get_payoff(1.30) == pytest.approx(PAYOUT)
        assert opt.get_payoff(1.20) == 0.0

    def test_asset_payoff(self):
        opt = make_option(payout_currency=FxPayoutCurrency.FOREIGN)
        assert opt.get_payoff(1.30) == pytest.approx(PAYOUT * 1.30)
        assert opt.get_payoff(1.20) == 0.0

    def test_payout_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_option(payout=-1.0)
