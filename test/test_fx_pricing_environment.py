"""
Tests for FxPricingEnvironment and CurrencyPair.
"""

import math
from datetime import datetime

import pytest

from quantark.asset.fx.product import CurrencyPair
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment, FxQuantoMarketData
from quantark.util.exceptions import MarketDataError


VALUATION_DATE = datetime(2026, 6, 12)


def make_env(**overrides):
    """Build a standard EUR/USD-style FX pricing environment."""
    kwargs = dict(
        valuation_date=VALUATION_DATE,
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )
    kwargs.update(overrides)
    return FxPricingEnvironment(**kwargs)


class TestCurrencyPair:
    def test_defaults(self):
        pair = CurrencyPair()
        assert pair.base_ccy == "FOR"
        assert pair.quote_ccy == "DOM"

    def test_aliases(self):
        pair = CurrencyPair(base_ccy="EUR", quote_ccy="USD")
        assert pair.foreign == "EUR"
        assert pair.domestic == "USD"

    def test_str(self):
        pair = CurrencyPair(base_ccy="EUR", quote_ccy="USD")
        assert str(pair) == "EUR/USD"

    def test_codes_uppercased(self):
        pair = CurrencyPair(base_ccy="eur", quote_ccy="usd")
        assert pair.base_ccy == "EUR"
        assert pair.quote_ccy == "USD"


class TestFxPricingEnvironmentValidation:
    def test_requires_domestic_curve(self):
        with pytest.raises(MarketDataError):
            make_env(domestic_curve=None)

    def test_requires_foreign_curve(self):
        with pytest.raises(MarketDataError):
            make_env(foreign_curve=None)

    def test_requires_spot_quote(self):
        with pytest.raises(MarketDataError):
            make_env(spot_quote=None)

    def test_requires_valuation_date(self):
        with pytest.raises(MarketDataError):
            make_env(valuation_date=None)

    def test_vol_surface_optional_but_guarded(self):
        env = make_env(vol_surface=None)
        with pytest.raises(MarketDataError):
            env.get_vol(1.20, 1.0)


class TestFxPricingEnvironmentAccessors:
    def test_spot(self):
        env = make_env()
        assert env.spot == pytest.approx(1.20)

    def test_rates_and_discount_factors(self):
        env = make_env()
        assert env.get_domestic_rate(1.0) == pytest.approx(0.05)
        assert env.get_foreign_rate(1.0) == pytest.approx(0.03)
        assert env.get_domestic_df(2.0) == pytest.approx(math.exp(-0.05 * 2.0))
        assert env.get_foreign_df(2.0) == pytest.approx(math.exp(-0.03 * 2.0))

    def test_get_vol(self):
        env = make_env()
        assert env.get_vol(1.25, 0.5) == pytest.approx(0.10)

    def test_forward_interest_rate_parity(self):
        env = make_env()
        t = 1.5
        expected = 1.20 * math.exp((0.05 - 0.03) * t)
        assert env.get_forward(t) == pytest.approx(expected)
        # Equivalent discount-factor form
        assert env.get_forward(t) == pytest.approx(
            1.20 * env.get_foreign_df(t) / env.get_domestic_df(t)
        )

    def test_market_forward_override(self):
        env = make_env(market_forward=1.2345)
        assert env.get_forward(1.0) == pytest.approx(1.2345)

    def test_effective_spot_no_lag(self):
        env = make_env()
        assert env.effective_spot() == pytest.approx(1.20)

    def test_effective_spot_with_spot_days(self):
        env = make_env(spot_days=2)
        expected = 1.20 * math.exp((0.05 - 0.03) * 2 / 365)
        assert env.effective_spot() == pytest.approx(expected)


class TestQuantoMarketData:
    def test_quanto_data_accessible(self):
        quanto = FxQuantoMarketData(
            settlement_curve=FlatRateCurve(rate=0.001),
            quanto_vol=0.12,
            correlation=-0.3,
        )
        env = make_env(quanto=quanto)
        assert env.quanto.correlation == pytest.approx(-0.3)
        assert env.quanto.quanto_vol == pytest.approx(0.12)
        assert env.quanto.settlement_curve.get_rate(1.0) == pytest.approx(0.001)

    def test_quanto_correlation_validated(self):
        with pytest.raises(MarketDataError):
            FxQuantoMarketData(
                settlement_curve=FlatRateCurve(rate=0.001),
                quanto_vol=0.12,
                correlation=1.5,
            )

    def test_quanto_default_none(self):
        env = make_env()
        assert env.quanto is None
