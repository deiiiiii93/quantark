"""
Tests for the FX option report formatter.
"""

from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.asset.fx.report import format_fx_option_report, get_currency_symbol
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType


def make_env():
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )


def make_option():
    return FxVanillaOption(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=1.25,
        option_type=OptionType.CALL,
        maturity=1.0,
        notional_foreign=1_000_000.0,
    )


class TestCurrencySymbols:
    def test_known_symbols(self):
        assert get_currency_symbol("USD") == "$"
        assert get_currency_symbol("EUR") == "€"
        assert get_currency_symbol("GBP") == "£"

    def test_unknown_code_returns_code(self):
        assert get_currency_symbol("XYZ") == "XYZ"

    def test_case_insensitive(self):
        assert get_currency_symbol("usd") == "$"


class TestOptionReport:
    def test_report_contains_key_fields(self):
        option = make_option()
        env = make_env()
        engine = GarmanKohlhagenEngine()
        report = format_fx_option_report(option, env, engine)

        assert "EUR/USD" in report
        assert "Call" in report
        assert "1.250000" in report  # strike
        assert "1.200000" in report  # spot
        assert "Delta" in report
        assert "Vega" in report

    def test_report_without_greeks(self):
        report = format_fx_option_report(
            make_option(), make_env(), GarmanKohlhagenEngine(), include_greeks=False
        )
        assert "Delta" not in report
        assert "Option Value" in report

    def test_symbols_mode(self):
        report = format_fx_option_report(
            make_option(),
            make_env(),
            GarmanKohlhagenEngine(),
            display_mode="symbols",
        )
        assert "$" in report
        assert "€" in report

    def test_invalid_display_mode_rejected(self):
        with pytest.raises(ValueError):
            format_fx_option_report(
                make_option(),
                make_env(),
                GarmanKohlhagenEngine(),
                display_mode="emoji",
            )
