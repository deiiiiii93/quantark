"""
Tests for FX quanto vanilla and quanto digital options.
"""

import math
from datetime import datetime

import pytest
from scipy.stats import norm

from quantark.asset.fx.engine.analytical import (
    FxQuantoDigitalAnalyticalEngine,
    GarmanKohlhagenEngine,
    GarmanKohlhagenQuantoEngine,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxQuantoDigitalOption,
    FxQuantoVanillaOption,
    FxVanillaOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import (
    FxPricingEnvironment,
    FxQuantoMarketData,
    QuantoConversionOrientation,
)
from quantark.util.enum import FxPayoutCurrency, OptionType
from quantark.util.exceptions import MarketDataError, ValidationError

SPOT = 1.20
STRIKE = 1.25
R_DOM = 0.05
R_FOR = 0.03
R_SET = 0.001
VOL = 0.10
QUANTO_VOL = 0.12
CORR = -0.30
QUANTO_FX_RATE = 150.0  # e.g. USD -> JPY conversion fixed at inception
T = 1.0
NOTIONAL = 1_000_000.0
PAYOUT = 100_000.0


def make_env(correlation=CORR, with_quanto=True, **overrides):
    quanto = (
        FxQuantoMarketData(
            settlement_curve=FlatRateCurve(rate=R_SET),
            quanto_vol=QUANTO_VOL,
            correlation=correlation,
            conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC,
        )
        if with_quanto
        else None
    )
    kwargs = dict(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=R_DOM),
        foreign_curve=FlatRateCurve(rate=R_FOR),
        vol_surface=FlatVolSurface(volatility=VOL),
        quanto=quanto,
    )
    kwargs.update(overrides)
    return FxPricingEnvironment(**kwargs)


def make_vanilla(option_type=OptionType.CALL, **overrides):
    kwargs = dict(
        currency_pair=CurrencyPair("EUR", "USD"),
        strike=STRIKE,
        option_type=option_type,
        maturity=T,
        notional_foreign=NOTIONAL,
        quanto_fx_rate=QUANTO_FX_RATE,
        settlement_ccy="JPY",
    )
    kwargs.update(overrides)
    return FxQuantoVanillaOption(**kwargs)


def reference_quanto_vanilla(is_call, correlation=CORR):
    quanto_adj = -correlation * VOL * QUANTO_VOL
    fwd = SPOT * math.exp((R_DOM - R_FOR + quanto_adj) * T)
    d1 = (math.log(fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
    d2 = d1 - VOL * math.sqrt(T)
    df_set = math.exp(-R_SET * T)
    if is_call:
        unit = fwd * norm.cdf(d1) - STRIKE * norm.cdf(d2)
    else:
        unit = STRIKE * norm.cdf(-d2) - fwd * norm.cdf(-d1)
    return NOTIONAL * QUANTO_FX_RATE * df_set * unit


class TestQuantoVanilla:
    def test_call_reference_price(self):
        price = GarmanKohlhagenQuantoEngine().price(make_vanilla(), make_env())
        assert price == pytest.approx(reference_quanto_vanilla(True), rel=1e-12)

    def test_put_reference_price(self):
        price = GarmanKohlhagenQuantoEngine().price(
            make_vanilla(OptionType.PUT), make_env()
        )
        assert price == pytest.approx(reference_quanto_vanilla(False), rel=1e-12)

    def test_zero_correlation_reduces_to_rescaled_gk(self):
        quanto_price = GarmanKohlhagenQuantoEngine().price(
            make_vanilla(), make_env(correlation=0.0)
        )
        gk_price = GarmanKohlhagenEngine().price(
            FxVanillaOption(
                strike=STRIKE,
                option_type=OptionType.CALL,
                maturity=T,
                notional_foreign=NOTIONAL,
            ),
            make_env(with_quanto=False),
        )
        scale = QUANTO_FX_RATE * math.exp(-(R_SET - R_DOM) * T)
        assert quanto_price == pytest.approx(gk_price * scale, rel=1e-10)

    def test_negative_correlation_raises_call_value(self):
        engine = GarmanKohlhagenQuantoEngine()
        low = engine.price(make_vanilla(), make_env(correlation=0.3))
        high = engine.price(make_vanilla(), make_env(correlation=-0.3))
        assert high > low

    def test_requires_quanto_market_data(self):
        with pytest.raises(MarketDataError):
            GarmanKohlhagenQuantoEngine().price(
                make_vanilla(), make_env(with_quanto=False)
            )

    def test_quanto_fx_rate_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_vanilla(quanto_fx_rate=-1.0)

    def test_fdm_greeks_run_and_delta_positive(self):
        greeks = GarmanKohlhagenQuantoEngine().calculate_greeks(
            make_vanilla(), make_env()
        )
        assert greeks["delta"] > 0
        assert greeks["vega"] > 0

    def test_payoff_in_settlement_currency(self):
        opt = make_vanilla()
        assert opt.get_payoff(1.30) == pytest.approx(
            NOTIONAL * 0.05 * QUANTO_FX_RATE
        )


class TestQuantoDigital:
    def make_digital(self, option_type=OptionType.CALL, **overrides):
        kwargs = dict(
            currency_pair=CurrencyPair("EUR", "USD"),
            strike=STRIKE,
            option_type=option_type,
            maturity=T,
            payout=PAYOUT,
            quanto_fx_rate=QUANTO_FX_RATE,
            settlement_ccy="JPY",
        )
        kwargs.update(overrides)
        return FxQuantoDigitalOption(**kwargs)

    def reference_price(self, is_call, correlation=CORR):
        quanto_adj = -correlation * VOL * QUANTO_VOL
        fwd = SPOT * math.exp((R_DOM - R_FOR + quanto_adj) * T)
        d1 = (math.log(fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
        d2 = d1 - VOL * math.sqrt(T)
        df_set = math.exp(-R_SET * T)
        prob = norm.cdf(d2) if is_call else norm.cdf(-d2)
        return PAYOUT * QUANTO_FX_RATE * df_set * prob

    def test_call_reference_price(self):
        price = FxQuantoDigitalAnalyticalEngine().price(
            self.make_digital(), make_env()
        )
        assert price == pytest.approx(self.reference_price(True), rel=1e-12)

    def test_put_reference_price(self):
        price = FxQuantoDigitalAnalyticalEngine().price(
            self.make_digital(OptionType.PUT), make_env()
        )
        assert price == pytest.approx(self.reference_price(False), rel=1e-12)

    def test_call_put_parity(self):
        engine = FxQuantoDigitalAnalyticalEngine()
        env = make_env()
        call = engine.price(self.make_digital(OptionType.CALL), env)
        put = engine.price(self.make_digital(OptionType.PUT), env)
        assert call + put == pytest.approx(
            PAYOUT * QUANTO_FX_RATE * math.exp(-R_SET * T), rel=1e-12
        )

    def test_requires_quanto_market_data(self):
        with pytest.raises(MarketDataError):
            FxQuantoDigitalAnalyticalEngine().price(
                self.make_digital(), make_env(with_quanto=False)
            )

    def test_foreign_payout_rejected(self):
        with pytest.raises(ValidationError):
            self.make_digital(payout_currency=FxPayoutCurrency.FOREIGN)

    def test_payoff_in_settlement_currency(self):
        opt = self.make_digital()
        assert opt.get_payoff(1.30) == pytest.approx(PAYOUT * QUANTO_FX_RATE)
        assert opt.get_payoff(1.20) == 0.0
