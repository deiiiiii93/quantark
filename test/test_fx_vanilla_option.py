"""
Tests for FxVanillaOption priced with the Garman-Kohlhagen engine.
"""

import math
from datetime import datetime

import pytest
from scipy.stats import norm

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import PricingError, ValidationError

SPOT = 1.20
STRIKE = 1.25
R_DOM = 0.05
R_FOR = 0.03
VOL = 0.10
T = 1.0
NOTIONAL_FOR = 1_000_000.0


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
        notional_foreign=NOTIONAL_FOR,
    )
    kwargs.update(overrides)
    return FxVanillaOption(**kwargs)


def gk_reference_price(is_call, spot=SPOT, strike=STRIKE, t=T, t_del=None):
    """Independent GK reference implementation."""
    t_del = t if t_del is None else t_del
    fwd = spot * math.exp((R_DOM - R_FOR) * t)
    d1 = (math.log(fwd / strike) + VOL**2 / 2 * t) / (VOL * math.sqrt(t))
    d2 = d1 - VOL * math.sqrt(t)
    df = math.exp(-R_DOM * t_del)
    if is_call:
        return NOTIONAL_FOR * (fwd * norm.cdf(d1) - strike * norm.cdf(d2)) * df
    return NOTIONAL_FOR * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1)) * df


class TestGarmanKohlhagenPrice:
    def test_call_price_reference(self):
        engine = GarmanKohlhagenEngine()
        price = engine.price(make_option(OptionType.CALL), make_env())
        assert price == pytest.approx(gk_reference_price(True), rel=1e-12)

    def test_put_price_reference(self):
        engine = GarmanKohlhagenEngine()
        price = engine.price(make_option(OptionType.PUT), make_env())
        assert price == pytest.approx(gk_reference_price(False), rel=1e-12)

    def test_put_call_parity(self):
        engine = GarmanKohlhagenEngine()
        env = make_env()
        call = engine.price(make_option(OptionType.CALL), env)
        put = engine.price(make_option(OptionType.PUT), env)
        fwd = SPOT * math.exp((R_DOM - R_FOR) * T)
        parity = NOTIONAL_FOR * (fwd - STRIKE) * math.exp(-R_DOM * T)
        assert call - put == pytest.approx(parity, rel=1e-10)

    def test_matches_equity_bsm_with_foreign_rate_as_dividend(self):
        from quantark.asset.equity.engine.analytical import BlackScholesEngine
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from quantark.param import ContinuousDividendYield
        from quantark.priceenv import PricingEnvironment

        eq_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=R_DOM),
            valuation_date=datetime(2026, 6, 12),
            spot_quote=SpotQuote(spot=SPOT),
            vol_surface=FlatVolSurface(volatility=VOL),
            div_yield=ContinuousDividendYield(div_yield=R_FOR),
        )
        eq_price = BlackScholesEngine().price(
            EuropeanVanillaOption(
                strike=STRIKE, option_type=OptionType.CALL, maturity=T
            ),
            eq_env,
        )
        fx_price = GarmanKohlhagenEngine().price(
            make_option(OptionType.CALL, notional_foreign=1.0), make_env()
        )
        assert fx_price == pytest.approx(eq_price, rel=1e-10)

    def test_delivery_lag_adjustment(self):
        engine = GarmanKohlhagenEngine()
        t_del = T + 2 / 365
        price = engine.price(make_option(delivery=t_del), make_env())
        assert price == pytest.approx(
            gk_reference_price(True, t_del=t_del), rel=1e-12
        )

    def test_delivery_lag_greeks_use_consistent_finite_differences(self):
        engine = GarmanKohlhagenEngine()
        option = make_option(delivery=T + 0.25)
        env = make_env()
        greeks = engine.calculate_greeks(option, env)
        expected_delta, expected_gamma = engine._fdm_delta_gamma(
            option, env, greeks["price"]
        )
        assert greeks["delta"] == pytest.approx(expected_delta, rel=1e-12)
        assert greeks["gamma"] == pytest.approx(expected_gamma, rel=1e-12)

    def test_market_forward_override(self):
        engine = GarmanKohlhagenEngine()
        market_fwd = 1.23
        price = engine.price(make_option(), make_env(market_forward=market_fwd))
        d1 = (math.log(market_fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
        d2 = d1 - VOL * math.sqrt(T)
        expected = (
            NOTIONAL_FOR
            * (market_fwd * norm.cdf(d1) - STRIKE * norm.cdf(d2))
            * math.exp(-R_DOM * T)
        )
        assert price == pytest.approx(expected, rel=1e-12)

    def test_participation_rate_scales_price(self):
        engine = GarmanKohlhagenEngine()
        base = engine.price(make_option(), make_env())
        half = engine.price(make_option(participation_rate=0.5), make_env())
        assert half == pytest.approx(0.5 * base, rel=1e-12)

    def test_annualization_opt_in(self):
        engine = GarmanKohlhagenEngine()
        env = make_env()
        base = engine.price(make_option(), env)
        annualized = engine.price(
            make_option(is_annualized=True, tenor_days=182.5), env
        )
        assert annualized == pytest.approx(base * 182.5 / 365, rel=1e-12)

    def test_expired_option_returns_payoff(self):
        engine = GarmanKohlhagenEngine()
        option = make_option(maturity=1e-14, strike=1.10)
        price = engine.price(option, make_env())
        assert price == pytest.approx(NOTIONAL_FOR * (SPOT - 1.10), rel=1e-8)

    def test_rejects_wrong_product(self):
        engine = GarmanKohlhagenEngine()

        class _NotAnFxOption:
            pass

        with pytest.raises(PricingError):
            engine.price(_NotAnFxOption(), make_env())


class TestNotionalHandling:
    def test_notional_foreign_only(self):
        opt = make_option()
        assert opt.notional == pytest.approx(NOTIONAL_FOR)

    def test_notional_domestic_requires_initial_spot(self):
        with pytest.raises(ValidationError):
            FxVanillaOption(
                strike=STRIKE,
                option_type=OptionType.CALL,
                maturity=T,
                notional_domestic=1_200_000.0,
            )

    def test_notional_domestic_with_initial_spot(self):
        opt = FxVanillaOption(
            strike=STRIKE,
            option_type=OptionType.CALL,
            maturity=T,
            notional_domestic=1_200_000.0,
            initial_spot=1.20,
        )
        assert opt.notional == pytest.approx(1_000_000.0)

    def test_both_notionals_consistency_enforced(self):
        with pytest.raises(ValidationError):
            FxVanillaOption(
                strike=STRIKE,
                option_type=OptionType.CALL,
                maturity=T,
                notional_domestic=1_200_000.0,
                notional_foreign=1_000_000.0,
                initial_spot=1.30,
            )

    def test_both_notionals_imply_initial_spot(self):
        opt = FxVanillaOption(
            strike=STRIKE,
            option_type=OptionType.CALL,
            maturity=T,
            notional_domestic=1_200_000.0,
            notional_foreign=1_000_000.0,
        )
        assert opt.initial_spot == pytest.approx(1.20)

    def test_no_notional_rejected(self):
        with pytest.raises(ValidationError):
            FxVanillaOption(strike=STRIKE, option_type=OptionType.CALL, maturity=T)


class TestAnalyticalGreeks:
    @pytest.fixture
    def greeks(self):
        engine = GarmanKohlhagenEngine()
        return engine.calculate_greeks(make_option(), make_env())

    def test_delta(self, greeks):
        fwd = SPOT * math.exp((R_DOM - R_FOR) * T)
        d1 = (math.log(fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
        expected = NOTIONAL_FOR * math.exp(-R_FOR * T) * norm.cdf(d1)
        assert greeks["delta"] == pytest.approx(expected, rel=1e-10)

    def test_fwd_delta_relationship(self, greeks):
        assert greeks["fwd_delta"] == pytest.approx(
            greeks["delta"] * math.exp(R_FOR * T), rel=1e-10
        )

    def test_gamma(self, greeks):
        fwd = SPOT * math.exp((R_DOM - R_FOR) * T)
        d1 = (math.log(fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
        expected = (
            NOTIONAL_FOR
            * math.exp(-R_FOR * T)
            * norm.pdf(d1)
            / (SPOT * VOL * math.sqrt(T))
        )
        assert greeks["gamma"] == pytest.approx(expected, rel=1e-10)

    def test_vega_per_one_percent(self, greeks):
        fwd = SPOT * math.exp((R_DOM - R_FOR) * T)
        d1 = (math.log(fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
        expected = (
            NOTIONAL_FOR
            * SPOT
            * math.exp(-R_FOR * T)
            * norm.pdf(d1)
            * math.sqrt(T)
            * 0.01
        )
        assert greeks["vega"] == pytest.approx(expected, rel=1e-10)

    def test_rho_dom(self, greeks):
        fwd = SPOT * math.exp((R_DOM - R_FOR) * T)
        d1 = (math.log(fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
        d2 = d1 - VOL * math.sqrt(T)
        expected = (
            NOTIONAL_FOR * STRIKE * math.exp(-R_DOM * T) * T * norm.cdf(d2) / 100.0
        )
        assert greeks["rho_dom"] == pytest.approx(expected, rel=1e-10)

    def test_rho_for(self, greeks):
        fwd = SPOT * math.exp((R_DOM - R_FOR) * T)
        d1 = (math.log(fwd / STRIKE) + VOL**2 / 2 * T) / (VOL * math.sqrt(T))
        expected = (
            -NOTIONAL_FOR * SPOT * math.exp(-R_FOR * T) * T * norm.cdf(d1) / 100.0
        )
        assert greeks["rho_for"] == pytest.approx(expected, rel=1e-10)

    def test_theta_negative_for_otm_call(self, greeks):
        assert greeks["theta"] < 0

    def test_delta_premium_foreign_premium(self):
        engine = GarmanKohlhagenEngine()
        premium = 25_000.0
        greeks = engine.calculate_greeks(
            make_option(premium_currency="FOR", premium_amount=premium), make_env()
        )
        assert greeks["delta_premium"] == pytest.approx(
            greeks["delta"] + premium / SPOT, rel=1e-10
        )
        fwd = SPOT * math.exp((R_DOM - R_FOR) * T)
        assert greeks["fwd_delta_premium"] == pytest.approx(
            greeks["fwd_delta"] + premium / fwd, rel=1e-10
        )

    def test_domestic_premium_no_adjustment(self):
        engine = GarmanKohlhagenEngine()
        greeks = engine.calculate_greeks(
            make_option(premium_currency="DOM", premium_amount=25_000.0), make_env()
        )
        assert greeks["delta_premium"] == pytest.approx(greeks["delta"], rel=1e-12)


class TestPayoff:
    def test_call_payoff(self):
        opt = make_option(OptionType.CALL)
        assert opt.get_payoff(1.30) == pytest.approx(NOTIONAL_FOR * 0.05)
        assert opt.get_payoff(1.20) == 0.0

    def test_put_payoff(self):
        opt = make_option(OptionType.PUT)
        assert opt.get_payoff(1.20) == pytest.approx(NOTIONAL_FOR * 0.05)
        assert opt.get_payoff(1.30) == 0.0

    def test_strike_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_option(strike=-1.0)
