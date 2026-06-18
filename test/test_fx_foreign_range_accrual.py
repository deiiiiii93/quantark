"""
Tests for the FX foreign-currency-coupon range accrual product and engine.

The decisive correctness gates are (1) a single INSTANT fixing reproduces the
existing asset-or-nothing (FOREIGN payout) digital, and (2) a fixing before
payment matches the martingale-collapse formula
``S_eff * DF_foreign(T_pay) * [N(d1_L) - N(d1_U)]`` with ``d1`` at the
fixing-date forward.
"""

import math
from datetime import datetime

import pytest
from scipy.stats import norm

from quantark.asset.fx.engine.analytical import (
    FxDigitalOptionAnalyticalEngine,
    FxForeignRangeAccrualAnalyticalEngine,
    FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxDigitalOption,
    FxForeignRangeAccrualOption,
    FxRangeAccrualConfig,
    FxRangeAccrualOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    SmileQuotes,
    VannaVolgaVolSurface,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import (
    CouponPayType,
    FxPayoutCurrency,
    FxRangeAccrualMethod,
    OptionType,
)
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError

VAL = datetime(2026, 6, 15)
SPOT, RD, RF, VOL, T = 1.20, 0.03, 0.05, 0.10, 1.0  # RF > RD so DF_for != DF_dom
NOTIONAL = 1_000_000.0
LOWER, UPPER, RATE = 1.10, 1.30, 0.04

BOTH = [FxRangeAccrualMethod.DIGITAL_COMBINATION, FxRangeAccrualMethod.CALL_PUT_SPREAD]


def env(vol_surface=None):
    return FxPricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=RD),
        foreign_curve=FlatRateCurve(rate=RF),
        vol_surface=vol_surface or FlatVolSurface(volatility=VOL),
    )


def smile_env(rr25=-0.02, bf25=0.004):
    surface = VannaVolgaVolSurface(
        FXEnv(spot=SPOT, rd=RD, rf=RF, tau=T),
        SmileQuotes(sigma_atm=VOL, rr25=rr25, bf25_2vol=bf25),
        DeltaConvention.SPOT,
    )
    return env(surface)


def make_foreign(pay_type=CouponPayType.EXPIRY, reverse=False, **kw):
    cfg = FxRangeAccrualConfig(
        upper_barrier=UPPER,
        lower_barrier=LOWER,
        accrual_rate=RATE,
        accrual_pay_type=pay_type,
        is_reverse=reverse,
    )
    base = dict(
        notional=NOTIONAL,
        range_config=cfg,
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=T,
        num_observations=12,
    )
    base.update(kw)
    return FxForeignRangeAccrualOption(**base)


# ---------------------------------------------------------------------------
# Cross-checks against the asset-or-nothing digital
# ---------------------------------------------------------------------------


class TestAssetOrNothingReduction:
    @pytest.mark.parametrize("method", BOTH)
    def test_single_instant_fixing_matches_aon_digital(self, method):
        t = 0.5
        cfg = FxRangeAccrualConfig(
            upper_barrier=UPPER,
            lower_barrier=LOWER,
            accrual_rate=1.0,
            accrual_pay_type=CouponPayType.INSTANT,
        )
        prod = FxForeignRangeAccrualOption(
            notional=1.0,
            range_config=cfg,
            currency_pair=CurrencyPair("EUR", "USD"),
            maturity=t,
            observation_times=[t],
        )
        e = env()
        ra = FxForeignRangeAccrualAnalyticalEngine(method=method).price(prod, e)

        deng = FxDigitalOptionAnalyticalEngine()

        def aon(strike):
            return deng.price(
                FxDigitalOption(
                    strike=strike,
                    option_type=OptionType.CALL,
                    payout=1.0,
                    maturity=t,
                    delivery=t,
                    payout_currency=FxPayoutCurrency.FOREIGN,
                ),
                e,
            )

        assert ra == pytest.approx(aon(LOWER) - aon(UPPER), rel=1e-5)


# ---------------------------------------------------------------------------
# Fixing-before-payment martingale collapse
# ---------------------------------------------------------------------------


class TestFixingBeforePayment:
    def test_expiry_pay_uses_payment_df_and_fixing_forward(self):
        t = 0.5  # fixing well before payment at T
        cfg = FxRangeAccrualConfig(
            upper_barrier=UPPER,
            lower_barrier=LOWER,
            accrual_rate=1.0,
            accrual_pay_type=CouponPayType.EXPIRY,
        )
        prod = FxForeignRangeAccrualOption(
            notional=1.0,
            range_config=cfg,
            currency_pair=CurrencyPair("EUR", "USD"),
            maturity=T,
            observation_times=[t],
        )
        price = FxForeignRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(prod, env())

        # PV = S_eff * DF_foreign(T) * [N(d1_L) - N(d1_U)], d1 at fixing forward F(0,t)
        fwd_t = SPOT * math.exp((RD - RF) * t)
        s = VOL * math.sqrt(t)
        d1_l = (math.log(fwd_t / LOWER) + 0.5 * VOL * VOL * t) / s
        d1_u = (math.log(fwd_t / UPPER) + 0.5 * VOL * VOL * t) / s
        expected = SPOT * math.exp(-RF * T) * (norm.cdf(d1_l) - norm.cdf(d1_u))
        assert price == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Method equivalence and smile behaviour
# ---------------------------------------------------------------------------


class TestMethodsAndSmile:
    def test_methods_agree_under_flat_vol(self):
        prod = make_foreign()
        e = env()
        dig = FxForeignRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(prod, e)
        spr = FxForeignRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(prod, e)
        assert spr == pytest.approx(dig, rel=1e-5)

    def test_methods_diverge_under_skewed_smile(self):
        prod = make_foreign()
        e = smile_env()
        dig = FxForeignRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.DIGITAL_COMBINATION
        ).price(prod, e)
        spr = FxForeignRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).price(prod, e)
        assert abs(spr - dig) > 1.0


# ---------------------------------------------------------------------------
# Accrual identities and pay timing
# ---------------------------------------------------------------------------


class TestIdentities:
    def test_normal_plus_reverse_is_full_foreign_coupon(self):
        e = env()
        engine = FxForeignRangeAccrualAnalyticalEngine()
        normal = engine.price(make_foreign(), e)
        reverse = engine.price(make_foreign(reverse=True), e)
        # Full coupon paid for certain at T, in foreign units, valued in domestic.
        full = NOTIONAL * RATE * SPOT * math.exp(-RF * T)
        assert normal + reverse == pytest.approx(full, rel=1e-10)

    def test_instant_pay_discounts_less_than_expiry(self):
        e = env()
        engine = FxForeignRangeAccrualAnalyticalEngine()
        expiry = engine.price(make_foreign(pay_type=CouponPayType.EXPIRY), e)
        instant = engine.price(make_foreign(pay_type=CouponPayType.INSTANT), e)
        assert instant > expiry


# ---------------------------------------------------------------------------
# Greeks, dispatch guards, plumbing
# ---------------------------------------------------------------------------


class TestGreeksAndDispatch:
    def test_greeks_present(self):
        greeks = FxForeignRangeAccrualAnalyticalEngine(
            method=FxRangeAccrualMethod.CALL_PUT_SPREAD
        ).calculate_greeks(make_foreign(), env())
        for key in ("price", "delta", "gamma", "vega", "theta"):
            assert key in greeks

    def test_domestic_engine_rejects_foreign_product(self):
        with pytest.raises(PricingError):
            FxRangeAccrualAnalyticalEngine().price(make_foreign(), env())

    def test_foreign_engine_rejects_plain_product(self):
        plain = FxRangeAccrualOption(
            notional=NOTIONAL,
            range_config=FxRangeAccrualConfig(upper_barrier=UPPER, lower_barrier=LOWER),
            maturity=T,
            num_observations=12,
        )
        with pytest.raises(PricingError):
            FxForeignRangeAccrualAnalyticalEngine().price(plain, env())

    def test_two_level_enum_method(self):
        engine = FxForeignRangeAccrualAnalyticalEngine(
            method=EngineType.ANALYTICAL(FxRangeAccrualMethod.CALL_PUT_SPREAD)
        )
        assert engine.method == FxRangeAccrualMethod.CALL_PUT_SPREAD
