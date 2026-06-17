"""Tests for the FX Target Redemption Note (digital-coupon TARN) MC engine.

Correctness gates:
  - a sigma->0 deterministic limit, where every path collapses to the forward
    path so the coupon / truncation / redemption / principal logic is checkable
    in closed form (covers binding target, early redemption stopping coupons,
    principal timing, band vs one-sided, single period);
  - stochastic closed-form strips: target=inf, include_principal=False -> a
    strip of FX digitals (FxDigitalOptionAnalyticalEngine); include_principal=True
    -> that strip plus notional * DF(T_final);
  - engine mechanics (SE semantics, product guard, Greeks).
"""

import math
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import FxDigitalOptionAnalyticalEngine
from quantark.asset.fx.engine.mc import (
    FxTargetRedemptionNoteMCEngine,
    FxTarnMCResult,
)
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxDigitalOption,
    FxTargetRedemptionNote,
    FxVanillaOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxPayoutCurrency, OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError

_VAL = datetime(2026, 6, 15)
_PAIR = CurrencyPair("EUR", "USD")


def _env(vol=0.10, spot=1.20, rd=0.05, rf=0.03):
    return FxPricingEnvironment(
        valuation_date=_VAL, spot_quote=SpotQuote(spot=spot),
        domestic_curve=FlatRateCurve(rate=rd),
        foreign_curve=FlatRateCurve(rate=rf),
        vol_surface=FlatVolSurface(volatility=vol),
    )


def _note(**kw):
    base = dict(fixing_times=[0.25, 0.5, 0.75, 1.0], coupon_rate=0.08,
               notional=1.0, strike=1.20, currency_pair=_PAIR)
    base.update(kw)
    return FxTargetRedemptionNote(**base)


def _mc(**kw):
    base = dict(num_paths=200_000, seed=11)
    base.update(kw)
    return FxMCParams(**base)


# ---------------------------------------------------------------------------
# Deterministic forward-path reference (sigma -> 0 limit)
# ---------------------------------------------------------------------------


def _det_pv(forwards, df_pay, opt):
    acc, pv, alive = 0.0, 0.0, True
    for i, (F, df) in enumerate(zip(forwards, df_pay)):
        if not alive:
            break
        coupon = opt.full_coupon(i) if opt.coupon_condition(F) else 0.0
        c_eff = min(coupon, opt.target - acc)
        pv += c_eff * df
        acc += c_eff
        hit = acc >= opt.target - 1e-12
        if opt.include_principal and hit:
            pv += opt.notional * df
        if hit:
            alive = False
    if opt.include_principal and alive:
        pv += opt.notional * df_pay[-1]
    return pv


def _forwards_and_df(env, opt):
    fwd = [env.get_forward(t) for t in opt.fixing_times]
    df = [env.get_domestic_df(t) for t in opt.pay_times]
    return fwd, df


class TestDeterministicLimit:
    def _check(self, opt, num_paths=20_000):
        env = _env(vol=1e-7)
        fwd, df = _forwards_and_df(env, opt)
        ref = _det_pv(fwd, df, opt)
        mc = FxTargetRedemptionNoteMCEngine(params=_mc(num_paths=num_paths)).price(opt, env)
        assert mc == pytest.approx(ref, abs=1e-6, rel=1e-6)

    def test_unbounded_no_principal(self):
        self._check(_note(target=math.inf, include_principal=False))

    def test_unbounded_with_principal(self):
        self._check(_note(target=math.inf, include_principal=True))

    def test_binding_target_truncates_and_redeems(self):
        # Forwards above strike so every period would pay; small target binds.
        self._check(_note(strike=1.10, coupon_rate=0.10, target=0.03))

    def test_binding_target_no_principal(self):
        self._check(_note(strike=1.10, coupon_rate=0.10, target=0.03,
                          include_principal=False))

    def test_below_strike_condition(self):
        # Forwards > strike here, so S<=strike never holds -> no coupons,
        # principal repaid at final maturity.
        self._check(_note(strike=1.10, is_above=False, target=math.inf))

    def test_band_condition(self):
        self._check(_note(strike=None, lower=1.15, upper=1.30, target=math.inf))

    def test_single_period(self):
        self._check(_note(fixing_times=[1.0], target=math.inf))


# ---------------------------------------------------------------------------
# Stochastic digital-strip gates
# ---------------------------------------------------------------------------


def _digital_strip(env, opt):
    eng = FxDigitalOptionAnalyticalEngine()
    total = 0.0
    for i, (tf, tp) in enumerate(zip(opt.fixing_times, opt.pay_times)):
        otype = OptionType.CALL if opt.is_above else OptionType.PUT
        dig = FxDigitalOption(
            strike=opt.strike, option_type=otype, payout=opt.full_coupon(i),
            maturity=tf, delivery=tp, payout_currency=FxPayoutCurrency.DOMESTIC,
            currency_pair=_PAIR,
        )
        total += eng.price(dig, env)
    return total


class TestDigitalStrip:
    def test_unbounded_no_principal_is_digital_strip(self):
        env = _env(vol=0.12)
        opt = _note(target=math.inf, include_principal=False)
        strip = _digital_strip(env, opt)
        eng = FxTargetRedemptionNoteMCEngine(params=_mc())
        mc = eng.price(opt, env)
        se = eng.get_last_result().std_error
        assert se is not None and abs(mc - strip) <= 4.0 * se

    def test_unbounded_with_principal_adds_discounted_notional(self):
        env = _env(vol=0.12)
        opt = _note(target=math.inf, include_principal=True)
        strip = _digital_strip(env, opt)
        principal = opt.notional * env.get_domestic_df(opt.pay_times[-1])
        eng = FxTargetRedemptionNoteMCEngine(params=_mc())
        mc = eng.price(opt, env)
        se = eng.get_last_result().std_error
        assert se is not None and abs(mc - (strip + principal)) <= 4.0 * se


# ---------------------------------------------------------------------------
# Redemption behaviour
# ---------------------------------------------------------------------------


class TestRedemption:
    def test_principal_inclusion_raises_pv(self):
        env = _env(vol=0.12)
        with_p = FxTargetRedemptionNoteMCEngine(params=_mc()).price(
            _note(target=math.inf, include_principal=True), env)
        without_p = FxTargetRedemptionNoteMCEngine(params=_mc()).price(
            _note(target=math.inf, include_principal=False), env)
        assert with_p > without_p

    def test_tiny_target_redeems_early(self):
        env = _env(vol=0.15)
        opt = _note(strike=1.05, coupon_rate=0.20, target=0.01)
        eng = FxTargetRedemptionNoteMCEngine(params=_mc(num_paths=80_000))
        eng.price(opt, env)
        assert eng.get_last_result().mean_redemption_fraction > 0.9


# ---------------------------------------------------------------------------
# Product validation
# ---------------------------------------------------------------------------


class TestProduct:
    def test_rejects_strike_and_band(self):
        with pytest.raises(ValidationError, match="not both"):
            _note(strike=1.20, lower=1.1, upper=1.3)

    def test_rejects_no_condition(self):
        with pytest.raises(ValidationError, match="condition"):
            _note(strike=None)

    def test_rejects_inverted_band(self):
        with pytest.raises(ValidationError, match="lower < upper"):
            _note(strike=None, lower=1.3, upper=1.1)

    def test_rejects_nonpositive_target(self):
        with pytest.raises(ValidationError, match="target"):
            _note(target=0.0)

    def test_default_accruals_are_period_lengths(self):
        opt = _note(fixing_times=[0.5, 1.0, 1.5])
        assert opt.accrual_factors == pytest.approx([0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Engine mechanics
# ---------------------------------------------------------------------------


class TestEngine:
    def test_result_fields(self):
        env = _env()
        eng = FxTargetRedemptionNoteMCEngine(params=_mc(num_paths=50_000))
        eng.price(_note(), env)
        res = eng.get_last_result()
        assert isinstance(res, FxTarnMCResult)
        assert res.num_paths == 50_000
        assert res.sigma == pytest.approx(0.10, rel=1e-6)

    def test_quasi_reports_no_se(self):
        env = _env()
        eng = FxTargetRedemptionNoteMCEngine(
            params=_mc(num_paths=65_536, method=MonteCarloMethod.QUASI))
        eng.price(_note(), env)
        assert eng.get_last_result().std_error is None

    def test_rejects_non_note(self):
        with pytest.raises(PricingError):
            FxTargetRedemptionNoteMCEngine().price(
                FxVanillaOption(strike=1.2, option_type=OptionType.CALL,
                                maturity=1.0, notional_foreign=1.0), _env())

    def test_greeks_finite(self):
        env = _env()
        greeks = FxTargetRedemptionNoteMCEngine(
            params=_mc(num_paths=40_000, seed=2)
        ).calculate_greeks(_note(target=0.05), env)
        for key in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
            assert key in greeks and math.isfinite(greeks[key])
