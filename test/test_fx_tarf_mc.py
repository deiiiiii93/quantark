"""Tests for the FX Target Redemption Forward (TARF) Monte Carlo engine.

Correctness is gated three ways:
  - a sigma->0 deterministic limit, where every path collapses to the forward
    path so the target accumulation / truncation / redemption logic is checkable
    in closed form (covers exact-hit, overshoot, never-hit, single-period);
  - stochastic closed-form strips: target=inf, gearing=1 -> plain forward strip;
    target=inf, gearing!=1 -> forward strip minus (gearing-1) x a GK put strip;
  - engine mechanics (SE semantics, product guard, Greeks).
"""

import math
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.engine.mc import FxTarnForwardMCEngine, FxTarnMCResult
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxTargetRedemptionForward,
    FxVanillaOption,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType
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


def _tarf(**kw):
    base = dict(strike=1.20, fixing_times=[0.25, 0.5, 0.75, 1.0],
               currency_pair=_PAIR)
    base.update(kw)
    return FxTargetRedemptionForward(**base)


def _mc(**kw):
    base = dict(num_paths=200_000, seed=7)
    base.update(kw)
    return FxMCParams(**base)


# ---------------------------------------------------------------------------
# Deterministic forward-path reference (sigma -> 0 limit)
# ---------------------------------------------------------------------------


def _det_pv(forwards, df_pay, K, notional, target, gearing, is_buy_base):
    sign = 1.0 if is_buy_base else -1.0
    acc, pv, alive = 0.0, 0.0, True
    for F, df in zip(forwards, df_pay):
        if not alive:
            break
        diff = sign * (F - K)
        fav, unf = notional * max(diff, 0.0), notional * max(-diff, 0.0)
        fav_eff = min(fav, target - acc)
        pv += (fav_eff - gearing * unf) * df
        acc += fav_eff
        if acc >= target - 1e-12:
            alive = False
    return pv


def _forwards_and_df(env, opt):
    fwd = [env.get_forward(t) for t in opt.fixing_times]
    df = [env.get_domestic_df(t) for t in opt.pay_times]
    return fwd, df


class TestDeterministicLimit:
    """sigma ~ 0: MC must reproduce the exact forward-path redemption PV."""

    def _check(self, opt):
        env = _env(vol=1e-7)
        fwd, df = _forwards_and_df(env, opt)
        ref = _det_pv(fwd, df, opt.strike, opt.notional, opt.target,
                      opt.gearing, opt.is_buy_base)
        mc = FxTarnForwardMCEngine(params=_mc(num_paths=20_000)).price(opt, env)
        assert mc == pytest.approx(ref, abs=1e-6, rel=1e-6)

    def test_unbounded_target(self):
        self._check(_tarf(target=math.inf))

    def test_never_hit(self):
        # Forwards barely above strike; target huge -> never redeems.
        self._check(_tarf(strike=1.20, target=1.0))

    def test_overshoot_truncated(self):
        # Small target so an early period overshoots and is clipped.
        self._check(_tarf(strike=1.18, target=0.01))

    def test_hit_exactly_on_a_period(self):
        # Target equal to the first period's exact favorable amount -> the deal
        # redeems precisely at the end of period 0.
        env = _env(vol=1e-7)
        opt = _tarf(strike=1.18)
        fwd, _ = _forwards_and_df(env, opt)
        first_fav = max(fwd[0] - 1.18, 0.0)
        assert first_fav > 0
        self._check(_tarf(strike=1.18, target=first_fav))

    def test_single_period(self):
        self._check(_tarf(fixing_times=[1.0], target=math.inf))

    def test_sell_base(self):
        # Sell-base: favorable when S < K. Forwards > K here so favorable=0.
        self._check(_tarf(strike=1.30, is_buy_base=False, target=math.inf))

    def test_geared_unfavorable(self):
        self._check(_tarf(strike=1.25, gearing=3.0, target=math.inf))


# ---------------------------------------------------------------------------
# Stochastic closed-form strip gates
# ---------------------------------------------------------------------------


class TestForwardStrip:
    def test_unbounded_gearing_one_is_forward_strip(self):
        env = _env(vol=0.12)
        opt = _tarf(target=math.inf, gearing=1.0, notional=1.0)
        fwd, df = _forwards_and_df(env, opt)
        strip = sum(opt.notional * (F - opt.strike) * d for F, d in zip(fwd, df))
        eng = FxTarnForwardMCEngine(params=_mc())
        mc = eng.price(opt, env)
        se = eng.get_last_result().std_error
        assert se is not None and abs(mc - strip) <= 4.0 * se

    def test_unbounded_geared_is_forward_minus_put_strip(self):
        env = _env(vol=0.12)
        gearing = 2.5
        opt = _tarf(target=math.inf, gearing=gearing, notional=1.0)
        fwd, df = _forwards_and_df(env, opt)
        forward_strip = sum(
            opt.notional * (F - opt.strike) * d for F, d in zip(fwd, df)
        )
        gk = GarmanKohlhagenEngine()
        put_strip = 0.0
        for tf, tp in zip(opt.fixing_times, opt.pay_times):
            put = FxVanillaOption(
                strike=opt.strike, option_type=OptionType.PUT, maturity=tf,
                delivery=tp, notional_foreign=opt.notional, currency_pair=_PAIR,
            )
            put_strip += gk.price(put, env)
        expected = forward_strip - (gearing - 1.0) * put_strip
        eng = FxTarnForwardMCEngine(params=_mc())
        mc = eng.price(opt, env)
        se = eng.get_last_result().std_error
        assert se is not None and abs(mc - expected) <= 4.0 * se


# ---------------------------------------------------------------------------
# Redemption behaviour
# ---------------------------------------------------------------------------


class TestRedemption:
    def test_tiny_target_redeems_almost_always(self):
        env = _env(vol=0.15)
        opt = _tarf(strike=1.10, target=0.005)  # easily reached early
        eng = FxTarnForwardMCEngine(params=_mc(num_paths=80_000))
        eng.price(opt, env)
        assert eng.get_last_result().mean_redemption_fraction > 0.9

    def test_unbounded_target_never_redeems(self):
        env = _env(vol=0.15)
        eng = FxTarnForwardMCEngine(params=_mc(num_paths=80_000))
        eng.price(_tarf(target=math.inf), env)
        assert eng.get_last_result().mean_redemption_fraction == 0.0


# ---------------------------------------------------------------------------
# Product validation
# ---------------------------------------------------------------------------


class TestProduct:
    def test_rejects_nonpositive_target(self):
        with pytest.raises(ValidationError, match="target"):
            _tarf(target=0.0)

    def test_rejects_unsorted_fixings(self):
        with pytest.raises(ValidationError, match="ascending"):
            _tarf(fixing_times=[0.5, 0.25])

    def test_rejects_pay_before_fixing(self):
        with pytest.raises(ValidationError, match="pay"):
            _tarf(fixing_times=[0.5, 1.0], pay_times=[0.4, 1.0])

    def test_rejects_negative_gearing(self):
        with pytest.raises(ValidationError, match="gearing"):
            _tarf(gearing=-1.0)


# ---------------------------------------------------------------------------
# Engine mechanics
# ---------------------------------------------------------------------------


class TestEngine:
    def test_result_fields(self):
        env = _env()
        eng = FxTarnForwardMCEngine(params=_mc(num_paths=50_000))
        eng.price(_tarf(), env)
        res = eng.get_last_result()
        assert isinstance(res, FxTarnMCResult)
        assert res.num_paths == 50_000
        assert res.sigma == pytest.approx(0.10, rel=1e-6)

    def test_quasi_reports_no_se(self):
        env = _env()
        eng = FxTarnForwardMCEngine(
            params=_mc(num_paths=65_536, method=MonteCarloMethod.QUASI))
        eng.price(_tarf(), env)
        assert eng.get_last_result().std_error is None

    def test_rejects_non_tarf(self):
        with pytest.raises(PricingError):
            FxTarnForwardMCEngine().price(
                FxVanillaOption(strike=1.2, option_type=OptionType.CALL,
                                maturity=1.0, notional_foreign=1.0), _env())

    def test_greeks_finite(self):
        env = _env()
        greeks = FxTarnForwardMCEngine(
            params=_mc(num_paths=40_000, seed=2)
        ).calculate_greeks(_tarf(target=0.05), env)
        for key in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
            assert key in greeks and math.isfinite(greeks[key])
