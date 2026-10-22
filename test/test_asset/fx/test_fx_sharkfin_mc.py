"""Tests for the FX sharkfin Monte Carlo engine.

A sharkfin is a capped knock-out vanilla + rebates. Under continuous monitoring
the cap (= barrier) does not bind on surviving paths, so a plain sharkfin
(participation 1, no bonus, no rebate) equals a knock-out vanilla — validated
against both FxBarrierMCEngine (CRN) and the Vanna-Volga analytic under flat
vol. Under discrete monitoring with an unobserved terminal the cap binds.
"""

import math
from datetime import datetime

import pytest

from quantark.asset.fx.engine.mc import (
    FxBarrierMCEngine,
    FxSharkfinMCEngine,
    FxSharkfinMCResult,
)
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.engine.analytical import VannaVolgaBarrierEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxBarrierOption, FxSharkfinOption
from quantark.param import FlatRateCurve, SpotQuote
from quantark.param.vol.vannavolga import (
    DeltaConvention, FXEnv, SmileQuotes, VannaVolgaVolSurface,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxBarrierType, ObservationType, OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError

_VAL = datetime(2026, 6, 15)


def _env(atm=0.10):
    surface = VannaVolgaVolSurface(
        FXEnv(spot=1.20, rd=0.05, rf=0.03, tau=1.0),
        SmileQuotes(sigma_atm=atm, rr25=0.0, bf25_2vol=0.0),
        DeltaConvention.SPOT,
    )
    return FxPricingEnvironment(
        valuation_date=_VAL, spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=surface,
    )


def _shark(**kw):
    base = dict(
        strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL,
        currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0,
    )
    base.update(kw)
    return FxSharkfinOption(**base)


def _equiv_barrier(**kw):
    base = dict(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0,
    )
    base.update(kw)
    return FxBarrierOption(**base)


def _mc(**kw):
    base = dict(num_paths=120_000, time_steps=120, seed=3)
    base.update(kw)
    return FxMCParams(**base)


# ---------------------------------------------------------------------------
# Product validation
# ---------------------------------------------------------------------------


class TestProduct:
    def test_rejects_up_out_put(self):
        with pytest.raises(ValidationError, match="up-and-out"):
            _shark(option_type=OptionType.PUT)

    def test_rejects_down_out_call(self):
        with pytest.raises(ValidationError, match="down-and-out"):
            FxSharkfinOption(strike=1.20, barrier=1.05, is_up=False,
                             option_type=OptionType.CALL, maturity=1.0)

    def test_rejects_barrier_below_strike_for_up_call(self):
        with pytest.raises(ValidationError, match="barrier > strike"):
            _shark(barrier=1.10)

    def test_capped_intrinsic_caps_at_barrier(self):
        s = _shark()
        assert s.capped_intrinsic(1.30) == pytest.approx(0.10)   # 1.30 - 1.20
        assert s.capped_intrinsic(1.50) == pytest.approx(0.15)   # capped at 1.35 - 1.20


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


class TestValidation:
    def test_continuous_equals_ko_vanilla(self):
        # Plain sharkfin (part=1, no bonus/rebate) == KO vanilla under continuous
        # monitoring: surviving paths never breach H so the cap never binds.
        env = _env()
        shark = _shark()
        barrier = _equiv_barrier()
        p_shark = FxSharkfinMCEngine(params=_mc()).price(shark, env)
        p_barrier = FxBarrierMCEngine(params=_mc()).price(barrier, env)
        # Same seed/paths/survival => essentially identical.
        assert p_shark == pytest.approx(p_barrier, rel=1e-9)

    def test_continuous_matches_analytic_flat(self):
        env = _env()
        shark = _shark()
        analytic = VannaVolgaBarrierEngine().price(_equiv_barrier(), env)
        eng = FxSharkfinMCEngine(params=_mc())
        mc = eng.price(shark, env)
        se = eng.get_last_result().std_error
        assert se is not None and abs(mc - analytic) <= 5.0 * se

    def test_ko_rebate_matches_barrier_with_rebate(self):
        env = _env()
        shark = _shark(ko_rebate=0.02)
        barrier = _equiv_barrier(rebate=0.02)
        p_shark = FxSharkfinMCEngine(params=_mc()).price(shark, env)
        p_barrier = FxBarrierMCEngine(params=_mc()).price(barrier, env)
        assert p_shark == pytest.approx(p_barrier, rel=1e-9)

    def test_no_hit_bonus_adds_survival_weighted_value(self):
        env = _env()
        base = FxSharkfinMCEngine(params=_mc()).price(_shark(), env)
        bonus = 0.05
        with_bonus_eng = FxSharkfinMCEngine(params=_mc())
        with_bonus = with_bonus_eng.price(_shark(no_hit_rebate=bonus), env)
        # The bonus is paid on surviving paths only, discounted.
        assert with_bonus > base
        assert (with_bonus - base) < bonus  # survival prob & discount < 1

    def test_discrete_cap_binds_below_ko_vanilla(self):
        # Terminal (T=1.0) is NOT an observation time, so a path can finish above
        # H without being knocked out; the sharkfin caps it, the KO vanilla does
        # not -> sharkfin <= barrier, strictly less in expectation.
        env = _env()
        obs = [0.25, 0.5, 0.75]  # excludes T=1.0
        shark = _shark(monitoring=ObservationType.DISCRETE, observation_times=obs)
        barrier = _equiv_barrier(monitoring=ObservationType.DISCRETE, observation_times=obs)
        p_shark = FxSharkfinMCEngine(params=_mc()).price(shark, env)
        p_barrier = FxBarrierMCEngine(params=_mc()).price(barrier, env)
        assert p_shark < p_barrier


# ---------------------------------------------------------------------------
# Engine mechanics
# ---------------------------------------------------------------------------


class TestEngine:
    def test_result_fields(self):
        env = _env()
        eng = FxSharkfinMCEngine(params=_mc())
        eng.price(_shark(), env)
        res = eng.get_last_result()
        assert isinstance(res, FxSharkfinMCResult)
        assert res.num_paths == 120_000
        assert res.sigma == pytest.approx(0.10, rel=1e-6)

    def test_quasi_reports_no_se(self):
        env = _env()
        eng = FxSharkfinMCEngine(
            params=_mc(num_paths=65_536, method=MonteCarloMethod.QUASI))
        eng.price(_shark(), env)
        assert eng.get_last_result().std_error is None

    def test_rejects_non_sharkfin(self):
        with pytest.raises(PricingError):
            FxSharkfinMCEngine().price(_equiv_barrier(), _env())

    def test_greeks_finite(self):
        env = _env()
        greeks = FxSharkfinMCEngine(
            params=_mc(num_paths=60_000, time_steps=100, seed=2)
        ).calculate_greeks(_shark(), env)
        for key in ("price", "delta", "gamma", "vega", "theta", "rho_dom", "rho_for"):
            assert key in greeks and math.isfinite(greeks[key])
