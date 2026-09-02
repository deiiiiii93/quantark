"""ScenarioCache: state-set construction, endpoints, memoisation (spec §6)."""
from copy import deepcopy
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.vol.vol_surface import BlackImpliedVolSurface
from quantark.pnlexplain.base import ExplainMethod, Factor
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.coordinate import resolve_coordinate
from quantark.pnlexplain.equity.factor_diff import build_factor_moves
from quantark.pnlexplain.equity.lifecycle import resolve_transition
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.greek_conventions import GreekConvention

FRI = datetime(2026, 6, 26)
MON = datetime(2026, 6, 29)
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))
ENG = BlackScholesEngine()


class SkewSurface(BlackImpliedVolSurface):
    is_smile = True

    def __init__(self, atm, slope):
        self.atm, self.slope = atm, slope

    def get_vol(self, strike, time_to_maturity, spot=None):
        return self.atm + self.slope * (float(strike) - 100.0)


def _env(spot, vol, rate, div, date):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div), valuation_date=date)


def _call(m):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=m)


def _pair(vol0=FlatVolSurface(0.20), vol1=FlatVolSurface(0.22), spot1=103.0, rate1=0.032, div1=0.012):
    e0 = _env(100.0, vol0, 0.03, 0.01, FRI)
    e1 = _env(spot1, vol1, rate1, div1, MON)
    s0 = ValuationSnapshot(_call(1.0), ENG, e0, date=FRI, quantity=2.0)
    s1 = ValuationSnapshot(_call(1.0 - 3 / 365), ENG, e1, date=MON, quantity=2.0)
    return s0, s1


def _cache(s0, s1, config=WF):
    t = resolve_transition(s0, s1, None, calendar_days=3)
    coord = resolve_coordinate(s0.product, float(s0.pricing_env.spot), t.product_alive_t1, s1.pricing_env)
    mv = build_factor_moves(s0, s1, coord, engine_alive_t1=t.engine_alive_t1, lifecycle_changed=t.changed)
    return ScenarioCache(s0, s1, t, mv, config)


def test_effective_is_changed_and_applicable_and_normalize_drops_the_rest():
    c = _cache(*_pair())
    assert c.effective == frozenset({Factor.TIME, Factor.SPOT, Factor.VOL, Factor.RATE, Factor.DIVIDEND})
    assert c.normalize((Factor.BASIS, Factor.SPOT, Factor.MODEL)) == frozenset({Factor.SPOT})
    assert c.bump_engine_t0 is c.bump_engine_alive is c.bump_engine_t1


def test_endpoints_and_time_pure():
    s0, s1 = _pair()
    c = _cache(s0, s1)
    eng = BlackScholesEngine()
    assert c.value_for(()).total == pytest.approx(2.0 * eng.price(s0.product, s0.pricing_env))
    assert c.all_market().total == pytest.approx(2.0 * eng.price(s1.product, s1.pricing_env))
    assert c.value_t1().total == pytest.approx(c.all_market().total)     # no event: endpoints coincide
    rolled = deepcopy(s0.pricing_env)
    rolled.valuation_date = MON
    expected = 2.0 * (eng.price(s1.product, rolled) - eng.price(s0.product, s0.pricing_env))
    assert c.time_pure() == pytest.approx(expected, abs=1e-12)


def test_build_state_applies_only_the_requested_factors_without_mutating_inputs():
    s0, s1 = _pair()
    c = _cache(s0, s1)
    product, engine, env, point = c.build_state((Factor.SPOT, Factor.RATE))
    assert product is s0.product and engine is c.bump_engine_t0
    assert env.valuation_date == FRI and env.spot == 103.0 and point.date == FRI
    assert env.get_rate(1.0) == pytest.approx(0.032)
    assert env.get_vol(100.0, 1.0) == pytest.approx(0.20)
    assert env.get_div_yield(1.0) == pytest.approx(0.01)
    product, _, env, point = c.build_state((Factor.TIME,))
    assert product is s1.product and env.valuation_date == MON and env.spot == 100.0 and point.date == MON
    assert s0.pricing_env.spot == 100.0 and s0.pricing_env.valuation_date == FRI


def test_sticky_moneyness_reshapes_the_surface_only_while_vol_is_unapplied():
    s0, s1 = _pair(vol0=SkewSurface(0.20, 0.002), vol1=SkewSurface(0.21, 0.002))
    strike = _cache(s0, s1)
    money = _cache(s0, s1, PnLExplainConfig(methods=(ExplainMethod.WATERFALL,),
                                            spot_convention=GreekConvention.STICKY_MONEYNESS))
    _, _, env_k, _ = strike.build_state((Factor.SPOT,))
    _, _, env_m, _ = money.build_state((Factor.SPOT,))
    # sticky strike: the t0 surface is read as-is at the new spot (the env is a deep copy)
    assert env_k.get_vol(103.0, 1.0) == pytest.approx(s0.pricing_env.get_vol(103.0, 1.0))
    # sticky moneyness: the vol at the new spot is the t0 vol at the same moneyness
    assert env_m.get_vol(103.0, 1.0) == pytest.approx(s0.pricing_env.get_vol(100.0, 1.0))
    assert env_m.get_vol(103.0, 1.0) != pytest.approx(env_k.get_vol(103.0, 1.0))
    _, _, env_mv, _ = money.build_state((Factor.SPOT, Factor.VOL))
    assert env_mv.vol_surface is s1.pricing_env.vol_surface                    # t1 surface wins


def test_values_are_memoised_per_normalised_set(monkeypatch):
    s0, s1 = _pair(vol1=FlatVolSurface(0.20), rate1=0.03, div1=0.01)          # only time + spot move
    calls = []
    real = BlackScholesEngine.price

    def counting(self, product, env, **kw):
        calls.append(env.valuation_date)
        return real(self, product, env, **kw)

    monkeypatch.setattr(BlackScholesEngine, "price", counting)
    c = _cache(s0, s1)
    c.value_for(())
    c.value_for((Factor.BASIS,))
    c.value_for((Factor.MODEL, Factor.VOL))
    assert len(calls) == 1                                    # all normalise to the empty set
    c.all_market()
    c.all_market()
    c.value_for((Factor.TIME, Factor.SPOT))
    assert len(calls) == 2
    c.value_t1()
    c.value_t1()
    assert len(calls) == 3
