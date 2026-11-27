"""Public ``knocked_in_at_valuation`` on Snowball, Phoenix and KO-reset (0.16.0 run 6).

The property is the public spelling of the lifecycle flag ``_otc_lifecycle_knocked_in`` (one storage slot, spec D1);
the constructor keyword writes the flag only when true, as its last statement (D2), so a product built with the keyword
is the same object state -- key order and pickle bytes included -- as one flagged through the alias after construction.
It is not a dataclass field (D3). The Snowball/Phoenix lifecycle tracker seeds its state from the flag as the KO-reset
one does, and both refuse a flag that contradicts an explicit not-knocked-in state (D7).
"""

from __future__ import annotations

import pickle
from copy import deepcopy
from dataclasses import asdict, fields, replace

import pandas as pd
import pytest

from quantark.asset.equity.engine.mc import HestonSnowballMCEngine, PhoenixMCEngine, SnowballMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import HestonPhoenixMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2.engine import (
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.ko_reset import KOResetLifecycleTracker
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import KnockOutResetSnowballOption, PhoenixOption, SnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

KO_DATES = [0.25, 0.5, 0.75, 1.0]
START = pd.Timestamp("2026-01-05")


def _schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def snowball(**kw):
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(KO_DATES), ki_barrier=75.0, ki_continuous=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True), **kw)


def phoenix(**kw):
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(KO_DATES), ki_barrier=75.0, ki_continuous=True),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0] * 4, coupon_rate=0.02,
                                          coupon_pay_type=CouponPayType.INSTANT,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=True),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True), **kw)


def ko_reset(mode=PostKOScheduleMode.ABSOLUTE, **kw):
    # REBASED: the maturity covers the last knock-in (1.0) plus the last offset (1.5) (owner ruling O1, 2026-11-27)
    maturity = 1.5 if mode is PostKOScheduleMode.ABSOLUTE else 2.5
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(KO_DATES, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule(KO_DATES, 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule((0.5, 0.75, 1.0, 1.25, 1.5), 95.0)),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True), post_ko_mode=mode, **kw)


BUILDERS = {"snowball": snowball, "phoenix": phoenix, "ko_reset": ko_reset}


def by_alias(family, flag=True):
    p = BUILDERS[family]()
    p._otc_lifecycle_knocked_in = flag
    return p


def env(spot=90.0):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="IDX"),
                              vol_surface=FlatVolSurface(volatility=0.22), rate_curve=FlatRateCurve(rate=0.03),
                              valuation_date=START.to_pydatetime())


# --- the property and the constructor keyword (AC1, AC2) -----------------------------------------------------------

@pytest.mark.parametrize("family", list(BUILDERS))
def test_default_writes_nothing_and_reads_false(family):
    p = BUILDERS[family]()
    assert p.knocked_in_at_valuation is False
    assert "_otc_lifecycle_knocked_in" not in vars(p)
    assert list(vars(p)) == list(vars(BUILDERS[family](knocked_in_at_valuation=False)))


@pytest.mark.parametrize("family", list(BUILDERS))
def test_keyword_is_the_alias_state_key_order_and_pickle_bytes(family):
    a = BUILDERS[family](knocked_in_at_valuation=True)
    b = by_alias(family)
    assert a.knocked_in_at_valuation is True and b.knocked_in_at_valuation is True
    assert list(vars(a).items()) == list(vars(b).items())
    assert pickle.dumps(a) == pickle.dumps(b)


@pytest.mark.parametrize("family", list(BUILDERS))
def test_setter_writes_the_alias(family):
    p = BUILDERS[family]()
    p.knocked_in_at_valuation = 1
    assert p._otc_lifecycle_knocked_in is True
    p.knocked_in_at_valuation = False
    assert p._otc_lifecycle_knocked_in is False and p.knocked_in_at_valuation is False
    p._otc_lifecycle_knocked_in = True
    assert p.knocked_in_at_valuation is True


@pytest.mark.parametrize("family", list(BUILDERS))
def test_not_a_dataclass_field_and_ignored_by_equality_asdict_and_cache_key(family):
    a = BUILDERS[family](knocked_in_at_valuation=True)
    plain = BUILDERS[family]()
    assert "knocked_in_at_valuation" not in {f.name for f in fields(a)}
    assert a == plain
    assert asdict(a) == asdict(plain)
    if hasattr(a, "cache_key"):
        assert a.cache_key() == plain.cache_key()


@pytest.mark.parametrize("family", list(BUILDERS))
def test_replace_refuses_both_spellings_as_before(family):
    for p in (BUILDERS[family](knocked_in_at_valuation=True), by_alias(family)):
        with pytest.raises(TypeError):
            replace(p)


@pytest.mark.parametrize("family", list(BUILDERS))
def test_copies_keep_the_flag(family):
    # pickle round-trips an object this test built itself (no untrusted data)
    p = BUILDERS[family](knocked_in_at_valuation=True)
    assert deepcopy(p).knocked_in_at_valuation is True
    assert pickle.loads(pickle.dumps(p)).knocked_in_at_valuation is True


# --- engine smoke (AC1) ---------------------------------------------------------------------------------------------

HESTON = HestonParams(v0=0.05, kappa=2.0, theta=0.05, sigma=0.4, rho=-0.6)
ENGINES = {
    "snowball": {
        "mc": lambda: SnowballMCEngine(params=MCParams(num_paths=20_000, seed=7)),
        "heston_mc": lambda: HestonSnowballMCEngine(HESTON, params=MCParams(num_paths=20_000, seed=7)),
        "pde": lambda: SnowballPDESolver(PDEParams()),
        "v1": lambda: SnowballQuadEngine(params=QuadParams(grid_points=401)),
        "v2": SnowballQuadEngineV2,
    },
    "phoenix": {
        "mc": lambda: PhoenixMCEngine(params=MCParams(num_paths=20_000, seed=7)),
        "heston_mc": lambda: HestonPhoenixMCEngine(HESTON, params=MCParams(num_paths=20_000, seed=7)),
        "pde": lambda: PhoenixPDESolver(PDEParams()),
        "v1": lambda: PhoenixQuadEngine(params=QuadParams(grid_points=401)),
        "v2": PhoenixQuadEngineV2,
    },
    "ko_reset": {
        "mc": lambda: SnowballMCEngine(params=MCParams(num_paths=20_000, seed=7)),
        "heston_mc": lambda: HestonSnowballMCEngine(HESTON, params=MCParams(num_paths=20_000, seed=7)),
        "pde": lambda: KOResetSnowballPDESolver(PDEParams()),
        "v1": lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=401)),
        "v2": KOResetSnowballQuadEngineV2,
    },
}
CASES = [(f, e) for f in ENGINES for e in ENGINES[f]]


def _hex_rows(stats):
    out = {"pv": float(stats.pv).hex()}
    for name in ("ko_probabilities", "ki_probabilities"):
        rows = getattr(stats, name, None)
        if rows is not None:
            out[name] = [float(v).hex() for v in rows]
    return out


@pytest.mark.parametrize("family,engine", CASES, ids=[f"{f}-{e}" for f, e in CASES])
def test_engines_price_the_keyword_as_the_alias_bitwise(family, engine):
    make = ENGINES[family][engine]
    a, b = BUILDERS[family](knocked_in_at_valuation=True), by_alias(family)
    pa, pb = make().price(a, env()), make().price(b, env())
    assert float(pa).hex() == float(pb).hex()
    assert pa != make().price(BUILDERS[family](), env())   # the flag is live on this engine
    if engine != "heston_mc":
        assert _hex_rows(make().calculate_event_stats(a, env())) == _hex_rows(make().calculate_event_stats(b, env()))


# --- lifecycle trackers (AC3, D7) -----------------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["snowball", "phoenix"])
def test_autocallable_tracker_seeds_its_state_from_the_flag(family):
    flagged = BUILDERS[family](knocked_in_at_valuation=True)
    tracker = AutocallableLifecycleTracker(product=flagged, quantity=1.0, start_date=START)
    assert tracker.lifecycle.knocked_in is True
    assert tracker.product_for_lifecycle().knocked_in_at_valuation is True
    stated = AutocallableLifecycleTracker(product=BUILDERS[family](), quantity=1.0, start_date=START,
                                          lifecycle=AutocallableLifecycleState(knocked_in=True))
    day = START + pd.Timedelta(days=10)
    e = env()
    a, b = tracker.product_for_pricing(day, e), stated.product_for_pricing(day, e)
    assert a.knocked_in_at_valuation is True and b.knocked_in_at_valuation is True
    engine = ENGINES[family]["v1"]
    assert float(engine().price(a, e)).hex() == float(engine().price(b, e)).hex()


@pytest.mark.parametrize("family", ["snowball", "phoenix"])
def test_autocallable_tracker_without_the_flag_starts_not_knocked_in(family):
    tracker = AutocallableLifecycleTracker(product=BUILDERS[family](), quantity=1.0, start_date=START)
    assert tracker.lifecycle.knocked_in is False


@pytest.mark.parametrize("tracker_cls,family", [(AutocallableLifecycleTracker, "snowball"),
                                                (AutocallableLifecycleTracker, "phoenix"),
                                                (KOResetLifecycleTracker, "ko_reset")])
def test_trackers_refuse_a_flag_that_contradicts_an_explicit_state(tracker_cls, family):
    with pytest.raises(ValidationError, match="knocked_in_at_valuation"):
        tracker_cls(product=BUILDERS[family](knocked_in_at_valuation=True), quantity=1.0, start_date=START,
                    lifecycle=AutocallableLifecycleState(knocked_in=False))


def test_ko_reset_tracker_seeds_from_the_keyword():
    tracker = KOResetLifecycleTracker(product=ko_reset(knocked_in_at_valuation=True), quantity=1.0, start_date=START)
    assert tracker.lifecycle.knocked_in is True


# --- REBASED KO-reset (AC1b) ----------------------------------------------------------------------------------------

def _error(call):
    try:
        call()
    except Exception as exc:   # noqa: BLE001 -- the test compares the two spellings' refusals
        return type(exc), str(exc)
    return None


def test_rebased_ko_reset_refuses_the_keyword_as_it_refuses_the_alias():
    kw = ko_reset(mode=PostKOScheduleMode.REBASED, knocked_in_at_valuation=True)
    alias = ko_reset(mode=PostKOScheduleMode.REBASED)
    alias._otc_lifecycle_knocked_in = True
    mc = lambda p: SnowballMCEngine(params=MCParams(num_paths=2_000, seed=7)).price(p, env())  # noqa: E731
    tracker = lambda p: KOResetLifecycleTracker(product=p, quantity=1.0, start_date=START)  # noqa: E731
    for call in (mc, tracker):
        got = _error(lambda: call(kw))
        assert got is not None and got == _error(lambda: call(alias))


# --- RFQ registry (AC5) ---------------------------------------------------------------------------------------------

def test_rfq_registry_builds_a_flagged_snowball():
    from quantark.rfq.registry import PRODUCT_BUILDERS

    p = PRODUCT_BUILDERS.build("snowball_option", dict(
        initial_price=100.0, strike=100.0, maturity=1.0,
        barrier_config=dict(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=list(KO_DATES), ki_barrier=75.0,
                            ki_continuous=True),
        knocked_in_at_valuation=True))
    assert p.knocked_in_at_valuation is True
