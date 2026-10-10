"""The vol-model routes resolve DATED observation schedules (W12, user-approved fixes 2026-10-10).

A numeric contract shows no difference from its dated twin. Each contract below is written twice: on dates (records
on ``INITIAL + d_k`` with ``d_k = round(k / 12 * 365)``, an ``exercise_date`` 365 days after ``INITIAL``) and
numerically (records at ``d_k / 365`` years, maturity 1.0), so the two forms are the same contract. At several
valuation dates mid-life -- including the days right before and right after an observation -- the numeric twin is
aged to the valuation date (``time_shift``, lifecycle aging), and both are priced by the same route. The two must
agree to rel 1e-10 (the aged numeric times ``d_k / 365 - a / 365`` and the dated ``(d_k - a) / 365`` differ only
in the last ulp: the 1D Black-Scholes snowball PDE shows the same 1e-14 gap).

1. Heston / SLV snowball and Phoenix PDE (2D ADI): refused a dated knock-out schedule before maturity ("needs a
   pricing environment to resolve its observation times") because the 2D S-axis grid request never received the
   environment the 1D adapter stashes. Price, delta and gamma (and the flat-vol-surrogate event statistics).
2. Local-vol / Heston / SLV barrier MC and Heston / SLV barrier PDE: read only the legacy ``observation_dates``,
   which a dated schedule leaves unset ("discrete monitoring requires observation_dates"). Price, delta and gamma on
   up/down, in/out, with a rebate (and a pay-at-hit knock-out).

The existing limits stay: per-observation varying barriers/payoffs, an observation after maturity and a memory
Phoenix on the 2D PDE are refused identically in both forms.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from test_dated_expiry_settles import HESTON, SLV_LEVERAGE, env as _env

from quantark.asset.equity.engine.mc import HestonBarrierMCEngine, HestonSLVBarrierMCEngine, LocalVolBarrierMCEngine
from quantark.asset.equity.engine.pde import (
    HestonBarrierPDESolver,
    HestonPhoenixPDESolver,
    HestonSLVBarrierPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonSLVPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSLVSnowballPDESolver
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.util.exceptions import ValidationError

INITIAL = datetime(2025, 1, 5)
EXERCISE = INITIAL + timedelta(days=365)
#: the twelve monthly observations' contract days
DAYS = [round(k / 12 * 365) for k in range(1, 13)]
#: valuation days: before the first observation, mid-period, right before / after the day-91 observation, late
VALUATION_DAYS = (10, 45, 90, 92, 200)
#: the twin tolerance: the forms differ only in the last ulp of their resolved times
REL = 1e-10
ABS = 1e-10

#: a small 2D ADI grid: parity is a property of the inputs, not of the grid
_ADI = dict(n_x=48, n_v=12, n_t=24)


def env(spot: float, valuation: datetime, grid: bool = False):
    return _env(spot, valuation, grid=grid)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _on(day: int) -> datetime:
    return INITIAL + timedelta(days=day)


def _records(form: str, level: float, days=DAYS, **kw):
    if form == "dated":
        return [ObservationRecord(observation_date=_on(d), barrier=level, **kw) for d in days]
    return [ObservationRecord(observation_time=d / 365.0, barrier=level, **kw) for d in days]


def _set_expiry(product, form: str) -> None:
    product.initial_date = INITIAL
    if form == "dated":
        product.maturity = None
        product.exercise_date = EXERCISE


def _aged(product, form: str, day: int, age_dated: bool = False):
    """The numeric twin aged to the valuation day (lifecycle aging: past records dropped); the dated form is priced
    as booked, its environment's valuation date resolving its records -- unless ``age_dated``: a dated DISCRETE
    knock-in schedule must not hold a record behind the valuation date on any route (the Black-Scholes PDE / MC /
    QUAD refuse it too, pre-existing), so that form is aged the same way."""
    if form == "numeric" or age_dated:
        valuation = _on(day)
        product.time_shift(day / 365.0, valuation, env(100.0, valuation))
    return product


def _assert_twin(dated, numeric, what):
    assert np.isfinite(dated) and np.isfinite(numeric), what
    assert dated == pytest.approx(numeric, rel=REL, abs=ABS), (what, dated, numeric)


# ======================================================================================================================
# 1. Heston / SLV snowball and Phoenix PDE
# ======================================================================================================================

def _snowball(discrete_ki: bool):
    """KO 103 monthly (flat 15%, so the forms pay the same cash), KI 75 continuous or monthly-discrete."""
    def build(form: str):
        p = create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15,
                                     is_annualized=False)
        _set_expiry(p, form)
        cfg = replace(p.barrier_config, ko_observation_dates=None,
                      ko_observation_schedule=ObservationSchedule(records=_records(form, 103.0)))
        if discrete_ki:
            cfg = replace(cfg, ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
                          ki_observation_dates=None,
                          ki_observation_schedule=ObservationSchedule(records=_records(form, 75.0)))
        p.barrier_config = cfg
        p.validate()
        return p
    return build


def _phoenix(form: str):
    """Non-memory Phoenix (the 2D PDE refuses memory coupons): KO 103, coupon 1% per period at 80, KI 70 continuous."""
    p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                coupon_rate=0.01, num_observations=12, memory_coupon=False, is_annualized=False)
    _set_expiry(p, form)
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None,
                               ko_observation_schedule=ObservationSchedule(records=_records(form, 103.0)))
    p.validate()
    return p


AUTOCALL_PRODUCTS = {
    "snowball": _snowball(discrete_ki=False),
    "snowball_discrete_ki": _snowball(discrete_ki=True),
    "phoenix": _phoenix,
}
AUTOCALL_ROUTES = {
    "heston_pde": lambda kind, **kw: (HestonPhoenixPDESolver if kind == "phoenix" else HestonSnowballPDESolver)(
        HESTON, **_ADI, **kw),
    "slv_pde": lambda kind, **kw: (HestonSLVPhoenixPDESolver if kind == "phoenix" else HestonSLVSnowballPDESolver)(
        HESTON, SLV_LEVERAGE, **_ADI, **kw),
}
AUTOCALL_CASES = [(route, kind, day) for route in AUTOCALL_ROUTES for kind in AUTOCALL_PRODUCTS
                  for day in VALUATION_DAYS]


def _autocall_twin(route, kind, day, run):
    out = {}
    for form in ("dated", "numeric"):
        product = _aged(AUTOCALL_PRODUCTS[kind](form), form, day, age_dated=kind == "snowball_discrete_ki")
        out[form] = _quiet(lambda: run(AUTOCALL_ROUTES[route](kind), product, env(100.0, _on(day))))
    return out["dated"], out["numeric"]


@pytest.mark.parametrize("route,kind,day", AUTOCALL_CASES, ids=[f"{r}-{k}-d{d}" for r, k, d in AUTOCALL_CASES])
def test_2d_autocallable_dated_twin_price(route, kind, day):
    dated, numeric = _autocall_twin(route, kind, day, lambda eng, p, e: float(eng.price(p, e)))
    _assert_twin(dated, numeric, "price")


_GREEK_CASES = [c for c in AUTOCALL_CASES if c[2] in (45, 92)]


@pytest.mark.parametrize("route,kind,day", _GREEK_CASES, ids=[f"{r}-{k}-d{d}" for r, k, d in _GREEK_CASES])
def test_2d_autocallable_dated_twin_greeks(route, kind, day):
    """Price, delta and gamma: the snowball's one-surface readout over a frozen bump context, the Phoenix's
    bump-and-reprice over its own (both bind the S-axis grid request with the environment)."""
    dated, numeric = _autocall_twin(route, kind, day, lambda eng, p, e: eng.calculate_greeks(p, e))
    assert set(dated) == set(numeric) and {"price", "delta", "gamma"} <= set(dated)
    for name in dated:
        _assert_twin(float(dated[name]), float(numeric[name]), name)


@pytest.mark.parametrize("route", list(AUTOCALL_ROUTES))
def test_2d_snowball_dated_twin_event_stats(route):
    """The flat-vol-surrogate statistics (the 1D march, with the 2D pv) agree across the forms."""
    def run(eng, p, e):
        return eng.calculate_event_stats(p, e)
    out = {}
    for form in ("dated", "numeric"):
        product = _aged(AUTOCALL_PRODUCTS["snowball"](form), form, 45)
        engine = AUTOCALL_ROUTES[route]("snowball", statistics="flat_vol_surrogate")
        out[form] = _quiet(lambda: run(engine, product, env(100.0, _on(45))))
    dated, numeric = out["dated"], out["numeric"]
    _assert_twin(dated.pv, numeric.pv, "pv")
    for name in ("ko_times", "ko_probability", "survival_probability", "expected_discounted_ko_cashflow"):
        np.testing.assert_allclose(np.asarray(getattr(dated, name)), np.asarray(getattr(numeric, name)),
                                   rtol=REL, atol=ABS, err_msg=name)


@pytest.mark.parametrize("route", list(AUTOCALL_ROUTES))
def test_2d_memory_phoenix_stays_refused_in_both_forms(route):
    """Memory coupons on the 2D Phoenix PDE: the separate, pre-existing refusal, identical in both forms."""
    for form in ("dated", "numeric"):
        p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                    coupon_rate=0.01, num_observations=12, memory_coupon=True, is_annualized=False)
        _set_expiry(p, form)
        p.barrier_config = replace(p.barrier_config, ko_observation_dates=None,
                                   ko_observation_schedule=ObservationSchedule(records=_records(form, 103.0)))
        p.validate()
        p = _aged(p, form, 45)
        with pytest.raises(ValidationError, match="non-memory Phoenix coupons only"):
            _quiet(lambda: AUTOCALL_ROUTES[route]("phoenix").price(p, env(100.0, _on(45))))


# ======================================================================================================================
# 2. Local-vol / Heston / SLV barrier MC and Heston / SLV barrier PDE
# ======================================================================================================================

def _barrier(barrier_type, option_type, level, pay_at_hit=False):
    """K100, rebate 3, monitored on the twelve monthly records."""
    def build(form: str):
        kw = dict(strike=100.0, option_type=option_type, barrier=level, barrier_type=barrier_type, rebate=3.0,
                  pay_at_hit=pay_at_hit, observation_type=ObservationType.DISCRETE,
                  observation_schedule=ObservationSchedule(records=_records(form, level)))
        if form == "dated":
            return BarrierOption(exercise_date=EXERCISE, initial_date=INITIAL, **kw)
        return BarrierOption(maturity=1.0, initial_date=INITIAL, **kw)
    return build


BARRIER_PRODUCTS = {
    "up_out_call": _barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0),
    "up_in_call": _barrier(BarrierType.UP_IN, OptionType.CALL, 120.0),
    "down_out_put": _barrier(BarrierType.DOWN_OUT, OptionType.PUT, 85.0),
    "down_in_put": _barrier(BarrierType.DOWN_IN, OptionType.PUT, 85.0),
    "up_out_call_pay_at_hit": _barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0, pay_at_hit=True),
}


def _mc():
    return MCParams(seed=11, num_paths=512, time_steps=73)


#: (route, engine factory, reads the market IV grid)
BARRIER_ROUTES = {
    "lv_mc": (lambda: LocalVolBarrierMCEngine(params=_mc()), True),
    "heston_mc": (lambda: HestonBarrierMCEngine(HESTON, params=_mc()), False),
    "slv_mc": (lambda: HestonSLVBarrierMCEngine(HESTON, SLV_LEVERAGE, params=_mc()), False),
    "heston_pde": (lambda: HestonBarrierPDESolver(HESTON, **_ADI), False),
    "slv_pde": (lambda: HestonSLVBarrierPDESolver(HESTON, SLV_LEVERAGE, **_ADI), False),
}


def _barrier_aged(kind: str, form: str, day: int):
    """Both forms aged to the valuation day: a dated barrier schedule must not hold a record behind the valuation
    date on any route (the Black-Scholes MC / PDE refuse it too), so the dated form is aged the same way."""
    valuation = _on(day)
    product = BARRIER_PRODUCTS[kind](form)
    product.time_shift(day / 365.0, valuation, env(100.0, valuation))
    return product


def _barrier_twin(route, kind, day, run):
    factory, grid = BARRIER_ROUTES[route]
    out = {}
    for form in ("dated", "numeric"):
        product = _barrier_aged(kind, form, day)
        out[form] = _quiet(lambda: run(factory(), product, env(100.0, _on(day), grid=grid)))
    return out["dated"], out["numeric"]


BARRIER_CASES = [(route, kind, day) for route in BARRIER_ROUTES for kind in BARRIER_PRODUCTS
                 for day in (10, 90, 92)]


@pytest.mark.parametrize("route,kind,day", BARRIER_CASES, ids=[f"{r}-{k}-d{d}" for r, k, d in BARRIER_CASES])
def test_vol_barrier_dated_twin_price(route, kind, day):
    dated, numeric = _barrier_twin(route, kind, day, lambda eng, p, e: float(eng.price(p, e)))
    _assert_twin(dated, numeric, "price")


_BARRIER_GREEK_CASES = [c for c in BARRIER_CASES if c[2] == 90 and not c[1].endswith("pay_at_hit")]


@pytest.mark.parametrize("route,kind,day", _BARRIER_GREEK_CASES,
                         ids=[f"{r}-{k}-d{d}" for r, k, d in _BARRIER_GREEK_CASES])
def test_vol_barrier_dated_twin_greeks(route, kind, day):
    dated, numeric = _barrier_twin(route, kind, day, lambda eng, p, e: eng.calculate_greeks(p, e))
    assert set(dated) == set(numeric) and {"price", "delta", "gamma"} <= set(dated)
    for name in dated:
        _assert_twin(float(dated[name]), float(numeric[name]), name)


def test_vol_barrier_dated_resolution_matches_the_numeric_observation_times():
    """The resolved dated times are the aged numeric times to the last ulp (what the kernels receive)."""
    from quantark.asset.equity.engine.vol_barrier_schedule import discrete_observation_times

    valuation = _on(92)
    dated = _barrier_aged("up_out_call", "dated", 92)
    numeric = _barrier_aged("up_out_call", "numeric", 92)
    e = env(100.0, valuation)
    dated_times = discrete_observation_times(dated, e, dated.get_maturity(e))
    numeric_times = discrete_observation_times(numeric, e, numeric.get_maturity(e))
    np.testing.assert_allclose(dated_times, numeric_times, rtol=0, atol=1e-15)
    assert len(dated_times) == len([d for d in DAYS if d > 92])


@pytest.mark.parametrize("route", list(BARRIER_ROUTES))
def test_vol_barrier_varying_schedule_stays_refused_in_both_forms(route):
    """Per-observation varying barriers stay out of scope, refused identically for the dated form."""
    factory, grid = BARRIER_ROUTES[route]
    for form in ("dated", "numeric"):
        records = _records(form, 120.0)
        records[-1] = replace(records[-1], barrier=125.0)
        kw = dict(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                  rebate=3.0, observation_type=ObservationType.DISCRETE,
                  observation_schedule=ObservationSchedule(records=records))
        product = (BarrierOption(exercise_date=EXERCISE, initial_date=INITIAL, **kw) if form == "dated"
                   else BarrierOption(maturity=1.0, initial_date=INITIAL, **kw))
        with pytest.raises(ValidationError, match="varying barriers/payoffs"):
            _quiet(lambda: factory().price(product, env(100.0, INITIAL, grid=grid)))


@pytest.mark.parametrize("route", list(BARRIER_ROUTES))
def test_vol_barrier_observation_after_maturity_stays_refused_in_both_forms(route):
    """An observation after the exercise date is refused in both forms (the product's own check, or the route's)."""
    factory, grid = BARRIER_ROUTES[route]
    for form in ("dated", "numeric"):
        days = DAYS[:-1] + [400]
        kw = dict(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                  rebate=3.0, observation_type=ObservationType.DISCRETE,
                  observation_schedule=ObservationSchedule(records=_records(form, 120.0, days=days)))
        with pytest.raises(ValidationError):
            product = (BarrierOption(exercise_date=EXERCISE, initial_date=INITIAL, **kw) if form == "dated"
                       else BarrierOption(maturity=1.0, initial_date=INITIAL, **kw))
            _quiet(lambda: factory().price(product, env(100.0, INITIAL, grid=grid)))
