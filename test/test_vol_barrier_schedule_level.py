"""The vol-model barrier routes price a discrete schedule at the records' own terms (W18, user rulings 2026-10-10).

Routes: local-vol / Heston / SLV barrier MC (``barrier_vol_mc_engines``) and Heston / SLV barrier PDE
(``barrier_vol_pde_solvers``). They accept a discrete schedule only when every record has the same barrier and
payoff (per-observation varying terms stay refused).

1. Ruling (a), barrier level: a schedule whose common record level differs from ``product.barrier`` was priced at
   ``product.barrier`` -- silently, in the numeric and the dated form alike. The routes now price at the records'
   level, exactly as the Black-Scholes barrier MC monitors each record at its own ``rec.barrier``
   (``barrier_option_mc_engine._build_discrete_grid``). A record without a level takes ``product.barrier``
   (``BarrierOption``'s own normalisation). The same holds for a knock-out's common record payoff vs
   ``product.rebate``: the Black-Scholes MC pays the hit record's payoff on a knock-out. A knock-in's rebate is paid
   when no observation hits, and the Black-Scholes MC pays ``product.rebate`` there (no record has fired); the vol
   routes keep that reading.
2. Ruling (a), aggregation: the vol kernels monitor first-hit. Any other ``aggregation_mode`` (ACCUMULATE, BEST,
   WORST) is refused on these routes, in both forms.

Each contract is written twice (dated / numeric, as in ``test_vol_dated_schedules``) and aged to the valuation day.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_vol_dated_schedules import BARRIER_ROUTES, EXERCISE, INITIAL, _on, _quiet, _records, env

from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.engine.pde import HestonBarrierPDESolver, HestonSLVBarrierPDESolver
from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.util.enum.option_enums import ObservationAggregation
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

#: (barrier type, option type, product.barrier, the records' common level)
KINDS = {
    "up_out_call": (BarrierType.UP_OUT, OptionType.CALL, 120.0, 115.0),
    "up_in_call": (BarrierType.UP_IN, OptionType.CALL, 120.0, 115.0),
    "down_out_put": (BarrierType.DOWN_OUT, OptionType.PUT, 85.0, 90.0),
    "down_in_put": (BarrierType.DOWN_IN, OptionType.PUT, 85.0, 90.0),
    "up_out_call_pay_at_hit": (BarrierType.UP_OUT, OptionType.CALL, 120.0, 115.0),
}
REBATE = 3.0
DAYS = (10, 92)
#: the Black-Scholes references (a different engine, or a settled value)
TOL = dict(rel=1e-12, abs=1e-12)


def _product(kind: str, form: str, barrier: float, level, *, rebate: float = REBATE, payoff=None,
             aggregation=ObservationAggregation.STOP_FIRST_HIT) -> BarrierOption:
    """K100, monitored on the twelve monthly records at ``level`` (``None``: no record level), paying ``payoff``
    per record (``None``: none set)."""
    barrier_type, option_type, _, _ = KINDS[kind]
    extra = {} if payoff is None else {"payoff": payoff}
    records = _records(form, level, **extra)
    kw = dict(strike=100.0, option_type=option_type, barrier=barrier, barrier_type=barrier_type, rebate=rebate,
              pay_at_hit=kind.endswith("pay_at_hit"), observation_type=ObservationType.DISCRETE,
              observation_schedule=ObservationSchedule(records=records, aggregation_mode=aggregation))
    if form == "dated":
        return BarrierOption(exercise_date=EXERCISE, initial_date=INITIAL, **kw)
    return BarrierOption(maturity=1.0, initial_date=INITIAL, **kw)


def _aged(product: BarrierOption, day: int) -> BarrierOption:
    valuation = _on(day)
    product.time_shift(day / 365.0, valuation, env(100.0, valuation))
    return product


def _price(route: str, product: BarrierOption, day: int) -> float:
    factory, grid = BARRIER_ROUTES[route]
    return _quiet(lambda: float(factory().price(_aged(product, day), env(100.0, _on(day), grid=grid))))


CASES = [(route, kind, form, day) for route in BARRIER_ROUTES for kind in KINDS for form in ("dated", "numeric")
         for day in DAYS]
_IDS = [f"{r}-{k}-{f}-d{d}" for r, k, f, d in CASES]


# ======================================================================================================================
# 1. the records' common level
# ======================================================================================================================

@pytest.mark.parametrize("route,kind,form,day", CASES, ids=_IDS)
def test_a_common_record_level_is_the_level_priced(route, kind, form, day):
    """``product.barrier`` B, every record at L != B: the price of the same contract booked with barrier L."""
    _, _, barrier, level = KINDS[kind]
    priced = _price(route, _product(kind, form, barrier, level), day)
    booked_at_level = _price(route, _product(kind, form, level, level), day)
    booked_at_barrier = _price(route, _product(kind, form, barrier, barrier), day)
    assert priced == booked_at_level      # bitwise: the kernel receives the same inputs
    # the level matters on this contract: the test is not vacuous
    assert not np.isclose(booked_at_level, booked_at_barrier, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("route,kind,form,day", [c for c in CASES if c[3] == 10],
                         ids=[i for c, i in zip(CASES, _IDS) if c[3] == 10])
def test_a_record_without_a_level_takes_the_product_barrier(route, kind, form, day):
    """``barrier=None`` on every record: the product's barrier (``BarrierOption``'s own normalisation)."""
    _, _, barrier, _ = KINDS[kind]
    assert _price(route, _product(kind, form, barrier, None), day) == _price(
        route, _product(kind, form, barrier, barrier), day)


_KO = [k for k in KINDS if "_out_" in k]
_KI = [k for k in KINDS if "_in_" in k]
PAYOFF_CASES = [(route, kind, form) for route in BARRIER_ROUTES for kind in KINDS for form in ("dated", "numeric")]
_PAYOFF_IDS = [f"{r}-{k}-{f}" for r, k, f in PAYOFF_CASES]


@pytest.mark.parametrize("route,kind,form", PAYOFF_CASES, ids=_PAYOFF_IDS)
def test_a_common_record_payoff_and_level_are_the_terms_priced(route, kind, form):
    """Every record at level L paying 4.5, ``product.barrier`` B and ``product.rebate`` 3. A knock-out pays the hit
    record's payoff: the contract booked at barrier L with rebate 4.5. A knock-in has one rebate, paid when no
    observation hits: a record paying anything else is refused on every route (W18b, user ruling 2026-10-10 (a);
    ``test_knock_in_record_payoff_refused``)."""
    _, _, barrier, level = KINDS[kind]
    if kind in _KI:
        with pytest.raises(ValidationError, match=r"a knock-in has one rebate \(3\.0\); a schedule record pays 4\.5"):
            _price(route, _product(kind, form, barrier, level, payoff=4.5), 10)
        return
    priced = _price(route, _product(kind, form, barrier, level, payoff=4.5), 10)
    booked = _price(route, _product(kind, form, level, level, rebate=4.5), 10)
    assert priced == booked               # bitwise


@pytest.mark.parametrize("kind", list(KINDS))
def test_the_black_scholes_mc_reads_the_record_terms_the_same_way(kind):
    """The reference reading, measured on the Black-Scholes barrier MC (numeric form, day 10): the record level and
    a knock-out's record payoff are priced; a knock-in record paying other than its rebate is refused (W18b)."""
    _, _, barrier, level = KINDS[kind]
    engine = lambda: BarrierOptionMCEngine(params=MCParams(seed=11, num_paths=512, time_steps=73))  # noqa: E731
    e = env(100.0, _on(10))
    if kind in _KI:
        with pytest.raises(ValidationError, match=r"a knock-in has one rebate \(3\.0\); a schedule record pays 4\.5"):
            _quiet(lambda: engine().price(_aged(_product(kind, "numeric", barrier, level, payoff=4.5), 10), e))
        return
    priced = _quiet(lambda: engine().price(_aged(_product(kind, "numeric", barrier, level, payoff=4.5), 10), e))
    booked = _quiet(lambda: engine().price(_aged(_product(kind, "numeric", level, level, rebate=4.5), 10), e))
    assert priced == pytest.approx(booked, **TOL)


_GREEK_ROUTES = list(BARRIER_ROUTES)


@pytest.mark.parametrize("route", _GREEK_ROUTES)
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_greeks_follow_the_common_record_level(route, form):
    factory, grid = BARRIER_ROUTES[route]
    e = env(100.0, _on(90), grid=grid)
    priced = _quiet(lambda: factory().calculate_greeks(_aged(_product("up_out_call", form, 120.0, 115.0), 90), e))
    booked = _quiet(lambda: factory().calculate_greeks(_aged(_product("up_out_call", form, 115.0, 115.0), 90), e))
    assert set(priced) == set(booked)
    for key in booked:
        assert float(priced[key]) == float(booked[key]), key   # bitwise


# -- the 2D PDE routes' deterministic-variance limit (vol-of-vol below the kernel floor): the 1D core -------------

def _flat():
    return HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=1e-9, rho=0.0)


def _flat_unit_leverage():
    return LeverageSurface(time_grid=np.linspace(0.0, 1.0, 6), strike_grid=np.array([40.0, 100.0, 250.0]),
                           leverage_grid=np.ones((6, 3)))


@pytest.mark.parametrize("make", [lambda: HestonBarrierPDESolver(_flat()),
                                  lambda: HestonSLVBarrierPDESolver(_flat(), _flat_unit_leverage())],
                         ids=["heston_pde_limit", "slv_pde_limit"])
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_the_deterministic_limit_prices_the_record_level_as_the_black_scholes_pde(make, form):
    """In the flat deterministic-variance limit the routes march the 1D core, which reads each record's level: the
    level-differing schedule prices as the Black-Scholes barrier PDE does on the same contract."""
    e = env(100.0, _on(10))
    priced = _quiet(lambda: make().price(_aged(_product("up_out_call", form, 120.0, 115.0), 10), e))
    reference = _quiet(lambda: BarrierPDESolver(PDEParams()).price(
        _aged(_product("up_out_call", form, 120.0, 115.0), 10), e))
    at_barrier = _quiet(lambda: make().price(_aged(_product("up_out_call", form, 120.0, 120.0), 10), e))
    assert priced == pytest.approx(reference, rel=2e-3)
    assert not np.isclose(priced, at_barrier, rtol=1e-3)


# -- on the exercise date (the W9 settled-value limit) ------------------------------------------------------------

def _settled_product(form: str, barrier: float, level: float) -> BarrierOption:
    """An up-and-out call (rebate 5) whose one record sits on its exercise instant, priced on that day."""
    kw = dict(strike=100.0, option_type=OptionType.CALL, barrier=barrier, barrier_type=BarrierType.UP_OUT,
              rebate=5.0, observation_type=ObservationType.DISCRETE)
    if form == "dated":
        recs = [ObservationRecord(observation_date=EXERCISE, barrier=level)]
        return BarrierOption(exercise_date=EXERCISE, initial_date=INITIAL,
                             observation_schedule=ObservationSchedule(records=recs), **kw)
    recs = [ObservationRecord(observation_time=1.0, barrier=level)]
    product = BarrierOption(maturity=1.0, initial_date=INITIAL, observation_schedule=ObservationSchedule(records=recs),
                            **kw)
    product.time_shift(1.0, EXERCISE, env(100.0, EXERCISE), keep_landing=True)
    return product


_PDE_ROUTES = [r for r in BARRIER_ROUTES if r.endswith("_pde")]


@pytest.mark.parametrize("route", _PDE_ROUTES)
@pytest.mark.parametrize("form", ["dated", "numeric"])
@pytest.mark.parametrize("spot", [117.0, 112.0])
def test_the_pde_routes_settle_on_the_record_level_on_the_exercise_date(route, form, spot):
    """Spot 117 is beyond the record level 115 but short of ``product.barrier`` 120: knocked out (the rebate 5); spot
    112 survives (the payoff 12). Equal to the Black-Scholes barrier PDE and to the contract booked at 115."""
    factory, grid = BARRIER_ROUTES[route]
    e = env(spot, EXERCISE, grid=grid)
    priced = _quiet(lambda: factory().price(_settled_product(form, 120.0, 115.0), e))
    reference = _quiet(lambda: BarrierPDESolver(PDEParams()).price(_settled_product(form, 120.0, 115.0), e))
    booked = _quiet(lambda: factory().price(_settled_product(form, 115.0, 115.0), e))
    assert priced == pytest.approx(reference, **TOL)
    assert priced == pytest.approx(booked, **TOL)
    assert priced == pytest.approx(5.0 if spot > 115.0 else spot - 100.0, **TOL)


# ======================================================================================================================
# 2. aggregation: first-hit only
# ======================================================================================================================

NON_FIRST_HIT = [ObservationAggregation.ACCUMULATE, ObservationAggregation.BEST, ObservationAggregation.WORST]
AGG_CASES = [(route, kind, form, mode) for route in BARRIER_ROUTES for kind in ("up_out_call", "down_in_put")
             for form in ("dated", "numeric") for mode in NON_FIRST_HIT]


@pytest.mark.parametrize("route,kind,form,mode", AGG_CASES,
                         ids=[f"{r}-{k}-{f}-{m.value}" for r, k, f, m in AGG_CASES])
def test_a_non_first_hit_aggregation_is_refused(route, kind, form, mode):
    _, _, barrier, _ = KINDS[kind]
    product = _product(kind, form, barrier, barrier, aggregation=mode)
    with pytest.raises(ValidationError, match=f"aggregation {mode.value} is refused"):
        _price(route, product, 10)


@pytest.mark.parametrize("make", [lambda: HestonBarrierPDESolver(_flat()),
                                  lambda: HestonSLVBarrierPDESolver(_flat(), _flat_unit_leverage())],
                         ids=["heston_pde_limit", "slv_pde_limit"])
@pytest.mark.parametrize("mode", NON_FIRST_HIT, ids=[m.value for m in NON_FIRST_HIT])
def test_the_deterministic_limit_refuses_a_non_first_hit_aggregation_too(make, mode):
    product = _aged(_product("up_out_call", "numeric", 120.0, 120.0, aggregation=mode), 10)
    with pytest.raises(ValidationError, match=f"aggregation {mode.value} is refused"):
        _quiet(lambda: make().price(product, env(100.0, _on(10))))


@pytest.mark.parametrize("route", list(BARRIER_ROUTES))
def test_greeks_refuse_a_non_first_hit_aggregation(route):
    factory, grid = BARRIER_ROUTES[route]
    product = _aged(_product("up_out_call", "dated", 120.0, 120.0, aggregation=ObservationAggregation.ACCUMULATE), 10)
    with pytest.raises(ValidationError, match="aggregation accumulate is refused"):
        _quiet(lambda: factory().calculate_greeks(product, env(100.0, _on(10), grid=grid)))
