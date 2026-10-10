"""A knock-in whose discrete schedule records pay other than ``product.rebate`` is refused on every route (W18b, user
ruling 2026-10-10, option (a)).

A knock-in's rebate is paid when no observation hits, so no record's payoff can apply: the contract has one rebate.
The 1D PDE core already refused a record payoff different from it (``_KnockInAdapter.validate``, spec 3C §5.2 (a));
every other route priced it (the Black-Scholes MC and the vol-model routes paid ``product.rebate``, the QUAD paid
vanilla + rebate - a knock-out paying the record's payoff). The same rule, the same message, now on every route, in
both forms. Unaffected: a knock-in whose records pay ``product.rebate`` or carry no payoff, and every knock-out.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_vol_dated_schedules import BARRIER_ROUTES, DAYS, EXERCISE, INITIAL, _on, _quiet, _records, env

from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.engine.pde import HestonBarrierPDESolver, HestonSLVBarrierPDESolver
from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver
from quantark.asset.equity.engine.quad import BarrierQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

REFUSAL = r"a knock-in has one rebate \(3\.0\); a schedule record pays 4\.5 \(spec 3C §5\.2 \(a\)\)"

_FLAT = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=1e-9, rho=0.0)
_UNIT = LeverageSurface(time_grid=np.linspace(0.0, 1.0, 6), strike_grid=np.array([40.0, 100.0, 250.0]),
                        leverage_grid=np.ones((6, 3)))

#: every route that prices a single-barrier ``BarrierOption`` on a discrete schedule: (factory, reads the IV grid)
ROUTES = {
    "bs_mc": (lambda: BarrierOptionMCEngine(params=MCParams(seed=11, num_paths=512, time_steps=73)), False),
    "bs_pde": (lambda: BarrierPDESolver(PDEParams()), False),
    "quad": (lambda: BarrierQuadEngine(params=QuadParams(grid_points=401)), False),
    "lv_pde": (lambda: LocalVolBarrierPDESolver(), True),
    **BARRIER_ROUTES,       # lv_mc, heston_mc, slv_mc, heston_pde, slv_pde (2D)
    "heston_pde_limit": (lambda: HestonBarrierPDESolver(_FLAT), False),
    "slv_pde_limit": (lambda: HestonSLVBarrierPDESolver(_FLAT, _UNIT), False),
}

#: (barrier type, option type, barrier)
KINDS = {
    "up_in_call": (BarrierType.UP_IN, OptionType.CALL, 120.0),
    "down_in_put": (BarrierType.DOWN_IN, OptionType.PUT, 85.0),
    "up_out_call": (BarrierType.UP_OUT, OptionType.CALL, 120.0),
    "down_out_put": (BarrierType.DOWN_OUT, OptionType.PUT, 85.0),
}
_KI = [k for k in KINDS if "_in_" in k]
_KO = [k for k in KINDS if "_out_" in k]


def _product(kind: str, form: str, payoff=None) -> BarrierOption:
    barrier_type, option_type, barrier = KINDS[kind]
    extra = {} if payoff is None else {"payoff": payoff}
    kw = dict(strike=100.0, option_type=option_type, barrier=barrier, barrier_type=barrier_type, rebate=3.0,
              observation_type=ObservationType.DISCRETE,
              observation_schedule=ObservationSchedule(records=_records(form, barrier, **extra)))
    if form == "dated":
        return BarrierOption(exercise_date=EXERCISE, initial_date=INITIAL, **kw)
    return BarrierOption(maturity=1.0, initial_date=INITIAL, **kw)


def _price(route: str, product: BarrierOption, day: int = 10) -> float:
    factory, grid = ROUTES[route]
    valuation = _on(day)
    product.time_shift(day / 365.0, valuation, env(100.0, valuation))
    return _quiet(lambda: float(factory().price(product, env(100.0, valuation, grid=grid))))


CASES = [(route, kind, form) for route in ROUTES for kind in _KI for form in ("dated", "numeric")]
_IDS = [f"{r}-{k}-{f}" for r, k, f in CASES]


@pytest.mark.parametrize("route,kind,form", CASES, ids=_IDS)
def test_a_knock_in_record_paying_other_than_the_rebate_is_refused(route, kind, form):
    with pytest.raises(ValidationError, match=REFUSAL):
        _price(route, _product(kind, form, payoff=4.5))


@pytest.mark.parametrize("route,kind,form", CASES, ids=_IDS)
def test_a_knock_in_record_paying_the_rebate_or_nothing_prices(route, kind, form):
    """Unaffected: the record's payoff equals ``product.rebate``, or is unset; the two are the same contract."""
    explicit = _price(route, _product(kind, form, payoff=3.0))
    unset = _price(route, _product(kind, form))
    assert np.isfinite(explicit) and explicit == unset


KO_CASES = [(route, kind, form) for route in ROUTES for kind in _KO for form in ("dated", "numeric")]


@pytest.mark.parametrize("route,kind,form", KO_CASES, ids=[f"{r}-{k}-{f}" for r, k, f in KO_CASES])
def test_a_knock_out_record_payoff_is_not_refused(route, kind, form):
    assert np.isfinite(_price(route, _product(kind, form, payoff=4.5)))


_MC_ROUTES = ["bs_mc", "lv_mc", "heston_mc", "slv_mc"]


@pytest.mark.parametrize("route", _MC_ROUTES)
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_the_refusal_holds_on_the_exercise_date(route, form):
    """The MC routes' settled value at T = 0 (``barrier_expiry_value``) is refused too: the rule is the route's."""
    product = _product("up_in_call", form, payoff=4.5)
    if form == "numeric":
        product.time_shift(1.0, EXERCISE, env(100.0, EXERCISE), keep_landing=True)
    factory, grid = ROUTES[route]
    with pytest.raises(ValidationError, match=REFUSAL):
        _quiet(lambda: factory().price(product, env(117.0, EXERCISE, grid=grid)))


def test_days_cover_the_schedule():
    """The twelve monthly records the contracts carry (sanity of the fixture)."""
    assert len(_product("up_in_call", "numeric", payoff=4.5).observation_schedule.records) == len(DAYS)
