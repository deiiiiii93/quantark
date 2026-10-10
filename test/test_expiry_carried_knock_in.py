"""A knock-in carried as triggered decides the exercise date (user ruling 2026-10-10: "check knock in state first").

On the exercise date (T = 0: a dated contract on its exercise instant, or its numeric twin aged to ``maturity == 0.0``)
the knock-in decision is the carried knock-in state FIRST -- ``lifecycle_state.knocked_in`` or the product's
``_otc_lifecycle_knocked_in`` (``monitoring_ended.knock_in_carried``, the helper the T > 0 path reads) -- then today's
observation. A knocked-in contract pays its payoff; only a contract never knocked in pays the rebate. Nothing else is
new: "carried OR hit today".

Every route that prices the contract is checked in both forms against the hand settled value, and the day before
(T = 1/365, dated and numeric) the carried state prices on the same branch: within one day's spot standard deviation
of the settled value (the other branch is further than that).

Part 1, barriers: single barrier on every route (BSM analytical / MC / PDE / QUAD, local-vol MC / PDE, Heston and SLV
MC / PDE), double knock-in (analytical, PDE, QUAD). Part 2, autocallables with a knock-in term: snowball, Phoenix
(memory and not), knocked-in KO-reset, on every engine that routes them.
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pytest

from test_dated_expiry_settles import (
    _ADI,
    HESTON,
    INITIAL,
    SLV_LEVERAGE,
    VALUATION,
    env as _env,
)

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine, DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.mc import (
    BarrierOptionMCEngine,
    HestonBarrierMCEngine,
    HestonPhoenixMCEngine,
    HestonSLVBarrierMCEngine,
    HestonSnowballMCEngine,
    LocalVolBarrierMCEngine,
    LocalVolPhoenixMCEngine,
    LocalVolSnowballMCEngine,
    PhoenixMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonSLVPhoenixMCEngine,
    HestonSLVQEPhoenixMCEngine,
    QEPhoenixMCEngine,
)
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    HestonSLVSnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    HestonBarrierPDESolver,
    HestonPhoenixPDESolver,
    HestonSLVBarrierPDESolver,
    HestonSnowballPDESolver,
    LocalVolBarrierPDESolver,
    LocalVolPhoenixPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
from quantark.asset.equity.engine.pde.double_barrier_pde_solver import DoubleBarrierPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonSLVPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSLVSnowballPDESolver
from quantark.asset.equity.engine.quad import (
    BarrierQuadEngine,
    KOResetSnowballQuadEngine,
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngine,
    PhoenixQuadEngineV2,
    SnowballQuadEngine,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import BarrierOption, DoubleBarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import BarrierType, DoubleBarrierType, ObservationType, OptionType

#: the day before the exercise date
BEFORE = VALUATION - timedelta(days=1)


def env(spot: float, valuation: datetime = VALUATION, route: str = ""):
    """Flat 20% vol; the local-vol and SLV routes read it as a market IV grid (their Dupire legs need one)."""
    return _env(spot, valuation, grid=route.startswith(("lv_", "slv_")))


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _aged(product, valuation: datetime):
    """The numeric contract (inception = INITIAL, one year) aged to ``valuation``: maturity 0.0 on the exercise date,
    1/365 the day before, keeping the record on the landing instant (ruling D2)."""
    years = (valuation - INITIAL).days / 365.0
    product.time_shift(years, valuation, env(100.0, valuation), keep_landing=True)
    return product


def _mc():
    """256 antithetic paths: the day-before knocked-in vanilla stays above the BS MC engine's European lower-bound
    check (the settled values on the exercise date read no path)."""
    return MCParams(seed=7, num_paths=256, use_antithetic=True)


def one_day_sd(spot: float) -> float:
    """One day's spot standard deviation at the test's 20% vol: the continuity bound on the day before."""
    return 0.20 * spot * math.sqrt(1.0 / 365.0)


# ======================================================================================================================
# Part 1: barriers
# ======================================================================================================================

def _barrier(barrier_type, option_type, level, discrete=False):
    """K100, rebate 5, monitored continuously or on one record on the exercise date (the earlier observations are
    behind the contract: what they decided is the carried state)."""
    def build(form, carried, valuation=VALUATION):
        kw = dict(strike=100.0, option_type=option_type, barrier=level, barrier_type=barrier_type, rebate=5.0)
        if discrete:
            records = ([ObservationRecord(observation_date=VALUATION, barrier=level)] if form == "dated"
                       else [ObservationRecord(observation_time=1.0, barrier=level)])
            kw.update(observation_type=ObservationType.DISCRETE, observation_schedule=ObservationSchedule(records=records))
        else:
            kw.update(observation_type=ObservationType.CONTINUOUS)
        product = (BarrierOption(exercise_date=VALUATION, initial_date=INITIAL, **kw) if form == "dated"
                   else _aged(BarrierOption(maturity=1.0, **kw), valuation))
        if carried:
            product._otc_lifecycle_knocked_in = True
        return product
    return build


def _double_knock_in(form, carried, valuation=VALUATION):
    kw = dict(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=80.0,
              barrier_type=DoubleBarrierType.KNOCK_IN, rebate=5.0, observation_type=ObservationType.CONTINUOUS)
    product = (DoubleBarrierOption(exercise_date=VALUATION, initial_date=INITIAL, **kw) if form == "dated"
               else _aged(DoubleBarrierOption(maturity=1.0, **kw), valuation))
    if carried:
        product._otc_lifecycle_knocked_in = True
    return product


def _knock_in_value(payoff: Callable[[float], float], hit: Callable[[float], bool]):
    """A knock-in's settled value: knocked in (carried, or hit today) pays the payoff, never knocked in the rebate."""
    return lambda spot, carried: payoff(spot) if (carried or hit(spot)) else 5.0


_CALL = lambda s: max(s - 100.0, 0.0)       # noqa: E731
_PUT = lambda s: max(100.0 - s, 0.0)        # noqa: E731

#: (name, builder, hand value (spot, carried) -> settled, spots)
BARRIER_PRODUCTS = (
    ("up_in_call", _barrier(BarrierType.UP_IN, OptionType.CALL, 130.0),
     _knock_in_value(_CALL, lambda s: s >= 130.0), (110.0, 90.0, 135.0)),
    ("down_in_put", _barrier(BarrierType.DOWN_IN, OptionType.PUT, 80.0),
     _knock_in_value(_PUT, lambda s: s <= 80.0), (90.0, 110.0, 75.0)),
    ("up_in_call_discrete", _barrier(BarrierType.UP_IN, OptionType.CALL, 130.0, discrete=True),
     _knock_in_value(_CALL, lambda s: s >= 130.0), (110.0, 135.0)),
)

#: (route, engine factory, accepts ``lifecycle_state``)
BARRIER_ROUTES = (
    ("analytical", BarrierAnalyticalEngine, True),
    ("mc", lambda: BarrierOptionMCEngine(params=_mc()), True),
    ("pde", lambda: BarrierPDESolver(PDEParams()), False),
    ("quad", BarrierQuadEngine, False),
    ("lv_mc", lambda: LocalVolBarrierMCEngine(params=_mc()), True),
    ("lv_pde", LocalVolBarrierPDESolver, False),
    ("heston_mc", lambda: HestonBarrierMCEngine(HESTON, params=_mc()), True),
    ("heston_pde", lambda: HestonBarrierPDESolver(HESTON, **_ADI), False),
    ("slv_mc", lambda: HestonSLVBarrierMCEngine(HESTON, SLV_LEVERAGE, params=_mc()), True),
    ("slv_pde", lambda: HestonSLVBarrierPDESolver(HESTON, SLV_LEVERAGE, **_ADI), False),
)

DOUBLE_ROUTES = (
    ("analytical", DoubleBarrierOptionAnalyticalEngine, True),
    ("pde", lambda: DoubleBarrierPDESolver(PDEParams()), False),
    ("quad", BarrierQuadEngine, False),
)
DOUBLE_PRODUCT = ("double_knock_in_call", _double_knock_in,
                  _knock_in_value(_CALL, lambda s: s <= 80.0 or s >= 130.0), (110.0, 90.0, 135.0))

#: (route, product) cells whose DAY-BEFORE price is refused for a reason that is not the exercise date (pre-existing,
#: T > 0): the BS barrier MC prices a knock-in as the European MC engine, whose lower-bound sanity check refuses its
#: one-day deep in-the-money put estimate (9.9940 against a 9.9943 bound with 256 antithetic paths); the Heston and
#: SLV barrier kernels read a discrete schedule from ``observation_dates`` only ("discrete monitoring requires
#: observation_dates")
_BEFORE_REFUSED = {("mc", "down_in_put")} | {(route, "up_in_call_discrete")
                                            for route in ("heston_mc", "heston_pde", "slv_mc", "slv_pde")}

BARRIER_CASES = ([(route, factory, state_ok, *product) for route, factory, state_ok in BARRIER_ROUTES
                  for product in BARRIER_PRODUCTS]
                 + [(route, factory, state_ok, *DOUBLE_PRODUCT) for route, factory, state_ok in DOUBLE_ROUTES])
BARRIER_SPOT_CASES = [(route, factory, state_ok, kind, build, hand, spot)
                      for route, factory, state_ok, kind, build, hand, spots in BARRIER_CASES for spot in spots]
_BARRIER_IDS = [f"{route}-{kind}-{spot:g}" for route, _, _, kind, _, _, spot in BARRIER_SPOT_CASES]


def _price(route, factory, product, spot, valuation=VALUATION, lifecycle_state=None):
    kwargs = {} if lifecycle_state is None else {"lifecycle_state": lifecycle_state}
    return _quiet(lambda: float(factory().price(product, env(spot, valuation, route), **kwargs)))


@pytest.mark.parametrize("route,factory,state_ok,kind,build,hand,spot", BARRIER_SPOT_CASES, ids=_BARRIER_IDS)
def test_a_carried_knock_in_settles_the_payoff_on_the_exercise_date(route, factory, state_ok, kind, build, hand, spot):
    """Carried on the product (``_otc_lifecycle_knocked_in``): dated == numeric == the payoff, never the rebate."""
    expected = hand(spot, True)
    for form in ("dated", "numeric"):
        assert _price(route, factory, build(form, True), spot) == pytest.approx(expected, rel=1e-12, abs=1e-12), form


@pytest.mark.parametrize("route,factory,state_ok,kind,build,hand,spot",
                         [c for c in BARRIER_SPOT_CASES if c[2]],
                         ids=[i for i, c in zip(_BARRIER_IDS, BARRIER_SPOT_CASES) if c[2]])
def test_a_knock_in_carried_by_the_lifecycle_state_settles_the_payoff(route, factory, state_ok, kind, build, hand,
                                                                      spot):
    """Carried by ``lifecycle_state.knocked_in`` (the product flag off), on the routes that take a lifecycle state."""
    expected = hand(spot, True)
    for form in ("dated", "numeric"):
        state = AutocallableLifecycleState(knocked_in=True)
        assert _price(route, factory, build(form, False), spot, lifecycle_state=state) == pytest.approx(
            expected, rel=1e-12, abs=1e-12), form


@pytest.mark.parametrize("route,factory,state_ok,kind,build,hand,spot", BARRIER_SPOT_CASES, ids=_BARRIER_IDS)
def test_a_knock_in_not_carried_is_decided_by_todays_observation(route, factory, state_ok, kind, build, hand, spot):
    """Unchanged: never knocked in, the rebate; hit today, the payoff."""
    expected = hand(spot, False)
    for form in ("dated", "numeric"):
        assert _price(route, factory, build(form, False), spot) == pytest.approx(expected, rel=1e-12, abs=1e-12), form


_BARRIER_BRANCH_CASES = [c for c in BARRIER_SPOT_CASES if c[5](c[6], True) != c[5](c[6], False)]


@pytest.mark.parametrize("route,factory,state_ok,kind,build,hand,spot", _BARRIER_BRANCH_CASES,
                         ids=[f"{c[0]}-{c[3]}-{c[6]:g}" for c in _BARRIER_BRANCH_CASES])
def test_a_carried_knock_in_the_day_before_is_on_the_settled_branch(route, factory, state_ok, kind, build, hand, spot):
    """The day before, the carried state prices the knocked-in contract: within one day's spot standard deviation of
    the knocked-in settled value, while the never-knocked-in value is further than that."""
    settled, other = hand(spot, True), hand(spot, False)
    assert abs(settled - other) > one_day_sd(spot)
    if (route, kind) in _BEFORE_REFUSED:
        pytest.skip("pre-existing T > 0 refusal, not the exercise date (see _BEFORE_REFUSED)")
    for form in ("dated", "numeric"):
        before = _price(route, factory, build(form, True, BEFORE), spot, valuation=BEFORE)
        assert abs(before - settled) <= one_day_sd(spot), (form, before, settled)


_PDE_GREEK_ROUTES = [r for r in BARRIER_ROUTES if r[0] in ("pde", "lv_pde", "heston_pde", "slv_pde")]


@pytest.mark.parametrize("route,factory", [(r, f) for r, f, _ in _PDE_GREEK_ROUTES],
                         ids=[r for r, _, _ in _PDE_GREEK_ROUTES])
def test_carried_knock_in_greeks_on_the_exercise_date_are_the_payoffs(route, factory):
    """The PDE routes' greeks at T = 0 (the core's maturity value): the payoff and its slope, gamma 0."""
    build = BARRIER_PRODUCTS[0][1]
    for form in ("dated", "numeric"):
        greeks = _quiet(lambda: factory().calculate_greeks(build(form, True), env(110.0, route=route)))
        assert float(greeks["price"]) == pytest.approx(10.0, rel=1e-12, abs=1e-12)
        assert float(greeks["delta"]) == pytest.approx(1.0, rel=1e-12, abs=1e-12)
        assert float(greeks["gamma"]) == 0.0


# ======================================================================================================================
# Part 2: autocallables with a knock-in term
# ======================================================================================================================

def _autocallable(kind):
    """The tracker-clock parity contracts (flat cash, so the two forms are one contract): snowball KO 103 (15%) /
    continuous KI 75; Phoenix KO 103 / coupon barrier 80 (1%) / continuous KI 70, memory or not; KO-reset ABSOLUTE
    (schedule 1 KO 103 / discrete KI 75 to month 6, schedule 2 KO 95 (3%) to month 12). A carried knock-in is set on
    the product (schedule 1, contract day 152)."""
    import pandas as pd
    from test_tracker_clock_parity import KINDS, phoenix

    builder = phoenix(annualized=False, memory=False) if kind == "phoenix_plain" else KINDS[kind]

    def build(form, carried, valuation=VALUATION):
        product = builder(form, pd.Timestamp(INITIAL))
        if carried:
            product._otc_lifecycle_knocked_in = True
            product._otc_lifecycle_knock_in_date = INITIAL + timedelta(days=152)
        return product if form == "dated" else _aged(product, valuation)
    return build


def _snowball_value(spot, knocked_in):
    """KO at 103 pays the 15% coupon (knock-out wins); knocked in (carried, or at the continuous 75 today) pays
    min(S - 100, 0); otherwise the 15% dividend coupon."""
    if spot >= 103.0:
        return 15.0
    if knocked_in or spot <= 75.0:
        return min(spot - 100.0, 0.0)
    return 15.0


def _phoenix_value(spot, knocked_in):
    """The 1% coupon at or above 80 (no unpaid coupon remembered); KO at 103 returns the principal; knocked in
    (carried, or at the continuous 70 today) adds min(S - 100, 0)."""
    coupon = 1.0 if spot >= 80.0 else 0.0
    if spot >= 103.0:
        return coupon
    if knocked_in or spot <= 70.0:
        return coupon + min(spot - 100.0, 0.0)
    return coupon


def _ko_reset_value(spot, knocked_in):
    """Knocked in on schedule 1, schedule 2 runs to the end: its last record knocks out at 95 and pays 3; below it
    the knocked-in terminal min(S - 100, 0). (Not knocked in, the contract matured with schedule 1: refused.)"""
    assert knocked_in
    return 3.0 if spot >= 95.0 else min(spot - 100.0, 0.0)


H, L = HESTON, SLV_LEVERAGE
_SNOWBALL_ROUTES = (
    ("mc", lambda: SnowballMCEngine(params=_mc())), ("pde", lambda: SnowballPDESolver(PDEParams())),
    ("quad_v1", lambda: SnowballQuadEngine(params=QuadParams(grid_points=1001))), ("quad_v2", SnowballQuadEngineV2),
    ("lv_mc", lambda: LocalVolSnowballMCEngine(params=_mc())), ("lv_pde", LocalVolSnowballPDESolver),
    ("heston_mc", lambda: HestonSnowballMCEngine(H, params=_mc())), ("qe_mc", lambda: QESnowballMCEngine(H, params=_mc())),
    ("heston_pde", lambda: HestonSnowballPDESolver(H, **_ADI)),
    ("slv_mc", lambda: HestonSLVSnowballMCEngine(H, params=_mc(), leverage_surface=L)),
    ("slv_qe_mc", lambda: HestonSLVQESnowballMCEngine(H, params=_mc(), leverage_surface=L)),
    ("slv_pde", lambda: HestonSLVSnowballPDESolver(H, L, **_ADI)),
)
_PHOENIX_ROUTES = (
    ("mc", lambda: PhoenixMCEngine(params=_mc())), ("pde", lambda: PhoenixPDESolver(PDEParams())),
    ("quad_v1", lambda: PhoenixQuadEngine(params=QuadParams(grid_points=1001))), ("quad_v2", PhoenixQuadEngineV2),
    ("lv_mc", lambda: LocalVolPhoenixMCEngine(params=_mc())), ("lv_pde", LocalVolPhoenixPDESolver),
    ("heston_mc", lambda: HestonPhoenixMCEngine(H, params=_mc())), ("qe_mc", lambda: QEPhoenixMCEngine(H, params=_mc())),
    ("heston_pde", lambda: HestonPhoenixPDESolver(H, **_ADI)),
    ("slv_mc", lambda: HestonSLVPhoenixMCEngine(H, params=_mc(), leverage_surface=L)),
    ("slv_qe_mc", lambda: HestonSLVQEPhoenixMCEngine(H, params=_mc(), leverage_surface=L)),
    ("slv_pde", lambda: HestonSLVPhoenixPDESolver(H, L, **_ADI)),
)
_KO_RESET_ROUTES = (
    ("mc", lambda: SnowballMCEngine(params=_mc())), ("pde", lambda: KOResetSnowballPDESolver(PDEParams())),
    ("quad_v1", lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001))),
    ("quad_v2", KOResetSnowballQuadEngineV2),
    ("lv_mc", lambda: LocalVolSnowballMCEngine(params=_mc())),
    ("heston_mc", lambda: HestonSnowballMCEngine(H, params=_mc())),
    ("slv_mc", lambda: HestonSLVSnowballMCEngine(H, params=_mc(), leverage_surface=L)),
)

#: (kind, routes, hand value, carried spots (between KI and KO, above and below the strike; KO), uncarried spots)
AUTOCALLABLES = (
    ("snowball", _SNOWBALL_ROUTES, _snowball_value, (101.0, 90.0, 104.0), (90.0, 74.0, 104.0)),
    ("phoenix", _PHOENIX_ROUTES, _phoenix_value, (101.0, 98.0, 90.0, 104.0), (90.0, 74.0, 60.0)),
    ("phoenix_plain", _PHOENIX_ROUTES, _phoenix_value, (101.0, 98.0, 90.0, 104.0), (90.0, 74.0, 60.0)),
    ("ko_reset_absolute", _KO_RESET_ROUTES, _ko_reset_value, (101.0, 90.0, 104.0), ()),
)

#: (kind, route) -> forms refused for a reason that is not the exercise date (pre-existing, at any valuation date)
AUTOCALL_REFUSED = {
    ("phoenix", "heston_pde"): ("dated", "numeric"),     # the 2D Phoenix PDE refuses memory coupons
    ("phoenix", "slv_pde"): ("dated", "numeric"),
    ("ko_reset_absolute", "quad_v2"): ("dated",),        # a dated KO-reset with past schedule-1 records
}
#: (kind, route) -> forms whose DAY-BEFORE price is refused for a reason that is not the exercise date (pre-existing,
#: T > 0): the 2D autocallable PDE kernels cannot resolve a dated knock-out schedule
AUTOCALL_BEFORE_REFUSED = {
    ("snowball", "heston_pde"): ("dated",), ("snowball", "slv_pde"): ("dated",),
    ("phoenix_plain", "heston_pde"): ("dated",), ("phoenix_plain", "slv_pde"): ("dated",),
    # a dated KO-reset with schedule-1 records in the past: settlement resolution of a past record (as QUAD V2's
    # refusal at any date, DATED_REFUSED_ANY_DATE in test_dated_expiry_settles); the exercise instant itself prices
    ("ko_reset_absolute", "pde"): ("dated",),
}


def _forms(kind, route, refused=AUTOCALL_REFUSED):
    return tuple(f for f in ("dated", "numeric") if f not in refused.get((kind, route), ()))


AUTOCALL_CASES = [(kind, route, factory, hand, spot, carried)
                  for kind, routes, hand, carried_spots, plain_spots in AUTOCALLABLES for route, factory in routes
                  if _forms(kind, route)
                  for carried, spots in ((True, carried_spots), (False, plain_spots)) for spot in spots]
_AUTOCALL_IDS = [f"{kind}-{route}-{'carried' if carried else 'plain'}-{spot:g}"
                 for kind, route, _, _, spot, carried in AUTOCALL_CASES]


@pytest.mark.parametrize("kind,route,factory,hand,spot,carried", AUTOCALL_CASES, ids=_AUTOCALL_IDS)
def test_an_autocallable_on_its_exercise_date_settles_its_knock_in_state(kind, route, factory, hand, spot, carried):
    """The carried knock-in first, then today's observation: both forms equal the hand settled value."""
    build = _autocallable(kind)
    for form in _forms(kind, route):
        assert _price(route, factory, build(form, carried), spot) == pytest.approx(hand(spot, carried), rel=1e-12,
                                                                            abs=1e-12), form


_V2 = [(kind, route, factory, hand, spot) for kind, route, factory, hand, spot, carried in AUTOCALL_CASES
       if route == "quad_v2" and carried]


@pytest.mark.parametrize("kind,route,factory,hand,spot", _V2, ids=[f"{c[0]}-{c[4]:g}" for c in _V2])
def test_quad_v2_settles_a_knock_in_carried_by_the_lifecycle_state(kind, route, factory, hand, spot):
    """QUAD V2 takes a lifecycle state: ``knocked_in=True`` there (the product flag off) settles the same."""
    build = _autocallable(kind)
    for form in _forms(kind, route):
        state = AutocallableLifecycleState(knocked_in=True)
        assert _price(route, factory, build(form, False), spot, lifecycle_state=state) == pytest.approx(
            hand(spot, True), rel=1e-12, abs=1e-12), form


_AUTOCALL_BRANCH = [c for c in AUTOCALL_CASES if c[5] and c[4] < 103.0]


@pytest.mark.parametrize("kind,route,factory,hand,spot,carried", _AUTOCALL_BRANCH,
                         ids=[f"{c[0]}-{c[1]}-{c[4]:g}" for c in _AUTOCALL_BRANCH])
def test_a_carried_autocallable_knock_in_the_day_before_is_on_the_settled_branch(kind, route, factory, hand, spot,
                                                                                 carried):
    """The day before, a carried knock-in prices within one day's spot standard deviation of the knocked-in settled
    value."""
    build = _autocallable(kind)
    settled = hand(spot, True)
    forms = [f for f in _forms(kind, route) if f not in AUTOCALL_BEFORE_REFUSED.get((kind, route), ())]
    assert forms
    for form in forms:
        before = _price(route, factory, build(form, True, BEFORE), spot, valuation=BEFORE)
        assert abs(before - settled) <= one_day_sd(spot), (form, before, settled)
