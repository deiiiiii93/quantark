"""A dated autocallable whose knock-in records lie in the past prices like its numeric twin (W11, 2026-10-10).

A dated knock-out-reset snowball with schedule-1 records before the valuation date was refused mid-life on every
engine that reads its knock-in schedule (``end_date (...) must be after start_date (...)``: a year fraction from the
valuation date to a past record's date), while the numeric twin, aged by ``time_shift`` (which drops the passed
records), priced. The knock-in resolution now keeps only the records not before the valuation date, as the knock-out
resolution already did (spec 3B decision 13): a past knock-in record is history, its outcome the carried state
(``_otc_lifecycle_knocked_in``), never re-resolved forward. The KO-reset contract rules are unchanged: a knock-in
replaces schedule 1 by schedule 2 for good, there is no knock-in test after schedule 1 ends, and a contract past
schedule 1 and not knocked in has matured (``require_alive`` refuses it on both forms).

The same resolver serves the plain snowball and (its own copy) the Phoenix: a dated one with a DISCRETE knock-in
schedule was refused the same way, and is pinned here too.

The twins: the dated contract's records sit on ``INITIAL + d`` for the snapped contract days ``d``; the numeric
twin's at ``d / 365`` years, aged to the valuation day with ``time_shift(keep_landing=True)``. The two forms reach the
engines as different floats (``d / 365 - n / 365`` against the day count ``(d - n) / 365``), so they agree to ulps of
the time grid, not bitwise (as ``test_tracker_clock_parity.PV_ATOL``).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Callable, Optional

import pytest

from test_dated_expiry_settles import HESTON, SLV_LEVERAGE, env as _env

from quantark.asset.equity.engine.mc import PhoenixMCEngine, SnowballMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
)
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.engine.pde import LocalVolPhoenixPDESolver, LocalVolSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad import (
    KOResetSnowballQuadEngine,
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngine,
    PhoenixQuadEngineV2,
    SnowballQuadEngine,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.util.enum import ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError

INITIAL = datetime(2025, 1, 5)
#: the twelve monthly fixings on snapped contract days ``round(k / 12 * 365)``: 30, 61, ..., 182 | 213, ..., 365
DAYS = [round(k / 12 * 365) for k in range(1, 13)]
PRE, POST = DAYS[:6], DAYS[6:]
SCHEDULE_1_END = PRE[-1]                                    # 182: schedule 1's last record (KO and KI)

#: the twins agree to ulps of the time grid (module docstring). Measured (W11, 2026-10-10) over every case here, price,
#: delta and gamma: worst |dated - numeric| 7.0e-13 (snowball PDE, day 292), 293 of 591 pairs bitwise equal.
ATOL = 1e-11


def on(day: int) -> datetime:
    return INITIAL + timedelta(days=day)


def env(spot: float, day: int, route: str = ""):
    """Flat 20% vol, 3% rate, 1% yield; the local-vol and SLV routes read it as a market IV grid."""
    return _env(spot, on(day), grid=route.startswith(("lv_", "slv_")))


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _schedule(form: str, days, level: float) -> ObservationSchedule:
    if form == "dated":
        return ObservationSchedule(records=[ObservationRecord(observation_date=on(d), barrier=level) for d in days])
    return ObservationSchedule(records=[ObservationRecord(observation_time=d / 365.0, barrier=level) for d in days])


def _carry(product, day: int, knocked_in_on: Optional[int]):
    """Age the numeric form to ``day`` (keeping a record landing on it); carry a knock-in observed on a past day."""
    if product.exercise_date is None:
        product.time_shift(day / 365.0, on(day), env(100.0, day), keep_landing=True)
    if knocked_in_on is not None:
        product._otc_lifecycle_knocked_in = True
        product._otc_lifecycle_knock_in_date = on(knocked_in_on)
    return product


def ko_reset(form: str, day: int, knocked_in_on: Optional[int] = None) -> KnockOutResetSnowballOption:
    """ABSOLUTE KO-reset: schedule 1 (KO 103, flat 15%) and the discrete KI 75 on the first six fixings; schedule 2
    (KO 95, flat 3%) on the last six; 5% rebate, no principal; one year."""
    dated = form == "dated"
    product = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=None if dated else 1.0, initial_date=INITIAL,
        exercise_date=on(365) if dated else None,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(form, PRE, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule(form, PRE, 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(form, POST, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)
    product.validate()
    return _carry(product, day, knocked_in_on)


def _discrete_ki(product, form: str, level: float):
    """Monthly KO 103 and a DISCRETE knock-in on the same twelve fixings."""
    product.barrier_config = replace(
        product.barrier_config, ko_observation_dates=None, ko_observation_schedule=_schedule(form, DAYS, 103.0),
        ki_observation_type=ObservationType.DISCRETE, ki_continuous=False, ki_observation_dates=None,
        ki_observation_schedule=_schedule(form, DAYS, level))
    product.initial_date = INITIAL
    if form == "dated":
        product.maturity = None
        product.exercise_date = on(365)
    product.validate()
    return product


def snowball(form: str, day: int, knocked_in_on: Optional[int] = None):
    product = create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15,
                                       is_annualized=False)
    return _carry(_discrete_ki(product, form, 75.0), day, knocked_in_on)


def phoenix(form: str, day: int, knocked_in_on: Optional[int] = None):
    product = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                      coupon_rate=0.01, num_observations=12, memory_coupon=False,
                                      is_annualized=False)
    return _carry(_discrete_ki(product, form, 70.0), day, knocked_in_on)


def _mc():
    return MCParams(seed=7, num_paths=256, use_antithetic=True)


H, L = HESTON, SLV_LEVERAGE
KO_RESET_ROUTES = (
    ("mc", lambda: SnowballMCEngine(params=_mc())), ("pde", lambda: KOResetSnowballPDESolver(PDEParams())),
    ("quad_v1", lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001))),
    ("quad_v2", KOResetSnowballQuadEngineV2),
    ("lv_mc", lambda: LocalVolSnowballMCEngine(params=_mc())),
    ("heston_mc", lambda: HestonSnowballMCEngine(H, params=_mc())),
    ("slv_mc", lambda: HestonSLVSnowballMCEngine(H, params=_mc(), leverage_surface=L)),
)
SNOWBALL_ROUTES = (
    ("mc", lambda: SnowballMCEngine(params=_mc())), ("pde", lambda: SnowballPDESolver(PDEParams())),
    ("quad_v1", lambda: SnowballQuadEngine(params=QuadParams(grid_points=1001))), ("quad_v2", SnowballQuadEngineV2),
    ("lv_mc", lambda: LocalVolSnowballMCEngine(params=_mc())), ("lv_pde", LocalVolSnowballPDESolver),
    ("heston_mc", lambda: HestonSnowballMCEngine(H, params=_mc())), ("qe_mc", lambda: QESnowballMCEngine(H, params=_mc())),
    ("slv_mc", lambda: HestonSLVSnowballMCEngine(H, params=_mc(), leverage_surface=L)),
)
PHOENIX_ROUTES = (
    ("mc", lambda: PhoenixMCEngine(params=_mc())), ("pde", lambda: PhoenixPDESolver(PDEParams())),
    ("quad_v1", lambda: PhoenixQuadEngine(params=QuadParams(grid_points=1001))), ("quad_v2", PhoenixQuadEngineV2),
    ("lv_mc", lambda: LocalVolPhoenixMCEngine(params=_mc())), ("lv_pde", LocalVolPhoenixPDESolver),
    ("heston_mc", lambda: HestonPhoenixMCEngine(H, params=_mc())),
    ("slv_mc", lambda: HestonSLVPhoenixMCEngine(H, params=_mc(), leverage_surface=L)),
)
#: Not here: the 2D Heston / SLV snowball and Phoenix PDE refuse a DATED knock-out schedule before maturity ("a
#: date-based knock-out observation schedule needs a pricing environment ..."), pre-existing and not a knock-in record
#: (carried-knock-in report section 6); their numeric twins price.

#: (valuation day, knock-in day or None, spot). Not knocked in: on schedule-1 records (a knock-out and a knock-in
#: decided at the instant), between them, and on schedule 1's last record. Knocked in (carried from a past record):
#: mid schedule 1, on its last record, the day after it ends, on a schedule-2 record, mid schedule 2, the day before
#: the exercise date and on it.
KO_RESET_STATES = (
    (30, None, 100.0), (30, None, 104.0), (100, None, 100.0), (100, None, 80.0), (152, None, 70.0),
    (SCHEDULE_1_END, None, 100.0),
    (100, 91, 100.0), (152, 122, 90.0), (SCHEDULE_1_END, 152, 100.0), (SCHEDULE_1_END + 1, 152, 100.0),
    (SCHEDULE_1_END + 1, 152, 90.0), (213, 152, 96.0), (292, 152, 100.0), (364, 152, 90.0), (365, 152, 100.0),
    (365, 152, 90.0),
)
#: plain snowball / Phoenix (discrete knock-in on all twelve fixings): alive throughout either way
AUTOCALL_STATES = ((100, None, 100.0), (152, None, 72.0), (292, None, 90.0), (292, 152, 100.0), (365, 152, 90.0))


def _ids(cases):
    return [f"{r}-d{day}-{'ki' + str(ki) if ki is not None else 'alive'}-{spot:g}" for r, _, day, ki, spot in cases]


KO_RESET_CASES = [(r, f, day, ki, spot) for r, f in KO_RESET_ROUTES for day, ki, spot in KO_RESET_STATES]


def _greeks(factory, build, day, ki, spot, route):
    return _quiet(lambda: factory().calculate_greeks(build(day, ki), env(spot, day, route)))


def _assert_twins(factory, build, day, ki, spot, route):
    dated = _greeks(factory, lambda d, k: build("dated", d, k), day, ki, spot, route)
    numeric = _greeks(factory, lambda d, k: build("numeric", d, k), day, ki, spot, route)
    for name in ("price", "delta", "gamma"):
        assert float(dated[name]) == pytest.approx(float(numeric[name]), rel=0.0, abs=ATOL), name
    price = _quiet(lambda: float(factory().price(build("dated", day, ki), env(spot, day, route))))
    assert price == pytest.approx(float(numeric["price"]), rel=0.0, abs=ATOL)


@pytest.mark.parametrize("route,factory,day,ki,spot", KO_RESET_CASES, ids=_ids(KO_RESET_CASES))
def test_a_dated_ko_reset_with_past_records_prices_like_its_numeric_twin(route, factory, day, ki, spot):
    """Price, delta and gamma of the dated contract equal the numeric twin's, mid-life and on the exercise date."""
    _assert_twins(factory, ko_reset, day, ki, spot, route)


_DEAD = [(r, f, day) for r, f in KO_RESET_ROUTES for day in (SCHEDULE_1_END + 1, 292)]


@pytest.mark.parametrize("route,factory,day", _DEAD, ids=[f"{r}-d{d}" for r, _, d in _DEAD])
def test_past_schedule_1_and_not_knocked_in_is_refused_on_both_forms(route, factory, day):
    """Not knocked in once schedule 1 has ended, the contract matured with it (contract rule): both forms refuse,
    for that reason and not a settlement resolution of a past record."""
    for form in ("dated", "numeric"):
        with pytest.raises(ValidationError, match="pre-KI schedule has ended"):
            _quiet(lambda: factory().price(ko_reset(form, day), env(100.0, day, route)))


AUTOCALL_CASES = ([("snowball", r, f, day, ki, spot) for r, f in SNOWBALL_ROUTES for day, ki, spot in AUTOCALL_STATES]
                  + [("phoenix", r, f, day, ki, spot) for r, f in PHOENIX_ROUTES for day, ki, spot in AUTOCALL_STATES])


@pytest.mark.parametrize("kind,route,factory,day,ki,spot", AUTOCALL_CASES,
                         ids=[f"{c[0]}-{i}" for c, i in zip(AUTOCALL_CASES, _ids([c[1:] for c in AUTOCALL_CASES]))])
def test_a_dated_autocallable_with_past_knock_in_records_prices_like_its_numeric_twin(kind, route, factory, day, ki,
                                                                                     spot):
    """The snowball and Phoenix resolve their discrete knock-in records the same way: past ones are history."""
    _assert_twins(factory, snowball if kind == "snowball" else phoenix, day, ki, spot, route)


@pytest.mark.parametrize("build", [ko_reset, snowball, phoenix], ids=["ko_reset", "snowball", "phoenix"])
def test_the_knock_in_resolution_keeps_only_records_not_before_the_valuation_date(build):
    """On a record day the record lands at time 0 (still pending); every earlier one is dropped, as the twin's
    ``time_shift`` drops it, and the kept records keep their source positions in the contract's schedule."""
    day = 91
    for form in ("dated", "numeric"):
        product = build(form, day)
        e = env(100.0, day)
        times = [rec.observation_time for rec in product.resolve_ki_observations(e)]
        assert times[0] == 0.0
        assert len(times) == len([d for d in (PRE if build is ko_reset else DAYS) if d >= day])
    dated = build("dated", day)
    kept = dated.active_ki_source_indices(env(100.0, day))
    assert kept == [i for i, d in enumerate(PRE if build is ko_reset else DAYS) if d >= day]
