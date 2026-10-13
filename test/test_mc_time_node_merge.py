"""The autocallable MC time grid merges two spellings of one instant into one node (W17a, approved 2026-10-10).

The MC grids were a Python ``set`` of float times: a knock-out record's time and a continuous knock-in
``np.linspace`` node that are the same instant but one ulp apart (the dated ``(d - a) / 365`` against the numeric
``d / 365 - a / 365`` against ``T * j / n``) survived as two nodes 1e-16 years apart. The extra column re-dealt every
normal draw, so a dated contract and its aged numeric twin were priced on two independent samples (W17 diagnosis:
``docs/superpowers/specs/2026-10-10-snowball-mc-twin-diagnosis/``). ``merge_time_nodes`` collapses any run of nodes
closer than ``Tolerance.ZERO`` into its LAST member (maturity stays the exact final node) and ``node_index`` maps
each observation time onto it. A grid with no such run is returned bit-identical.

The twins: records on contract days ``d_k = round(k / 12 * 365)``; dated on ``INITIAL + d_k`` (exercise 365 days
on), numeric at ``d_k / 365`` (maturity 1.0) aged with ``time_shift(day / 365, keep_landing=True)``. The days are
those whose grid held such a run in one form or both (W17 ``grid_scan.txt``).
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
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
)
from quantark.asset.equity.engine.mc.term_inputs import make_df_fn
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.util.enum import ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.numerical import Tolerance, merge_time_nodes, node_index

INITIAL = datetime(2025, 1, 5)
DAYS = [round(k / 12 * 365) for k in range(1, 13)]
PRE, POST = DAYS[:6], DAYS[6:]
#: the days whose snowball / Phoenix MC grid held a sub-tolerance run in one form or both (W17 grid_scan.txt)
COLLISION_DAYS = (14, 41, 53, 61, 92, 170, 209)
#: the same for the continuous-knock-in KO-reset (alive to day 181)
KO_RESET_COLLISION_DAYS = (26, 65, 78, 82)
REL = 1e-10


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def on(day: int) -> datetime:
    return INITIAL + timedelta(days=day)


def env(spot: float, day: int, grid: bool = False):
    return _env(spot, on(day), grid=grid)


def _schedule(form, days, level):
    if form == "dated":
        return ObservationSchedule(records=[ObservationRecord(observation_date=on(d), barrier=level) for d in days])
    return ObservationSchedule(records=[ObservationRecord(observation_time=d / 365.0, barrier=level) for d in days])


def _expiry(product, form):
    product.initial_date = INITIAL
    if form == "dated":
        product.maturity = None
        product.exercise_date = on(365)


def _age(product, form, day):
    if form == "numeric":
        product.time_shift(day / 365.0, on(day), env(100.0, day), keep_landing=True)
    return product


def snowball(form, day):
    """KO 103 monthly (flat 15%), continuous KI 75 (W12's contract)."""
    p = create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15,
                                 is_annualized=False)
    _expiry(p, form)
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None,
                               ko_observation_schedule=_schedule(form, DAYS, 103.0))
    p.validate()
    return _age(p, form, day)


def phoenix(form, day):
    """Non-memory Phoenix: KO 103, 1% coupon at 80, continuous KI 70 (W12's contract)."""
    p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                coupon_rate=0.01, num_observations=12, memory_coupon=False, is_annualized=False)
    _expiry(p, form)
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None,
                               ko_observation_schedule=_schedule(form, DAYS, 103.0))
    p.validate()
    return _age(p, form, day)


def ko_reset(form, day):
    """W11's ABSOLUTE KO-reset (schedule 1: KO 103 flat 15% on the first six fixings; schedule 2: KO 95 flat 3% on
    the last six; 5% rebate, no principal) with a CONTINUOUS knock-in at 75 over schedule 1."""
    dated = form == "dated"
    p = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=None if dated else 1.0, initial_date=INITIAL,
        exercise_date=on(365) if dated else None,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(form, PRE, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(form, POST, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)
    p.validate()
    return _age(p, form, day)


def _mc(seed=7, paths=256):
    return MCParams(seed=seed, num_paths=paths)


#: (route, product builder, engine factory, reads the market IV grid)
ROUTES = {
    "bs_snowball": (snowball, lambda: SnowballMCEngine(params=_mc()), False),
    "bs_phoenix": (phoenix, lambda: PhoenixMCEngine(params=_mc()), False),
    "lv_snowball": (snowball, lambda: LocalVolSnowballMCEngine(params=_mc()), True),
    "heston_snowball": (snowball, lambda: HestonSnowballMCEngine(HESTON, params=_mc()), False),
    "slv_snowball": (snowball, lambda: HestonSLVSnowballMCEngine(HESTON, params=_mc(), leverage_surface=SLV_LEVERAGE),
                     True),
    "lv_phoenix": (phoenix, lambda: LocalVolPhoenixMCEngine(params=_mc()), True),
    "heston_phoenix": (phoenix, lambda: HestonPhoenixMCEngine(HESTON, params=_mc()), False),
    "slv_phoenix": (phoenix, lambda: HestonSLVPhoenixMCEngine(HESTON, params=_mc(), leverage_surface=SLV_LEVERAGE),
                    True),
}
TWIN_CASES = [(route, day) for route in ROUTES for day in COLLISION_DAYS]


def _price(build, factory, grid, form, day):
    return _quiet(lambda: float(factory().price(build(form, day), env(100.0, day, grid=grid))))


# ======================================================================================================================
# 1. The helper
# ======================================================================================================================

def test_merge_time_nodes_collapses_a_sub_tolerance_run_into_its_last_member():
    a = 0.49863013698630126                       # the day-92 linspace node and the numeric KO time, one ulp apart
    b = np.nextafter(a, 1.0)
    nodes = merge_time_nodes([0.25, b, a, 0.75, 0.25])
    assert nodes.tolist() == [0.25, b, 0.75]       # sorted, de-duplicated, the run kept as its LAST member
    np.testing.assert_array_equal(node_index(nodes, [a, b, 0.25, 0.75]), [1, 1, 0, 2])


def test_merge_time_nodes_tolerance_boundary():
    tol = Tolerance.ZERO
    assert merge_time_nodes([0.5, 0.5 + 0.5 * tol]).tolist() == [0.5 + 0.5 * tol]
    assert merge_time_nodes([0.5, 0.5 + 10.0 * tol]).tolist() == [0.5, 0.5 + 10.0 * tol]
    # a wider tolerance is honoured
    assert merge_time_nodes([0.5, 0.5 + 1e-6], tol=1e-5).tolist() == [0.5 + 1e-6]


def test_merge_time_nodes_chained_run_maps_every_member_to_the_last():
    tol = Tolerance.ZERO
    run = [0.3, 0.3 + 0.6 * tol, 0.3 + 1.2 * tol]   # each step below tol, the span above it
    nodes = merge_time_nodes([0.1] + run + [0.9])
    assert nodes.tolist() == [0.1, run[-1], 0.9]
    np.testing.assert_array_equal(node_index(nodes, run), [1, 1, 1])


def test_merge_time_nodes_keeps_maturity_as_the_final_node():
    T = 0.747945205479452
    above = np.nextafter(T, 1.0)                     # an observation spelled one ulp past maturity
    nodes = merge_time_nodes([0.2, above, T], pin=T)
    assert nodes[-1] == T and nodes.tolist() == [0.2, T]
    np.testing.assert_array_equal(node_index(nodes, [above, T]), [1, 1])
    below = np.nextafter(T, 0.0)                     # one ulp before maturity: keep-last already keeps T
    nodes = merge_time_nodes([0.2, below, T], pin=T)
    assert nodes.tolist() == [0.2, T]
    np.testing.assert_array_equal(node_index(nodes, [below, T]), [1, 1])
    # without the pin the run keeps its last member, the observation's spelling
    assert merge_time_nodes([0.2, above, T]).tolist() == [0.2, above]


def test_node_index_is_the_first_node_strictly_after_t_minus_tol():
    """A gap of exactly ``tol`` is two instants; each time maps to its own node (``at or after t - tol`` would map
    the later one onto the earlier). Dyadic values make the gap exact."""
    nodes = merge_time_nodes([0.5, 0.75], tol=0.25)
    assert nodes.tolist() == [0.5, 0.75]
    np.testing.assert_array_equal(node_index(nodes, [0.5, 0.75], tol=0.25), [0, 1])
    assert isinstance(node_index(nodes, 0.75, tol=0.25), int)


def test_merge_time_nodes_is_the_identity_without_a_run():
    rng = np.random.default_rng(3)
    times = list(rng.uniform(0.001, 1.0, 200)) + [0.5, 0.5, 1.0]
    legacy = np.array(sorted(set(times)))
    nodes = merge_time_nodes(times, pin=1.0)
    assert nodes.dtype == legacy.dtype and np.array_equal(nodes, legacy)
    assert nodes.tobytes() == legacy.tobytes()
    np.testing.assert_array_equal(node_index(nodes, times), [int(np.searchsorted(legacy, t)) for t in times])


# ======================================================================================================================
# 2. The grids: no sub-tolerance step, and bit-identical where nothing collided
# ======================================================================================================================

def _engine_grid(kind, form, day):
    e = env(100.0, day)
    eng = (PhoenixMCEngine if kind == "phoenix" else SnowballMCEngine)(params=_mc())
    eng._term_ctx = (e, 100.0)
    eng._df = make_df_fn(e)
    if kind == "ko_reset":
        p = ko_reset(form, day)
        T = p.get_max_maturity_time(e)
        g = eng._build_time_grid_ko_reset(p, e, T)
        return p, e, T, g["all_times"], g
    p = (phoenix if kind == "phoenix" else snowball)(form, day)
    T = p.get_maturity(e)
    out = eng._build_time_grid(p, e, T)
    return p, e, T, out[0], out


def _legacy_grid(kind, p, e, T):
    """The c02296cb construction: the sorted SET of the positive observation times, the continuous-KI linspace and T."""
    if kind == "ko_reset":
        pre = list(p.get_pre_ko_observation_profile(e)["observation_times"])
        post = list(p.get_post_ko_observation_profile(e)["observation_times"])
        horizon = float(p.get_pre_maturity_time(e))
        ki = list(np.linspace(0, horizon, int(e.bus_days_in_year * horizon) + 2)[1:])
        times = set(pre) | set(ki) | {T} | set(post)
    else:
        ko = list(p.get_ko_observation_profile(e)["observation_times"])
        ki = list(np.linspace(0, T, int(e.bus_days_in_year * T) + 2)[1:])
        times = set(ko) | set(ki) | {T}
    return np.array(sorted(t for t in times if t > 0.0), dtype=float)


GRID_CASES = ([("snowball", d) for d in COLLISION_DAYS] + [("phoenix", d) for d in COLLISION_DAYS]
              + [("ko_reset", d) for d in KO_RESET_COLLISION_DAYS])


@pytest.mark.parametrize("kind,day", GRID_CASES, ids=[f"{k}-d{d}" for k, d in GRID_CASES])
def test_collision_day_grids_hold_no_sub_tolerance_step_and_match_across_forms(kind, day):
    grids = {}
    for form in ("dated", "numeric"):
        _, _, T, at, _ = _quiet(lambda: _engine_grid(kind, form, day))
        steps = np.diff(np.concatenate([[0.0], at]))
        assert steps.min() >= Tolerance.ZERO, (form, steps.min())
        assert at[-1] == T
        grids[form] = at
    assert grids["dated"].size == grids["numeric"].size
    np.testing.assert_allclose(grids["dated"], grids["numeric"], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("kind,last", [("snowball", 364), ("phoenix", 364), ("ko_reset", 181)])
def test_grids_without_a_collision_are_bit_identical_to_the_set_construction(kind, last):
    """Every valuation day, both forms: where the c02296cb set construction held no sub-tolerance step, the grid
    is that construction byte for byte, and every observation index is the legacy ``searchsorted``."""
    checked = merged = 0
    for day in range(1, last + 1):
        for form in ("dated", "numeric"):
            p, e, T, at, out = _quiet(lambda: _engine_grid(kind, form, day))
            legacy = _legacy_grid(kind, p, e, T)
            if np.diff(np.concatenate([[0.0], legacy])).min() < Tolerance.ZERO:
                merged += 1
                assert at.size < legacy.size, (day, form)          # the run became one node
                continue
            checked += 1
            assert at.tobytes() == legacy.tobytes(), (day, form)
            if kind == "ko_reset":
                pre, pre_idx = np.asarray(out["pre_times"]), np.asarray(out["pre_indices"])
                np.testing.assert_array_equal(pre_idx[pre > 0.0], [int(np.searchsorted(legacy, t)) for t in pre
                                                                   if t > 0.0])
                assert np.all(pre_idx[pre <= 0.0] == -1)
            else:
                ko = p.get_ko_observation_profile(e)["observation_times"]
                legacy_idx = [int(np.searchsorted(legacy, t)) for t in ko if t > 0.0]
                ko_idx = np.asarray(out[2])
                if kind == "phoenix":                # Phoenix indices are path columns (node index + 1)
                    ko_idx = ko_idx - 1
                np.testing.assert_array_equal(ko_idx[ko_idx >= 0], legacy_idx)
    assert checked > 0 and merged > 0


# ======================================================================================================================
# 3. The twins on the collision days
# ======================================================================================================================

@pytest.mark.parametrize("route,day", TWIN_CASES, ids=[f"{r}-d{d}" for r, d in TWIN_CASES])
def test_dated_twin_equals_numeric_on_collision_days(route, day):
    build, factory, grid = ROUTES[route]
    dated = _price(build, factory, grid, "dated", day)
    numeric = _price(build, factory, grid, "numeric", day)
    assert np.isfinite(dated) and dated == pytest.approx(numeric, rel=REL), (dated, numeric)


@pytest.mark.parametrize("day", KO_RESET_COLLISION_DAYS)
def test_ko_reset_continuous_ki_twin_on_collision_days(day):
    """BS MC (the local-vol / Heston / SLV MC twins: ``test_ko_reset_vol_mc_continuous_ki.py``)."""
    dated = _price(ko_reset, lambda: SnowballMCEngine(params=_mc()), False, "dated", day)
    numeric = _price(ko_reset, lambda: SnowballMCEngine(params=_mc()), False, "numeric", day)
    assert np.isfinite(dated) and dated == pytest.approx(numeric, rel=REL), (dated, numeric)
