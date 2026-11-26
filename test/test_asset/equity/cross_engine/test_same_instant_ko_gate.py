"""Cross-engine gate for records on one env instant (0.16.0 DCN/dated run, T5b: spec AC2/AC3/AC6/AC7, D14).

A dated note on a US BUSINESS_DAYS env with PRECEDING holiday adjustment: 2025-07-04 is a US holiday, so its record
sits on 2025-07-03's env instant (env time 0.5). D1/D14 zero-gap rule: the records of one instant are consecutive
observations with zero elapsed time, at one spot, in contract order (a Snowball's dominated knock-out drops out; a
Phoenix record is its own coupon date; a KO-reset instant is every schedule-1 knock-out, then the knock-in, then every
schedule-2 knock-out). Desk ruling: a knock-out beats a same-instant discrete knock-in everywhere -- under
``disable_ko_after_ki`` the knock-in at the instant does NOT disable a knock-out at that instant (the flag reads the
state carried INTO the observation).

One matrix, product x engine, every engine that supports the product:

* exact references (AC2 Snowball pair, AC7 Phoenix pair, the Phoenix EXPIRY pair): price and per-record knock-out
  (and coupon) probabilities against the closed form -- MC within 3 SE, PDE rel 1e-3, QUAD V1 / V2 rel 1e-6;
* the repro products (Snowball step-down / duplicate, Phoenix duplicate / step-down, KO-reset schedule-1 step-down /
  knock-in pair) and the ``disable_ko_after_ki`` products: price within 3 SE + 1e-3 relative of the same model's MC
  (AC3/AC7), and every per-record statistics row -- knock-out, coupon, discrete knock-in, P(knock in before any
  knock-out) -- within 3 binomial SE of MC (AC6/AC7).

The local-vol engines price on a flat grid surface (the Black-Scholes model) against Black-Scholes MC; the Heston
engines against Heston MC. MC runs once per (product, model) per worker (module-scoped cache): ``price`` and the
statistics are one run (``calculate_event_stats`` runs the engine's own ``price``; its ``pv`` and SE are the price's).

Products and exact references are the per-engine tests' (``engine_pde/test_same_instant_ko.py``,
``engine_pde/test_same_instant_phoenix_ko_reset.py``, ``engine_quad/test_same_instant_ko_quad.py``).
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import pytest

from test_asset.equity.engine_pde.test_same_instant_ko import (
    HESTON,
    HESTON_GRID,
    JUL3,
    JUL4,
    REPRO_DATES,
    _exact_pair_value,
    env,
    pair,
    repro,
    snowball,
)
from test_asset.equity.engine_pde.test_same_instant_phoenix_ko_reset import _exact_expiry_pair, _expiry_pair
from test_asset.equity.engine_quad.test_same_instant_ko_quad import (
    GRID_EXACT,
    GRID_PHX_EXACT,
    KOR_CASES,
    _exact_phoenix_pair,
    ko_reset,
    phoenix,
    phoenix_pair,
    phoenix_repro,
)
from quantark.asset.equity.engine.mc import (
    HestonPhoenixMCEngine,
    HestonSnowballMCEngine,
    PhoenixMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
    PhoenixPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import (
    HestonKOResetSnowballPDESolver,
    LocalVolKOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver, LocalVolPhoenixPDESolver
from quantark.asset.equity.engine.quad import PhoenixQuadEngine, SnowballQuadEngine
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2.engine import (
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError

MC_PATHS = 400_000
#: the Heston statistics grids: on the repro grids the rows BEFORE the shared instant sit 3-10 SE off Heston MC with
#: one record too (the route's own discretization there, ``engine_pde/test_same_instant_ko.py`` /
#: ``test_same_instant_ki.py``); the finer grids bring every row in
HESTON_STATS_GRID = dict(n_x=128, n_v=32, n_t=128)


# --- products -------------------------------------------------------------------------------------------------------

#: the dates of the repro schedules other than the shared instant: 03-03, 05-05, 09-03, 11-03
_OTHERS = [d for d in REPRO_DATES if d not in (JUL3, JUL4)]


def _records(items):
    return [ObservationRecord(observation_date=it[0], barrier=it[1], return_rate=it[2] if len(it) > 2 else None)
            for it in items]


def _tie_layout(layout, level=90.0):
    """``(knock-out items, knock-in items)`` of a ``disable_ko_after_ki`` tie at the 07-03 / 07-04 instant: a 90
    knock-out (10%) against a 95 discrete knock-in -- the [90, 95) band of the instant's spot breaches both. The ruling
    knocks that band out (the knock-in at the instant does not disable the knock-out at the instant); the other
    reading (the 07-03 knock-in, first in contract order, disables the 07-04 knock-out) prices the band as knocked in
    (``test_the_tie_band_tells_the_two_readings_apart``). Every other date: knock-out 103, knock-in 75."""
    head_ko, tail_ko = [(d, 103.0) for d in _OTHERS[:2]], [(d, 103.0) for d in _OTHERS[2:]]
    head_ki, tail_ki = [(d, 75.0) for d in _OTHERS[:2]], [(d, 75.0) for d in _OTHERS[2:]]
    ko_at, ki_at = {"ki_then_ko": ([(JUL4, level, 0.10)], [(JUL3, 95.0)]),     # the knock-in first in contract order
                    "ko_then_ki": ([(JUL3, level, 0.10)], [(JUL4, 95.0)]),
                    "same_date": ([(JUL3, level, 0.10)], [(JUL3, 95.0)]),
                    # a knock-out pair (103 on 07-03, the step-down on 07-04) and the knock-in on 07-03
                    "ko_pair_ki": ([(JUL3, 103.0), (JUL4, level, 0.10)], [(JUL3, 95.0)])}[layout]
    return head_ko + ko_at + tail_ko, head_ki + ki_at + tail_ki


def _discrete_knock_in(p, items, disable):
    p.barrier_config = replace(p.barrier_config, ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
                               ki_barrier=items[0][1], disable_ko_after_ki=disable,
                               ki_observation_schedule=ObservationSchedule(records=_records(items)))
    p.validate()
    return p


def tie_snowball(layout, disable=True, level=90.0):
    ko, ki = _tie_layout(layout, level)
    return _discrete_knock_in(snowball(_records(ko)), ki, disable)


def tie_phoenix(layout, disable=True, level=90.0):
    ko, ki = _tie_layout(layout, level)
    return _discrete_knock_in(phoenix(_records(ko)), ki, disable)


def tie_ko_reset(layout, disable=True, level=90.0):
    """The KO-reset repro's schedule 1 / knock-in (dates up to the instant) with the tie; schedule 2 from September."""
    ko, ki = _tie_layout(layout, level)
    p = ko_reset(pre=[it for it in ko if it[0] <= JUL4], ki=[it for it in ki if it[0] <= JUL4])
    p.barrier_config = replace(p.barrier_config, disable_ko_after_ki=disable)
    p.validate()
    return p


TIE_LAYOUTS = ("ki_then_ko", "ko_then_ki", "same_date", "ko_pair_ki")
#: name: (family, product)
PRODUCTS = {
    "snowball_step_down": ("snowball", lambda: repro()),
    "snowball_duplicate": ("snowball", lambda: repro(103.0, 0.40)),
    "phoenix_duplicate": ("phoenix", lambda: phoenix_repro("duplicate")),
    "phoenix_step_down": ("phoenix", lambda: phoenix_repro("stepdown")),
    "ko_reset_schedule1_step_down": ("ko_reset", lambda: ko_reset(**KOR_CASES["schedule1_step_down"])),
    "ko_reset_ki_pair": ("ko_reset", lambda: ko_reset(**KOR_CASES["ki_pair"])),
    **{f"{family}_disable_{layout}": (family, (lambda m=make, x=layout: m(x)))
       for family, make in (("snowball", tie_snowball), ("phoenix", tie_phoenix), ("ko_reset", tie_ko_reset))
       for layout in TIE_LAYOUTS},
}


# --- engines --------------------------------------------------------------------------------------------------------

def _v1(cls, grid_points=1001):
    return lambda: cls(params=QuadParams(grid_points=grid_points))


def _grid_env():
    return env(grid=True)


#: family: engine: (price engine, statistics engine (None: the price engine), MC model, env)
ENGINES = {
    "snowball": {
        "pde": (lambda: SnowballPDESolver(PDEParams()), None, "bs", env),
        "lv_pde": (lambda: LocalVolSnowballPDESolver(PDEParams()), None, "bs", _grid_env),
        "heston_pde": (lambda: HestonSnowballPDESolver(HESTON, **HESTON_GRID),
                       lambda: HestonSnowballPDESolver(HESTON, **HESTON_STATS_GRID), "heston", env),
        "v1": (_v1(SnowballQuadEngine), None, "bs", env),
        "v2": (SnowballQuadEngineV2, None, "bs", env),
    },
    "phoenix": {
        "pde": (lambda: PhoenixPDESolver(PDEParams()), None, "bs", env),
        "lv_pde": (lambda: LocalVolPhoenixPDESolver(PDEParams()), None, "bs", _grid_env),
        "heston_pde": (lambda: HestonPhoenixPDESolver(HESTON, n_x=128, n_v=12, n_t=48),
                       lambda: HestonPhoenixPDESolver(HESTON, **HESTON_STATS_GRID), "heston", env),
        "v1": (_v1(PhoenixQuadEngine), None, "bs", env),
        "v2": (PhoenixQuadEngineV2, None, "bs", env),
    },
    "ko_reset": {
        "pde": (lambda: KOResetSnowballPDESolver(PDEParams()), None, "bs", env),
        "lv_pde": (lambda: LocalVolKOResetSnowballPDESolver(PDEParams()), None, "bs", _grid_env),
        "heston_pde": (lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=96, n_v=16, n_t=64),
                       lambda: HestonKOResetSnowballPDESolver(HESTON, **HESTON_STATS_GRID), "heston", env),
        "v1": (_v1(KOResetSnowballQuadEngine), None, "bs", env),
        "v2": (KOResetSnowballQuadEngineV2, None, "bs", env),
    },
}
_MC = {"snowball": (SnowballMCEngine, HestonSnowballMCEngine), "phoenix": (PhoenixMCEngine, HestonPhoenixMCEngine),
       "ko_reset": (SnowballMCEngine, HestonSnowballMCEngine)}
#: (product, engine) cells the engine refuses: QUAD V2 does not support two KO-reset knock-out records of one kind on
#: one instant ("share a canonical time"; a knock-in pair merges into one knock-in, T5a)
REFUSED = {("ko_reset_schedule1_step_down", "v2"), ("ko_reset_disable_ko_pair_ki", "v2")}
CASES = [(p, e) for p, (family, _) in PRODUCTS.items() for e in ENGINES[family] if (p, e) not in REFUSED]
IDS = [f"{p}-{e}" for p, e in CASES]


def _mc_engine(family, model):
    bs, heston = _MC[family]
    if model == "bs":
        return bs(params=MCParams(num_paths=MC_PATHS, seed=7))
    return heston(HESTON, params=MCParams(num_paths=MC_PATHS, seed=7))


@pytest.fixture(scope="module")
def mc_reference():
    """``get(product, model) -> (price, SE, statistics)``: one MC run per (product, model), cached for the module."""
    cache = {}

    def get(name, model):
        if (name, model) not in cache:
            family, make = PRODUCTS[name]
            engine = _mc_engine(family, model)
            stats = engine.calculate_event_stats(make(), env())
            cache[(name, model)] = (float(stats.pv), float(engine.get_last_std_error()), stats)
        return cache[(name, model)]

    return get


# --- comparisons ----------------------------------------------------------------------------------------------------

def _binomial_sd(m):
    return math.sqrt(max(m * (1.0 - m), 0.0) / MC_PATHS)


def row_z(rows, mc_rows):
    """Per-row ``(engine - MC) / binomial SD`` (0 where both are exactly 0)."""
    assert len(rows) == len(mc_rows), (rows, mc_rows)
    out = []
    for p, m in zip(rows, mc_rows):
        sd = _binomial_sd(m)
        out.append(0.0 if p == m else ((p - m) / sd if sd > 0.0 else math.inf))
    return out


def _has_discrete_knock_in(product):
    bc = product.barrier_config
    return (bc.ki_observation_type == ObservationType.DISCRETE and not bc.ki_continuous
            and bc.ki_observation_schedule is not None)


def statistics_rows(stats, product):
    """The per-record rows the gate compares: knock-out, coupon (Phoenix), discrete knock-in, and the total
    P(knock in before any knock-out)."""
    out = {"ko": [float(p) for p in stats.observations.probability]}
    if stats.coupons is not None and stats.coupons.probability is not None:
        out["coupon"] = [float(p) for p in stats.coupons.probability]
    if _has_discrete_knock_in(product) and stats.ki_table is not None:
        out["ki"] = [float(p) for p in stats.ki_table.probability]
    if stats.knock_in is not None and stats.knock_in.probability is not None:
        out["ki_total"] = [float(stats.knock_in.probability)]
    return out


def compared_z(stats, mc, product, engine):
    """``{row kind: per-row z against MC}``. The Heston PDE's discrete knock-in rows are compared from the shared
    instant on: its rows BEFORE the instant (03-03 / 05-05) sit 3-10 SE off Heston MC with one record too, at the
    statistics grid as well -- the route's own discretization there, not the instant's (T5a,
    ``engine_pde/test_same_instant_ki.py``); the total P(knock in) is compared in full."""
    rows, mc_rows = statistics_rows(stats, product), statistics_rows(mc, product)
    assert rows.keys() == mc_rows.keys(), (rows.keys(), mc_rows.keys())
    if engine == "heston_pde" and "ki" in rows:
        dates = list(stats.ki_table.dates)
        assert dates == list(mc.ki_table.dates)
        k = min(i for i, d in enumerate(dates) if d in (JUL3, JUL4))
        rows["ki"], mc_rows["ki"] = rows["ki"][k:], mc_rows["ki"][k:]
    return {kind: row_z(rows[kind], mc_rows[kind]) for kind in rows}


# --- the repro and tie products against MC --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("name,engine", CASES, ids=IDS)
def test_price_agrees_with_mc(mc_reference, name, engine):
    """AC3 / AC7 / the tie ruling: within 3 SE + 1e-3 relative of the same model's MC."""
    family, make = PRODUCTS[name]
    price_engine, _, model, make_env = ENGINES[family][engine]
    ref, se, _ = mc_reference(name, model)
    value = price_engine().price(make(), make_env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se, (value - ref) / se)


@pytest.mark.slow
@pytest.mark.parametrize("name,engine", CASES, ids=IDS)
def test_statistics_agree_with_mc(mc_reference, name, engine):
    """AC6 / AC7: every per-record row (knock-out, coupon, discrete knock-in) and P(knock in before any knock-out)
    within 3 binomial SE of the same model's MC; the engine's statistics reconcile with its price."""
    family, make = PRODUCTS[name]
    price_engine, stats_engine, model, make_env = ENGINES[family][engine]
    product = make()
    stats = (stats_engine or price_engine)().calculate_event_stats(product, make_env())
    _, _, mc = mc_reference(name, model)
    for kind, z in compared_z(stats, mc, product, engine).items():
        assert max(abs(x) for x in z) <= 3.0, (kind, z)
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))


@pytest.mark.parametrize("name,engine", sorted(REFUSED), ids=[f"{p}-{e}" for p, e in sorted(REFUSED)])
def test_unsupported_cells_refuse(name, engine):
    """QUAD V2 keeps its refusal of two KO-reset knock-out records of one kind on one instant."""
    family, make = PRODUCTS[name]
    price_engine, _, _, make_env = ENGINES[family][engine]
    with pytest.raises(ValidationError, match="share a canonical time"):
        price_engine().price(make(), make_env())


@pytest.mark.slow
@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
def test_the_tie_band_tells_the_two_readings_apart(mc_reference, family):
    """The tie products discriminate: the other reading of the instant (the 07-03 knock-in disables the 07-04
    knock-out) is the product whose knock-out sits at the knock-in level (95: the band below it knocks in, the band
    above knocks out) -- on MC, far more than the gate's tolerance from the ruling's price. And the ruling itself does
    not depend on which date of the instant each record carries (``ko_then_ki`` and ``same_date``: one MC price)."""
    make = {"snowball": tie_snowball, "phoenix": tie_phoenix, "ko_reset": tie_ko_reset}[family]
    ruling, se, _ = mc_reference(f"{family}_disable_ki_then_ko", "bs")
    alternative = _mc_engine(family, "bs").price(make("ki_then_ko", level=95.0), env())
    assert abs(alternative - ruling) > 20.0 * se, (ruling, alternative, se)
    assert mc_reference(f"{family}_disable_ko_then_ki", "bs")[0] == mc_reference(f"{family}_disable_same_date",
                                                                                 "bs")[0]


# --- exact references -----------------------------------------------------------------------------------------------

def _exact_snowball(reverse):
    product = snowball(pair(97.0, 100.0) if reverse else pair(), ki=None, reverse=reverse)
    value, ko = _exact_pair_value(product, env(), reverse)
    return product, value, {"ko": list(ko)}


def _exact_phoenix(b1, b2):
    product = phoenix_pair(b1, b2)
    value, ko, coupon = _exact_phoenix_pair(product, env())
    return product, value, {"ko": list(ko), "coupon": list(coupon)}


def _exact_expiry(b1, b2, memory):
    product = _expiry_pair(b1, b2, memory)
    _, ko, coupon = _exact_phoenix_pair(product, env())
    return product, _exact_expiry_pair(product, env(), memory), {"ko": list(ko), "coupon": list(coupon)}


#: name: (family, () -> (product, exact price, exact per-record rows))
EXACT = {
    "snowball_pair_up": ("snowball", lambda: _exact_snowball(False)),
    "snowball_pair_reverse": ("snowball", lambda: _exact_snowball(True)),
    "phoenix_pair_b2_below_b1": ("phoenix", lambda: _exact_phoenix(103.0, 100.0)),
    "phoenix_pair_b2_above_b1": ("phoenix", lambda: _exact_phoenix(100.0, 103.0)),
    **{f"phoenix_expiry_pair_{tag}_{'memory' if memory else 'no_memory'}": (
        "phoenix", (lambda a=b1, b=b2, m=memory: _exact_expiry(a, b, m)))
       for tag, (b1, b2) in (("b2_below_b1", (103.0, 100.0)), ("b2_above_b1", (100.0, 103.0)))
       for memory in (True, False)},
}
#: engine: (rel tolerance on the price, abs tolerance on the probabilities); MC: 3 SE / 3 binomial SD
EXACT_TOLERANCE = {"pde": (1e-3, 1e-3), "lv_pde": (1e-3, 1e-3), "v1": (1e-6, 1e-6), "v2": (1e-6, 1e-8)}


def _exact_engine(family, engine):
    if engine == "mc":
        return _mc_engine(family, "bs"), env
    if engine == "v1":
        # the exact-reference lattices: V1's own floor at 1001 nodes is ~1e-5 on one record
        cls = SnowballQuadEngine if family == "snowball" else PhoenixQuadEngine
        return _v1(cls, GRID_EXACT if family == "snowball" else GRID_PHX_EXACT)(), env
    price_engine, _, _, make_env = ENGINES[family][engine]
    return price_engine(), make_env


EXACT_CASES = [(n, e) for n in EXACT for e in ("mc", "pde", "lv_pde", "v1", "v2")]


@pytest.mark.parametrize("name,engine", EXACT_CASES, ids=[f"{n}-{e}" for n, e in EXACT_CASES])
def test_matches_the_exact_reference(name, engine):
    """AC2 / AC7 (and the EXPIRY pair: record 1's coupon is paid at the note's end, record 2's settlement when record 2
    knocks out at the same instant): price and each record's knock-out / coupon probability against the closed form.
    MC within 3 SE (3 binomial SD per row); PDE rel 1e-3 (abs 1e-3); QUAD V1 at the exact lattices and QUAD V2 rel 1e-6
    (V2 abs 1e-8, V1 abs 1e-6 on the rows); the statistics reconcile."""
    family, build = EXACT[name]
    product, ref, exact_rows = build()
    engine_obj, make_env = _exact_engine(family, engine)
    stats = engine_obj.calculate_event_stats(product, make_env())
    rows = {kind: [float(p) for p in getattr(stats, table).probability]
            for kind, table in (("ko", "observations"), ("coupon", "coupons")) if kind in exact_rows}
    if engine == "mc":
        se = engine_obj.get_last_std_error()
        assert abs(stats.pv - ref) <= 3.0 * se, (stats.pv, ref, se, (stats.pv - ref) / se)
        for kind, exact in exact_rows.items():
            for p, m in zip(rows[kind], exact):
                assert abs(p - m) <= 3.0 * _binomial_sd(m) + 1e-12, (kind, rows[kind], exact)
        return
    rel, prob = EXACT_TOLERANCE[engine]
    value = engine_obj.price(product, make_env())
    assert abs(value / ref - 1.0) <= rel, (value, ref, value / ref - 1.0)
    assert stats.pv == pytest.approx(value, rel=1e-12)
    for kind, exact in exact_rows.items():
        assert rows[kind] == pytest.approx(exact, abs=prob), (kind, rows[kind], exact)
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))
