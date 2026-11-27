"""Discrete knock-in records on one env instant, every engine (0.16.0 DCN/dated run, T5a: zero-gap rule D14).

A dated note on a US BUSINESS_DAYS env with PRECEDING holiday adjustment: 2025-07-04 is a US holiday, so a knock-in
record on it sits on 2025-07-03's env instant. Ruling (run 4, T5): the records of one instant are consecutive
observations with zero elapsed time at one spot, so a discrete knock-in pair is ONE knock-in test at the widest level
(a knock-in is idempotent at one spot), in either contract order. The knock-in statistics keep one row per contract
record: the instant's first knock-in mass sits on the first record in contract order whose level the spot breaches
(``same_instant_ki_groups``), later records of the instant carry what their level adds -- a record inside an earlier
one's region carries 0. The total knock-in probability is the widest single record's.

Before T5a: the PDE (Snowball through the core adapter) staged every record (a duplicate drifted: two cell averages
on one level are not idempotent), the Phoenix PDE kept the LAST record's level, QUAD V1 the FIRST, QUAD V2 refused,
and every engine's knock-in table refused the repeated time (``KnockInTable.times has duplicate rows``) -- MC
included.

Products: the run's Snowball repro (KO 103 on 03-03 / 05-05 / 07-03 / 09-03 / 11-03, discrete KI 75), the Phoenix
repro (KI 70) and the KO-reset repro (schedule-1 KI 75), each with the 07-03 knock-in record at one level and a 07-04
record at another.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from datetime import datetime

import pytest

from test_asset.equity.engine_pde.test_same_instant_ko import HESTON, HESTON_GRID, JUL3, JUL4, REPRO_DATES, env
from test_asset.equity.engine_pde.test_same_instant_ko import repro, snowball
from test_asset.equity.engine_quad.test_same_instant_ko_quad import KOR_PRE, PHX_DATES, ko_reset, phoenix_repro
from quantark.asset.equity.engine.mc import (
    HestonPhoenixMCEngine,
    HestonSnowballMCEngine,
    PhoenixMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.pde import HestonSnowballPDESolver, LocalVolSnowballPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.pde import PhoenixPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import HestonKOResetSnowballPDESolver
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
from quantark.util.enum.engine_enums import EventProjectionMode

MC_PATHS = 400_000
#: the Heston statistics grid: the repro grid's first knock-in row (03-03, not the shared instant) sits 4-10 SE off
#: Heston MC with or without the pair (its own discretization there); the finer variance axis brings it in
HESTON_STATS_GRID = dict(n_x=128, n_v=32, n_t=128)


# --- products --------------------------------------------------------------------------------------------------------

def _discrete_ki(p, items):
    p.barrier_config = replace(p.barrier_config, ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
                               ki_barrier=items[0][1], ki_observation_schedule=ObservationSchedule(
                                   records=[ObservationRecord(observation_date=d, barrier=b) for d, b in items]))
    p.validate()
    return p


def _items(dates, level, a, b):
    """The knock-in records: ``level`` on every date but the instant; ``a`` on 07-03, ``b`` on 07-04 (none if None)."""
    out = []
    for d in dates:
        if d == JUL3:
            out.append((d, a))
            if b is not None:
                out.append((JUL4, b))
        elif d != JUL4:
            out.append((d, level))
    return out


def snow(a, b=None, reverse=False):
    """The Snowball repro (KO 103, the 07-04 KO record dropped) with a discrete KI on every KO date; reverse: the
    mirrored note (KO 97 below, KI above at 125)."""
    if reverse:
        p = snowball([ObservationRecord(observation_date=d, barrier=97.0) for d in REPRO_DATES if d != JUL4],
                     reverse=True)
        return _discrete_ki(p, _items(REPRO_DATES, 125.0, a, b))
    return _discrete_ki(repro(drop_jul4=True), _items(REPRO_DATES, 75.0, a, b))


def phx(a, b=None):
    return _discrete_ki(phoenix_repro("single"), _items(PHX_DATES, 70.0, a, b))


def kor(a, b=None, reverse=False):
    first = 125.0 if reverse else 75.0
    return ko_reset(ki=[(d, first) for d in KOR_PRE[:2]] + [(JUL3, a)] + ([(JUL4, b)] if b is not None else []),
                    reverse=reverse)


#: per family: (product, {variant: (07-03 level, 07-04 level)}); the widest level is the single record's
FAMILIES = {
    "snowball": (snow, dict(pair=(75.0, 80.0), reversed=(80.0, 75.0), wide=(80.0, None), dup=(75.0, 75.0),
                            base=(75.0, None))),
    "snowball_up": (lambda a, b: snow(a, b, reverse=True),
                    dict(pair=(125.0, 120.0), reversed=(120.0, 125.0), wide=(120.0, None), dup=(125.0, 125.0),
                         base=(125.0, None))),
    "phoenix": (phx, dict(pair=(70.0, 75.0), reversed=(75.0, 70.0), wide=(75.0, None), dup=(70.0, 70.0),
                          base=(70.0, None))),
    "ko_reset": (kor, dict(pair=(75.0, 80.0), reversed=(80.0, 75.0), wide=(80.0, None), dup=(75.0, 75.0),
                           base=(75.0, None))),
    "ko_reset_up": (lambda a, b: kor(a, b, reverse=True),
                    dict(pair=(125.0, 120.0), reversed=(120.0, 125.0), wide=(120.0, None), dup=(125.0, 125.0),
                         base=(125.0, None))),
}
#: the row of the 07-03 record in each family's knock-in table
INSTANT_ROW = {"snowball": 2, "snowball_up": 2, "phoenix": 2, "ko_reset": 2, "ko_reset_up": 2}


def product(family, variant):
    make, variants = FAMILIES[family]
    return make(*variants[variant])


def _v1(cls, **kwargs):
    return lambda: cls(params=QuadParams(grid_points=1001, **kwargs))


ENGINES = {
    "snowball": {
        "pde": (lambda: SnowballPDESolver(PDEParams()), env),
        "pde_nodal": (lambda: SnowballPDESolver(PDEParams(event_projection=EventProjectionMode.NODAL)), env),
        "lv_pde": (lambda: LocalVolSnowballPDESolver(PDEParams()), lambda: env(grid=True)),
        "heston_pde": (lambda: HestonSnowballPDESolver(HESTON, **HESTON_GRID), env),
        "v1": (_v1(SnowballQuadEngine), env),
        "v1_nodal": (_v1(SnowballQuadEngine, event_projection=EventProjectionMode.NODAL), env),
        "v2": (SnowballQuadEngineV2, env),
    },
    "snowball_up": {
        "pde": (lambda: SnowballPDESolver(PDEParams()), env),
        "v1": (_v1(SnowballQuadEngine), env),
        "v2": (SnowballQuadEngineV2, env),
    },
    "phoenix": {
        "pde": (lambda: PhoenixPDESolver(PDEParams()), env),
        "lv_pde": (lambda: LocalVolPhoenixPDESolver(PDEParams()), lambda: env(grid=True)),
        "heston_pde": (lambda: HestonPhoenixPDESolver(HESTON, n_x=128, n_v=12, n_t=48), env),
        "v1": (_v1(PhoenixQuadEngine), env),
        "v2": (PhoenixQuadEngineV2, env),
    },
    "ko_reset": {
        "pde": (lambda: KOResetSnowballPDESolver(PDEParams()), env),
        "heston_pde": (lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=96, n_v=16, n_t=64), env),
        "v1": (_v1(KOResetSnowballQuadEngine), env),
        "v1_nodal": (_v1(KOResetSnowballQuadEngine, event_projection=EventProjectionMode.NODAL), env),
        "v2": (KOResetSnowballQuadEngineV2, env),
    },
    "ko_reset_up": {
        "pde": (lambda: KOResetSnowballPDESolver(PDEParams()), env),
        "v1": (_v1(KOResetSnowballQuadEngine), env),
    },
}
CASES = [(f, e) for f, engines in ENGINES.items() for e in engines]
IDS = [f"{f}-{e}" for f, e in CASES]


def _rows(stats):
    return [float(p) for p in stats.ki_table.probability]


# --- price: one knock-in at the widest level -------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["pair", "reversed"])
@pytest.mark.parametrize("family,engine", CASES, ids=IDS)
def test_knock_in_pair_prices_as_the_widest_single_record(family, engine, variant):
    """Either contract order prices as the single record at the widest level (the knock-in is idempotent at one spot):
    bitwise on the PDE and QUAD V1 (the same stage at the same step); QUAD V2 adds the narrower level as a region
    break (the statistics' split), an exact integral split in two (measured |rel| <= 2.3e-16)."""
    make, make_env = ENGINES[family][engine]
    e = make_env()
    value, wide = make().price(product(family, variant), e), make().price(product(family, "wide"), e)
    if engine.startswith("v2"):
        assert abs(value / wide - 1.0) <= 1e-14, (value, wide)
    else:
        assert value == wide, (value, wide)


@pytest.mark.parametrize("family,engine", CASES, ids=IDS)
def test_knock_in_duplicate_prices_as_the_product_without_it(family, engine):
    """A same-level duplicate is dominated: bitwise the product without it. (Before: the Snowball PDE staged both
    records -- two cell averages on one level drift, 2.474447 against 2.476399 on the 1D PDE; QUAD V2 refused.)"""
    make, make_env = ENGINES[family][engine]
    e = make_env()
    assert make().price(product(family, "dup"), e) == make().price(product(family, "base"), e)


# --- statistics: one row per record, the instant's mass on the first record breached -----------------------------------

@pytest.mark.parametrize("family,engine", CASES, ids=IDS)
def test_knock_in_pair_statistics_attribute_each_record(family, engine):
    """The pair (narrow level first): the instant's two rows sum to the widest single record's row, every other row
    and the total are that product's; the first row is the narrow level's band (the single narrow record's row, to
    the discretization: abs 2e-4); the reconciliation holds. (Before: ``KnockInTable.times has duplicate rows``.)"""
    make, make_env = ENGINES[family][engine]
    e = make_env()
    k = INSTANT_ROW[family]
    pair, wide, base = (make().calculate_event_stats(product(family, v), e) for v in ("pair", "wide", "base"))
    rows, wide_rows, base_rows = _rows(pair), _rows(wide), _rows(base)
    assert len(rows) == len(wide_rows) + 1
    assert list(pair.ki_table.dates) == sorted(list(wide.ki_table.dates) + [JUL4])
    assert rows[k] + rows[k + 1] == pytest.approx(wide_rows[k], rel=1e-12, abs=1e-15)
    assert rows[:k] + rows[k + 2:] == pytest.approx(wide_rows[:k] + wide_rows[k + 1:], rel=1e-12, abs=1e-15)
    assert rows[k] == pytest.approx(base_rows[k], abs=2e-4) and rows[k + 1] > 0.01
    assert pair.knock_in.probability == pytest.approx(wide.knock_in.probability, rel=1e-12)
    assert abs(pair.reconciliation_error) <= 1e-9 * max(1.0, abs(pair.pv))


@pytest.mark.parametrize("variant,dominated", [("reversed", "wide"), ("dup", "base")])
@pytest.mark.parametrize("family,engine", CASES, ids=IDS)
def test_dominated_knock_in_record_has_zero_probability(family, engine, variant, dominated):
    """A later record inside an earlier one's region (the wide level first, or a duplicate) never knocks in first: its
    row is exactly 0, every other row the product's without it (to round-off)."""
    make, make_env = ENGINES[family][engine]
    e = make_env()
    k = INSTANT_ROW[family]
    rows = _rows(make().calculate_event_stats(product(family, variant), e))
    ref = _rows(make().calculate_event_stats(product(family, dominated), e))
    assert rows[k + 1] == 0.0
    assert rows[:k + 1] + rows[k + 2:] == pytest.approx(ref, rel=1e-12, abs=1e-15)


# --- against Monte Carlo ---------------------------------------------------------------------------------------------

_MC_CLASSES = {"snowball": (SnowballMCEngine, HestonSnowballMCEngine), "phoenix": (PhoenixMCEngine, HestonPhoenixMCEngine),
               "ko_reset": (SnowballMCEngine, HestonSnowballMCEngine)}
#: nodal events are left out: their own O(h) bias puts the rows after the instant 3.6 SE off MC with one record too
MC_CASES = [(f, e) for f in ("snowball", "phoenix", "ko_reset") for e in ENGINES[f] if "nodal" not in e]


@pytest.fixture(scope="module")
def mc_reference():
    """MC (400k paths, seed 7) per (family, model, variant): price, SE and the statistics. Black-Scholes MC is the
    local-vol engines' reference too (a flat grid surface is the Black-Scholes model)."""
    cache = {}

    def get(family, model, variant):
        key = (family, model, variant)
        if key not in cache:
            bs, heston = _MC_CLASSES[family]
            make = ((lambda: bs(params=MCParams(num_paths=MC_PATHS, seed=7))) if model == "bs" else
                    (lambda: heston(HESTON, params=MCParams(num_paths=MC_PATHS, seed=7))))
            engine = make()
            value = engine.price(product(family, variant), env())
            cache[key] = (value, engine.get_last_std_error(), make().calculate_event_stats(product(family, variant),
                                                                                          env()))
        return cache[key]

    return get


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["pair", "reversed"])
@pytest.mark.parametrize("family,engine", MC_CASES, ids=[f"{f}-{e}" for f, e in MC_CASES])
def test_knock_in_pair_agrees_with_mc(mc_reference, family, engine, variant):
    """Price within 3 SE + 1e-3 relative of the same model's MC; each knock-in row and P(knock in before any knock-out)
    within 3 binomial SE (MC attributes a path's first knock-in to the first record in contract order its spot
    breaches -- the zero-gap rule -- unchanged)."""
    make, make_env = ENGINES[family][engine]
    model = "heston" if engine.startswith("heston") else "bs"
    ref, se, mc = mc_reference(family, model, variant)
    p = product(family, variant)
    value = make().price(p, make_env())
    assert abs(value - ref) <= 3.0 * se + 1e-3 * abs(ref), (value, ref, se, (value - ref) / se)
    if engine == "heston_pde":
        make = {"snowball": lambda: HestonSnowballPDESolver(HESTON, **HESTON_STATS_GRID),
                "phoenix": lambda: HestonPhoenixPDESolver(HESTON, **HESTON_STATS_GRID),
                "ko_reset": lambda: HestonKOResetSnowballPDESolver(HESTON, **HESTON_STATS_GRID)}[family]
    stats = make().calculate_event_stats(p, make_env())
    rows, mc_rows = _rows(stats), _rows(mc)
    assert len(rows) == len(mc_rows)
    if engine == "heston_pde":
        # the first rows (03-03 / 05-05, before the instant) are the Heston route's own: 03-03 sits 3.5 SE off MC on
        # this grid as with the single wide record (the same row, ``test_knock_in_pair_statistics_attribute_each_record``)
        k = INSTANT_ROW[family]
        rows, mc_rows = rows[k:], mc_rows[k:]
    for r, m in zip(rows, mc_rows):
        sd = math.sqrt(m * (1.0 - m) / MC_PATHS)
        assert abs(r - m) <= 3.0 * sd + 1e-12, (rows, mc_rows)
    m = mc.knock_in.probability
    assert abs(stats.knock_in.probability - m) <= 3.0 * math.sqrt(m * (1.0 - m) / MC_PATHS)
    if variant == "reversed":
        assert _rows(mc)[INSTANT_ROW[family] + 1] == 0.0    # MC: the inert record never knocks in first


# --- the valuation instant and the exact first step ------------------------------------------------------------------

VALUATION = {"snowball": (78.0, 73.0, 90.0), "phoenix": (73.0, 69.0, 90.0), "ko_reset": (78.0, 73.0, 90.0)}
DETERMINISTIC = [(f, e) for f in ("snowball", "phoenix", "ko_reset") for e in ENGINES[f]
                 if e in ("pde", "heston_pde", "v1", "v2")]


@pytest.mark.parametrize("family,engine", DETERMINISTIC, ids=[f"{f}-{e}" for f, e in DETERMINISTIC])
def test_knock_in_pair_at_the_valuation_instant_and_as_the_first_observation(family, engine):
    """Valued on 07-03 (the pair at t = 0) with the spot in the band only the wide level breaches, below both, or above
    both: the note knocks in exactly where the wide record does (the solvers' valuation-instant knock-in tests every
    record of the instant -- before, the PDE and QUAD V1 read the instant's first record only); valued on 07-02 (the
    pair is the exact first step's event): as the wide record (QUAD V1's first step carries both levels: round-off)."""
    make, _ = ENGINES[family][engine]
    for e in [env(JUL3, spot=s) for s in VALUATION[family]] + [env(datetime(2025, 7, 2))]:
        if family == "phoenix" and engine == "heston_pde" and e.spot == 69.0:
            continue        # the Heston Phoenix statistics refuse that spot with one record too (a negative row)
        for variant in ("pair", "reversed"):
            value, wide = make().price(product(family, variant), e), make().price(product(family, "wide"), e)
            assert value == pytest.approx(wide, rel=1e-14, abs=1e-14), (e.spot, variant)
            rows = _rows(make().calculate_event_stats(product(family, variant), e))
            ref = _rows(make().calculate_event_stats(product(family, "wide"), e))
            assert sum(rows) == pytest.approx(sum(ref), rel=1e-12, abs=1e-15)


# --- the knock-in table --------------------------------------------------------------------------------------------

def test_knock_in_table_repeats_a_time_only_with_distinct_dates():
    """A dated table repeats a time once per record (its own date); a dateless table, or a repeated date, refuses."""
    from quantark.asset.equity.engine.event_stats import KnockInTable
    from quantark.util.exceptions import ValidationError

    t = [0.25, 0.5, 0.5, 0.75]
    dates = (datetime(2025, 4, 3), JUL3, JUL4, datetime(2025, 10, 3))
    table = KnockInTable(times=t, payment_times=t, dates=dates, payment_dates=dates, probability=[0.1, 0.1, 0.05, 0.1],
                         survival=[0.9, 0.8, 0.75, 0.65], interpolated=[False] * 4)
    assert table.size == 4
    with pytest.raises(ValidationError, match="duplicate rows"):
        KnockInTable(times=t, payment_times=t, probability=[0.1, 0.1, 0.05, 0.1], survival=[0.9, 0.8, 0.75, 0.65],
                     interpolated=[False] * 4)
    same = (datetime(2025, 4, 3), JUL3, JUL3, datetime(2025, 10, 3))
    with pytest.raises(ValidationError, match="duplicate rows"):
        KnockInTable(times=t, payment_times=t, dates=same, payment_dates=same, probability=[0.1, 0.1, 0.05, 0.1],
                     survival=[0.9, 0.8, 0.75, 0.65], interpolated=[False] * 4)
    with pytest.raises(ValidationError, match="sorted"):
        KnockInTable(times=[0.5, 0.25], payment_times=[0.5, 0.25], dates=dates[:2], payment_dates=dates[:2],
                     probability=[0.1, 0.1], survival=[0.9, 0.8], interpolated=[False] * 2)


# --- products without a shared knock-in instant: bitwise ----------------------------------------------------------

_PINS_PATH = os.path.join(os.path.dirname(__file__), "same_instant_ki_pins.json")
with open(_PINS_PATH) as _f:
    #: float.hex pins captured at ac8c0b2a BEFORE the T5a change: price and full statistics (KI rows included).
    #: 0.16.0 D1 (QUAD numerics run 5, KO-reset V1 knock-outs by the cell-average projection) re-pinned the six
    #: KO-reset V1 default-projection entries (kor_{base,wide}_v1_{fwd,stk}, kor_base_v1_val78 / _val74); their
    #: previous values are kept in the file under "_superseded_by_0.16.0_D1" (e.g. kor_base_v1_fwd price
    #: 0x1.1be31f43f709dp+3 = 8.87147486948407 -> 0x1.1bd6ebe4d78e9p+3 = 8.869985529858655). 0.16.0 T7 (the
    #: knocked-in KO-reset ends at schedule 2's last fixing, 12-03, not the 01-05 exercise date: owner ruling D31)
    #: re-pinned the thirteen ``kor_*`` grid entries; their previous values are under "_superseded_by_0.16.0_T7"
    #: (e.g. kor_base_pde price 8.871664315881244 -> 8.86368327529885, kor_base_pde_val78 -20.686395546090417 ->
    #: -20.788510675853676). 0.16.0 T5 (V1 convolutions on fast FFT lengths, owner-signed round-off) re-pinned the
    #: twenty-two QUAD V1 entries; previous values under "_superseded_by_0.16.0_T5_fft" (max rel move 1.7e-13 over
    #: values >= 1e-6, e.g. phx_wide_v1_fwd; the reconciliation residual ``recon`` is itself round-off, |v| <= 1.3e-14)
    PINS = json.load(_f)


def _pinned():
    v1 = lambda cls, route, **kw: (lambda: _route(cls(params=QuadParams(grid_points=1001, **kw)), route))  # noqa: E731
    nodal = dict(event_projection=EventProjectionMode.NODAL)
    out = {}
    for name, (lo, wide) in {"snow": (75.0, 80.0), "phx": (70.0, 75.0), "kor": (75.0, 80.0)}.items():
        make = {"snow": snow, "phx": phx, "kor": kor}[name]
        for variant, level in (("base", lo), ("wide", wide)):
            p = (lambda m=make, x=level: m(x))
            if name == "snow":
                out.update({
                    f"snow_{variant}_pde": (lambda: SnowballPDESolver(PDEParams()), p, env),
                    f"snow_{variant}_pde_nodal": (lambda: SnowballPDESolver(PDEParams(**nodal)), p, env),
                    f"snow_{variant}_lv": (lambda: LocalVolSnowballPDESolver(PDEParams()), p, lambda: env(grid=True)),
                    f"snow_{variant}_heston": (lambda: HestonSnowballPDESolver(HESTON, **HESTON_GRID), p, env),
                    f"snow_{variant}_v1_fwd": (v1(SnowballQuadEngine, "forward"), p, env),
                    f"snow_{variant}_v1_stk": (v1(SnowballQuadEngine, "stacked"), p, env),
                    f"snow_{variant}_v1_nodal": (v1(SnowballQuadEngine, "forward", **nodal), p, env),
                    f"snow_{variant}_v2": (SnowballQuadEngineV2, p, env),
                })
            elif name == "phx":
                out.update({
                    f"phx_{variant}_pde": (lambda: PhoenixPDESolver(PDEParams()), p, env),
                    f"phx_{variant}_lv": (lambda: LocalVolPhoenixPDESolver(PDEParams()), p, lambda: env(grid=True)),
                    f"phx_{variant}_heston": (lambda: HestonPhoenixPDESolver(HESTON, n_x=128, n_v=12, n_t=48), p,
                                              env),
                    f"phx_{variant}_v1_fwd": (v1(PhoenixQuadEngine, "forward"), p, env),
                    f"phx_{variant}_v1_stk": (v1(PhoenixQuadEngine, "stacked"), p, env),
                    f"phx_{variant}_v2": (PhoenixQuadEngineV2, p, env),
                })
            else:
                out.update({
                    f"kor_{variant}_pde": (lambda: KOResetSnowballPDESolver(PDEParams()), p, env),
                    f"kor_{variant}_heston": (lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=96, n_v=16, n_t=64),
                                              p, env),
                    f"kor_{variant}_v1_fwd": (v1(KOResetSnowballQuadEngine, "forward"), p, env),
                    f"kor_{variant}_v1_stk": (v1(KOResetSnowballQuadEngine, "stacked"), p, env),
                })
    for e_name, make_env in {"val78": lambda: env(JUL3, spot=78.0), "val74": lambda: env(JUL3, spot=74.0),
                             "first": lambda: env(datetime(2025, 7, 2))}.items():
        out.update({
            f"snow_base_pde_{e_name}": (lambda: SnowballPDESolver(PDEParams()), lambda: snow(80.0), make_env),
            f"snow_base_v1_{e_name}": (v1(SnowballQuadEngine, "forward"), lambda: snow(80.0), make_env),
            f"snow_base_v2_{e_name}": (SnowballQuadEngineV2, lambda: snow(80.0), make_env),
            f"phx_base_pde_{e_name}": (lambda: PhoenixPDESolver(PDEParams()), lambda: phx(75.0), make_env),
            f"phx_base_v1_{e_name}": (v1(PhoenixQuadEngine, "forward"), lambda: phx(75.0), make_env),
            f"kor_base_pde_{e_name}": (lambda: KOResetSnowballPDESolver(PDEParams()), lambda: kor(80.0), make_env),
            f"kor_base_v1_{e_name}": (v1(KOResetSnowballQuadEngine, "forward"), lambda: kor(80.0), make_env),
        })
    return out


def _route(engine, route):
    engine._v1_statistics_route = route
    return engine


def _stats_pins(s):
    h = lambda x: float(x).hex()  # noqa: E731
    out = {"pv": h(s.pv), "ko": [h(p) for p in s.observations.probability],
           "ko_cash": [h(p) for p in s.observations.discounted_cash],
           "maturity": h(s.maturity.probability), "ki_settlement": h(s.ki_settlement.probability),
           "lifetime": h(s.expected_lifetime), "recon": h(s.reconciliation_error)}
    if s.knock_in is not None and s.knock_in.probability is not None:
        out["ki"] = h(s.knock_in.probability)
        if s.knock_in.expected_time is not None:
            out["ki_time"] = h(s.knock_in.expected_time)
    if s.ki_table is not None:
        out["ki_rows"] = [h(p) for p in s.ki_table.probability]
        out["ki_surv"] = [h(p) for p in s.ki_table.survival]
    if s.coupons is not None:
        out["coupon_cash"] = [h(p) for p in s.coupons.discounted_cash]
        if s.coupons.probability is not None:
            out["coupon"] = [h(p) for p in s.coupons.probability]
    return out


PINNED = _pinned()


@pytest.mark.parametrize("name", list(PINNED))
def test_products_without_a_shared_knock_in_instant_price_and_report_bitwise(name):
    make, p, make_env = PINNED[name]
    assert float(make().price(p(), make_env())).hex() == PINS[name]["price"]
    assert _stats_pins(make().calculate_event_stats(p(), make_env())) == PINS[name]["stats"]
