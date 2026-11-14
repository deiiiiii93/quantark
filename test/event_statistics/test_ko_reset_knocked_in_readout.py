"""KO-reset Monte Carlo: the knocked-in payoff reads the spot at schedule 2's end (spec D31, AC34).

Owner ruling (2026-11-14, option A). A knocked-in KO-reset is a knocked-in snowball on the second schedule, alive
until that schedule ends: its knocked-in payoff V1 is decided on the spot at schedule 2's last fixing and paid there.
The MC kernel's ABSOLUTE branch (and the carried-knock-in branch, which is ABSOLUTE) settled V1 at schedule 2's end
but read its spot at the grid's last column. When schedule 1's dates run past schedule 2's end -- the ABSOLUTE copy
the KO-reset tracker prices for an early knock-in, or the T23 pair settling inside the axis -- that column is
schedule 1's end, so V1 was read on the wrong spot (diagnosis T25d: the whole REBASED/ABSOLUTE gap on the pair, to
1.4e-14). REBASED MC already read each path's own schedule-2 end, as PDE and QUAD do. Paths do not change.

- the T23 ``settles_inside`` pair (one contract, two layouts, the same paths): ABSOLUTE MC price == REBASED MC price
  to round-off on the BSM, LocalVol, Heston and Heston-SLV MC engines (they share ``_compute_payoffs_ko_reset``);
- on the tracker's ABSOLUTE copy of an early knock-in (schedule 1 quarterly to 2y, schedule 2 quarterly offsets to
  1y after a knock-in on 0.25, aged 0.3: schedule 2 ends at 0.95, the grid at 1.70) MC agrees with PDE 1D within
  4 standard errors (base: about +5.7 SE);
- every knocked-in path's payoff is V1 of the spot at schedule 2's end node, on the carried knock-in and on the
  native ABSOLUTE pair; the node is found here from the grid's times, the knocked-in paths decided by the
  independent same-path reference;
- the statistics follow the price: ``price_with_events(...).npv == price()`` bitwise, and the knocked-in settlement
  cash is the sum of V1 at schedule 2's end over the knocked-in paths, discounted from the settlement.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from test_asset.equity.engine_mc import mc_same_path_reference

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.numerical import is_close

warnings.filterwarnings("ignore", category=DeprecationWarning)

BANK_DIR = Path(__file__).resolve().parents[1] / "cashleg" / "event_stats_baseline"


def _load_cases():
    name = "_cashleg_event_stats_baseline_cases"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BANK_DIR / "cases.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cases = _load_cases()

#: same paths, same tenor, same settlement: only summation order may differ
ROUND_OFF = 1e-12
PAIR_SPOT = 78.0
_MC = dict(num_paths=4096, seed=7)

#: MC engines sharing ``_compute_payoffs_ko_reset``, each with the market it prices on
MC_ENGINES = {
    "mc": (lambda: SnowballMCEngine(params=MCParams(**_MC)), cases.flat_env),
    "mc_lv": (lambda: LocalVolSnowballMCEngine(MCParams(**_MC)), cases.skewed_env),
    "mc_heston": (lambda: HestonSnowballMCEngine(cases.HESTON, params=MCParams(**_MC)), cases.flat_env),
    "mc_slv": (lambda: HestonSLVSnowballMCEngine(cases.HESTON, params=MCParams(**_MC),
                                                 leverage_surface=cases._leverage()), cases.skewed_env),
}


# --- contracts ------------------------------------------------------------------------------------------------------


def pair(layout: str):
    """The T23 ``settles_inside`` pair: a REBASED KO-reset knocked in at valuation (its one knock-in record at 0, spot
    78 below 80), schedule 2 offsets (0.25, 0.5), schedule 1 to 1.0; ABSOLUTE = ``with_absolute_schedule_2(0)``."""
    product = cases._ko_reset(PostKOScheduleMode.REBASED)
    product.barrier_config = replace(product.barrier_config, ki_observation_schedule=cases._schedule((0.0,), 80.0))
    product.post_barrier_config = replace(product.post_barrier_config,
                                          ko_observation_schedule=cases._schedule((0.25, 0.5), 95.0))
    product.maturity = 1.0
    product.validate()
    if layout == "rebased":
        return product
    record = product.barrier_config.ki_observation_schedule.records[0]
    return product.with_absolute_schedule_2(0.0, anchor=knock_in_anchor_for(record, 0))


def tracker_copy(ki_record: int = 0, age: float = 0.30, spot: float = 78.0):
    """The ABSOLUTE copy the KO-reset tracker prices for a knocked-in REBASED note: schedule 1 quarterly to 2y (KO 103,
    discrete KI 80 on the same dates), schedule 2 quarterly offsets to 1y after the knock-in; knocked in on record
    ``ki_record`` (0.25) and aged ``age`` (0.3): schedule 2 ends at 0.95, schedule 1 at 1.70."""
    product = cases._ko_reset(PostKOScheduleMode.REBASED)
    pre = tuple(np.arange(1, 9) * 0.25)
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=cases._schedule(pre, 103.0),
                                     ki_observation_schedule=cases._schedule(pre, 80.0))
    product.maturity = 2.0
    product.validate()
    env = cases.flat_env(spot)
    record = product.barrier_config.ki_observation_schedule.records[ki_record]
    anchor = knock_in_anchor_for(record, ki_record)
    ki_time = float(record.observation_time) - age
    product.time_shift(age, env.valuation_date, env)
    converted = product.with_absolute_schedule_2(ki_time, anchor=anchor)
    converted._otc_lifecycle_knocked_in = True
    return converted, env


# --- helpers --------------------------------------------------------------------------------------------------------


def _kernel_call(engine, product, env) -> dict:
    """The paths and grid of a single-batch price (a spy on the shared kernel)."""
    calls = []
    original = engine._compute_payoffs_ko_reset

    def spy(product_, env_, paths, grid, *args, **kwargs):
        out = original(product_, env_, paths, grid, *args, **kwargs)
        calls.append(dict(paths=paths, grid=grid, payoffs=out[0], settle=out[1]))
        return out

    engine._compute_payoffs_ko_reset = spy
    try:
        price = float(engine.price(product, env))
    finally:
        del engine.__dict__["_compute_payoffs_ko_reset"]
    assert len(calls) == 1, f"a single-batch route prices one kernel call, got {len(calls)}"
    return dict(calls[0], price=price)


def _schedule_two_end_column(grid, end: float) -> int:
    """The path column of the grid node at ``end`` (column 0 is the valuation spot)."""
    times = np.r_[0.0, np.asarray(grid["all_times"], dtype=float)]
    matches = [j for j, t in enumerate(times) if is_close(float(t), end)]
    assert len(matches) == 1, (end, times)
    return matches[0]


def _v1_at_schedule_two_end(engine_make, product_make, env):
    """Per path: the kernel's payoff, the reference's knocked-in flags, V1 of the spot at schedule 2's end."""
    product = product_make()
    call = _kernel_call(engine_make(), product, env)
    ref = mc_same_path_reference.ko_reset_paths(engine_make(), product_make(), env)
    np.testing.assert_array_equal(ref["payoffs"], call["payoffs"])  # the same paths
    end = float(product.get_post_maturity_time(env))
    column = _schedule_two_end_column(call["grid"], end)
    v1 = np.array([product.get_maturity_payoff_v1(s, env) for s in call["paths"][:, column]], dtype=float)
    return call, ref, v1, end, column


# --- AC34: the pair -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_the_settles_inside_pair_prices_the_same_on_both_layouts(engine):
    """One contract, two layouts, the same paths (T23): ABSOLUTE MC == REBASED MC to round-off. Base: ABSOLUTE read
    V1 on schedule 1's end (1.0) instead of schedule 2's (0.5), 88.0885 vs 88.0633 on BSM MC."""
    make, market = MC_ENGINES[engine]
    env = market(PAIR_SPOT)
    rebased = float(make().price(pair("rebased"), env))
    absolute = float(make().price(pair("absolute"), env))
    assert abs(absolute - rebased) <= ROUND_OFF * abs(rebased), (engine, absolute, rebased, absolute - rebased)


@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_the_pair_absolute_reads_v1_at_schedule_two_end(engine):
    """Native ABSOLUTE (no carried knock-in; knocked in at valuation on the paths): every knocked-in path pays V1 of the
    spot at schedule 2's end node (0.5), not the grid's last (1.0)."""
    make, market = MC_ENGINES[engine]
    env = market(PAIR_SPOT)
    call, ref, v1, end, column = _v1_at_schedule_two_end(make, lambda: pair("absolute"), env)
    assert end == 0.5 and column < call["paths"].shape[1] - 1  # the grid runs past schedule 2's end
    assert ref["v1"].any()
    np.testing.assert_array_equal(call["payoffs"][ref["v1"]], v1[ref["v1"]])


# --- AC34: the carried knock-in (tracker copy) ----------------------------------------------------------------------


@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_the_carried_knock_in_reads_v1_at_schedule_two_end(engine):
    """The tracker's ABSOLUTE copy carries the knock-in: every path is knocked in, and each one that does not knock out
    on schedule 2 pays V1 of the spot at schedule 2's end node (0.95), not the grid's last (1.70)."""
    make, market = MC_ENGINES[engine]
    env = market(78.0)
    call, ref, v1, end, column = _v1_at_schedule_two_end(make, lambda: tracker_copy()[0], env)
    assert is_close(end, 0.95) and column < call["paths"].shape[1] - 1
    assert ref["v1"].any() and not ref["v0"].any()
    np.testing.assert_array_equal(call["payoffs"][ref["v1"]], v1[ref["v1"]])


def native_short_schedule_two():
    """A native ABSOLUTE KO-reset (not knocked in) whose schedule 2 (0.25, 0.5) ends before schedule 1 and its knock-in
    records (0.25 .. 1.0): a knock-in on 0.25 / 0.5 meets schedule 2 in force, one on 0.75 / 1.0 comes after it."""
    product = cases._ko_reset(PostKOScheduleMode.ABSOLUTE)
    product.post_barrier_config = replace(product.post_barrier_config,
                                          ko_observation_schedule=cases._schedule((0.25, 0.5), 95.0))
    product.validate()
    return product


@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_a_knock_in_by_schedule_two_end_reads_there_a_later_one_is_left_alone(engine):
    """A path knocked in at or before schedule 2's end pays V1 of the spot there (D31). A knock-in after schedule 2's
    end has no schedule 2 left in force and no ruled readout yet (T25e open item, owner question): those paths keep
    the grid's last node, as at base (reading them at schedule 2's end would read a spot before the knock-in)."""
    make, market = MC_ENGINES[engine]
    env = market(90.0)
    call, ref, v1, end, column = _v1_at_schedule_two_end(make, native_short_schedule_two, env)
    early = ref["v1"] & (ref["ki_time"] <= end)
    late = ref["v1"] & (ref["ki_time"] > end)
    assert early.any() and late.any()
    np.testing.assert_array_equal(call["payoffs"][early], v1[early])
    product = native_short_schedule_two()
    at_grid_end = np.array([product.get_maturity_payoff_v1(s, env) for s in call["paths"][late, -1]], dtype=float)
    np.testing.assert_array_equal(call["payoffs"][late], at_grid_end)


#: 16 independent PSEUDO runs: base sits ~5.7 SE above PDE (72.707 vs 72.631), the fix ~-0.6 SE
TRACKER_SEEDS, TRACKER_PATHS = tuple(range(501, 517)), 262_144


def test_the_tracker_copy_agrees_with_pde_within_four_standard_errors():
    """MC vs PDE 1D on the tracker's ABSOLUTE copy of an early knock-in (PDE was already right, T25d: PDE 72.6312,
    QUAD V2 72.6312)."""
    runs = [float(SnowballMCEngine(params=MCParams(num_paths=TRACKER_PATHS, seed=s)).price(*tracker_copy()))
            for s in TRACKER_SEEDS]
    mc = float(np.mean(runs))
    se = float(np.std(runs, ddof=1) / np.sqrt(len(runs)))
    pde = float(KOResetSnowballPDESolver(params=PDEParams()).price(*tracker_copy()))
    assert abs(mc - pde) <= 4.0 * se, (mc, se, pde, (mc - pde) / se)


# --- AC34: the statistics follow the price --------------------------------------------------------------------------


@pytest.mark.parametrize("contract", ("pair_absolute", "tracker_copy"))
def test_statistics_follow_the_knocked_in_readout(contract):
    """``npv`` is ``price()`` bitwise (D23) and the knocked-in settlement cash is V1 at schedule 2's end summed over
    the knocked-in paths (decided by the reference), each discounted from its settlement."""
    if contract == "pair_absolute":
        make_product, env = (lambda: pair("absolute")), cases.flat_env(PAIR_SPOT)
    else:
        make_product, env = (lambda: tracker_copy()[0]), tracker_copy()[1]

    def make():
        return SnowballMCEngine(params=MCParams(**_MC))

    result = make().price_with_events(make_product(), env)
    price = float(make().price(make_product(), env))
    assert result.npv == price
    _, ref, v1, _, _ = _v1_at_schedule_two_end(make, make_product, env)
    df = np.array([env.rate_curve.get_discount_factor(float(t)) for t in ref["settle"]], dtype=float)
    expected = float(np.sum(np.where(ref["v1"], v1 * df, 0.0)) / ref["n"])
    stats = make().calculate_event_stats(make_product(), env)
    assert stats.pv == price
    cash = float(stats.ki_settlement.discounted_cash)
    assert abs(cash - expected) <= ROUND_OFF * abs(expected), (cash, expected)
