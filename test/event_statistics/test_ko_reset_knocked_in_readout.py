"""KO-reset Monte Carlo: the knocked-in payoff reads the spot at the PRODUCT'S EXERCISE DATE (owner ruling 2026-11-27).

Owner ruling (2026-11-27, run 9; replaces D31 of 2026-11-14). A knocked-in KO-reset observes its schedule-2 knock-outs
up to schedule 2's last observation; if it has not knocked out it is a live put to the product's exercise date and
pays V1 of the spot there, at the exercise settlement (``resolve_terminal_timing``: the explicit settlement date when
booked, else the exercise date plus the settlement delay). This holds for ABSOLUTE, a carried knock-in and REBASED.
D31 had read V1 at schedule 2's end (ABSOLUTE) and at each path's rebased end (REBASED); the MC kernel
(``_compute_payoffs_ko_reset``, shared by the BSM, LocalVol, Heston and Heston-SLV MC engines) now reads it on the
exercise node, a pinned node of every grid. Annualized knocked-in amounts accrue to the exercise date (owner ruling O2).
Paths do not change.

Contract shapes. The pair keeps schedule 2 ending with schedule 1 (1.0) and books a maturity (the exercise) of 1.5 on
both layouts (the same grid, the same paths), so the readout separates schedule 2's end from the exercise.

- the pair (one contract, two layouts, the same paths): ABSOLUTE MC price == REBASED MC price to round-off on the
  BSM, LocalVol, Heston and Heston-SLV MC engines (both read at the exercise);
- every knocked-in path's payoff is V1 of the spot at the exercise node, settled at the terminal payment, on the
  carried knock-in, the native ABSOLUTE pair, a native ABSOLUTE whose knock-ins run to schedule 1's end and REBASED
  notes whose knock-ins end their schedule 2 early (AC3); the node is found here from the grid's times, the
  knocked-in paths decided by the independent same-path reference;
- on the tracker's ABSOLUTE copy of an early knock-in MC agrees with PDE 1D within 4 standard errors;
- the statistics follow the price: ``price_with_events(...).npv == price()`` bitwise, and the knocked-in settlement
  cash is the sum of V1 at the exercise over the knocked-in paths, discounted from the terminal payment.
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
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError
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


#: the pair's product maturity (its exercise): past schedule 2's end (1.0)
PAIR_MATURITY = 1.5


def pair(layout: str, *, knocked_in: bool = False):
    """The pair: a REBASED KO-reset knocked in at valuation (its one knock-in record at 0, spot 78 below 80), schedule
    2 offsets quarterly to 1.0 (ending with schedule 1: 0 + 1.0 == 1.0, D35's boundary), product maturity 1.5;
    ABSOLUTE = ``with_absolute_schedule_2(0)`` booked with the same maturity. ``knocked_in``: the ABSOLUTE layout
    carrying the knock-in (the tracker's flag)."""
    product = cases._ko_reset(PostKOScheduleMode.REBASED)
    product.barrier_config = replace(product.barrier_config, ki_observation_schedule=cases._schedule((0.0,), 80.0))
    product.post_barrier_config = replace(product.post_barrier_config,
                                          ko_observation_schedule=cases._schedule(QUARTERS, 95.0))
    product.maturity = PAIR_MATURITY
    product.validate()
    if layout == "rebased":
        return product
    record = product.barrier_config.ki_observation_schedule.records[0]
    converted = product.with_absolute_schedule_2(0.0, anchor=knock_in_anchor_for(record, 0))
    converted.maturity = PAIR_MATURITY
    converted.validate()
    if knocked_in:
        converted._otc_lifecycle_knocked_in = True
    return converted


QUARTERS = (0.25, 0.5, 0.75, 1.0)


#: ``rebased_early``'s maturity (its exercise): the last knock-in (1.0) plus schedule 2's last offset (0.75)
EARLY_MATURITY = 1.75


def rebased_early():
    """A REBASED note whose knock-ins (quarterly to 1.0, at 80) start schedule 2 offsets (0.25, 0.5, 0.75): a knock-in
    at 0.25 ends schedule 2 at 1.0 (with schedule 1, D35's floor), one at 1.0 at 1.75 -- the exercise (maturity 1.75,
    owner ruling O1's floor)."""
    product = cases._ko_reset(PostKOScheduleMode.REBASED)
    product.post_barrier_config = replace(product.post_barrier_config,
                                          ko_observation_schedule=cases._schedule((0.25, 0.5, 0.75), 95.0))
    product.maturity = EARLY_MATURITY
    product.validate()
    return product


def tracker_copy(ki_record: int = 0, age: float = 0.30, spot: float = 78.0):
    """The ABSOLUTE copy the KO-reset tracker prices for a knocked-in REBASED note: schedule 1 quarterly to 2y (KO 103,
    discrete KI 80 on the same dates), schedule 2 quarterly offsets to 1.75y after the knock-in (0.25 + 1.75 == 2.0,
    D35's boundary); knocked in on record ``ki_record`` (0.25) and aged ``age`` (0.3): schedule 2 and schedule 1 both
    end at 1.70. The REBASED note's maturity covers its latest end, the last knock-in 2.0 plus 1.75 (owner ruling O1,
    2026-11-27; was 2.0, the copy's prices are bitwise unchanged)."""
    product = cases._ko_reset(PostKOScheduleMode.REBASED)
    pre = tuple(np.arange(1, 9) * 0.25)
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=cases._schedule(pre, 103.0),
                                     ki_observation_schedule=cases._schedule(pre, 80.0))
    product.post_barrier_config = replace(product.post_barrier_config,
                                          ko_observation_schedule=cases._schedule(pre[:7], 95.0))
    product.maturity = 3.75
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


def _column(grid, time: float) -> int:
    """The path column of the grid node at ``time`` (column 0 is the valuation spot)."""
    times = np.r_[0.0, np.asarray(grid["all_times"], dtype=float)]
    matches = [j for j, t in enumerate(times) if is_close(float(t), time)]
    assert len(matches) == 1, (time, times)
    return matches[0]


def _v1_at_exercise(engine_make, product_make, env):
    """Per path: the kernel's payoff, the reference's knocked-in flags, V1 of the spot at the exercise (its tenor to
    the exercise, O2), the exercise time and its column."""
    product = product_make()
    call = _kernel_call(engine_make(), product, env)
    ref = mc_same_path_reference.ko_reset_paths(engine_make(), product_make(), env)
    np.testing.assert_array_equal(ref["payoffs"], call["payoffs"])  # the same paths
    exercise = float(resolve_terminal_timing(product, env).determination_time)
    column = _column(call["grid"], exercise)
    v1 = np.array([product.get_maturity_payoff_v1(s, env) for s in call["paths"][:, column]], dtype=float)
    return call, ref, v1, exercise, column


def _assert_v1_at_exercise(product, env, call, ref, v1):
    assert ref["v1"].any()
    np.testing.assert_array_equal(call["payoffs"][ref["v1"]], v1[ref["v1"]])
    terminal = resolve_terminal_timing(product, env)
    np.testing.assert_array_equal(call["settle"][ref["v1"]], float(terminal.payment_time))
    np.testing.assert_array_equal(ref["det"][ref["v1"]], float(terminal.determination_time))


# --- the pair -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_the_pair_prices_the_same_on_both_layouts(engine):
    """One contract, two layouts, the same paths (T23, AC3): ABSOLUTE MC == REBASED MC to round-off, both reading V1
    at the exercise (1.5) on the tenor to the exercise."""
    make, market = MC_ENGINES[engine]
    env = market(PAIR_SPOT)
    rebased = float(make().price(pair("rebased"), env))
    absolute = float(make().price(pair("absolute"), env))
    assert abs(absolute - rebased) <= ROUND_OFF * abs(rebased), (engine, absolute, rebased, absolute - rebased)


@pytest.mark.parametrize("layout", ("absolute", "rebased"))
@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_the_pair_reads_v1_at_the_exercise(engine, layout):
    """Knocked in at valuation on the paths: every knocked-in path pays V1 of the spot at the exercise node (1.5), not
    schedule 2's end (1.0), and settles at the terminal payment."""
    make, market = MC_ENGINES[engine]
    env = market(PAIR_SPOT)
    product_make = lambda: pair(layout)  # noqa: E731
    call, ref, v1, exercise, column = _v1_at_exercise(make, product_make, env)
    assert exercise == PAIR_MATURITY and column > _column(call["grid"], 1.0)
    _assert_v1_at_exercise(product_make(), env, call, ref, v1)


@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_a_rebased_early_knock_in_reads_v1_at_the_exercise(engine):
    """AC3: REBASED knock-ins whose schedule 2 ends before the exercise (a knock-in at 0.25 ends it at 1.0) read V1
    at the exercise (1.75), the hand payoff on the kernel's own paths."""
    make, market = MC_ENGINES[engine]
    env = market(90.0)
    call, ref, v1, exercise, column = _v1_at_exercise(make, rebased_early, env)
    assert exercise == EARLY_MATURITY
    early = ref["v1"] & (ref["post_end"] < exercise - 1e-9)
    assert early.any()
    assert column > _column(call["grid"], 1.0)
    _assert_v1_at_exercise(rebased_early(), env, call, ref, v1)


# --- the carried knock-in (tracker copy) ----------------------------------------------------------------------------


@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_the_carried_knock_in_reads_v1_at_the_exercise(engine):
    """A carried knock-in (the tracker's flag) on the pair's ABSOLUTE layout: every path is knocked in, and each one that
    does not knock out on schedule 2 pays V1 of the spot at the exercise node (1.5), not schedule 2's end (1.0)."""
    make, market = MC_ENGINES[engine]
    env = market(78.0)
    product_make = lambda: pair("absolute", knocked_in=True)  # noqa: E731
    call, ref, v1, exercise, column = _v1_at_exercise(make, product_make, env)
    assert exercise == PAIR_MATURITY and column > _column(call["grid"], 1.0)
    assert not ref["v0"].any()
    _assert_v1_at_exercise(product_make(), env, call, ref, v1)


def test_the_tracker_copy_of_a_valid_rebased_note_ends_with_schedule_two():
    """D35: the tracker's ABSOLUTE copy ends at schedule 2's end and schedule 1 no later."""
    product, env = tracker_copy()
    assert is_close(float(product.get_post_maturity_time(env)), 1.70)
    assert is_close(float(product.get_max_maturity_time(env)), float(product.get_post_maturity_time(env)))


def valid_native_absolute():
    """A native ABSOLUTE KO-reset (not knocked in): schedule 1 and the knock-in quarterly to 1.0, schedule 2 quarterly
    to 1.0 (ending with schedule 1), product maturity 1.5. A knock-in on any record, 1.0 included, meets schedule 2
    in force."""
    product = cases._ko_reset(PostKOScheduleMode.ABSOLUTE)
    product.post_barrier_config = replace(product.post_barrier_config,
                                          ko_observation_schedule=cases._schedule(QUARTERS, 95.0))
    product.maturity = PAIR_MATURITY
    product.validate()
    return product


def test_a_native_absolute_whose_schedule_two_ends_first_is_refused():
    """T25e's late knock-in (schedule 2 (0.25, 0.5), knock-ins to 1.0) cannot be booked any more (D35)."""
    product = cases._ko_reset(PostKOScheduleMode.ABSOLUTE)
    product.post_barrier_config = replace(product.post_barrier_config,
                                          ko_observation_schedule=cases._schedule((0.25, 0.5), 95.0))
    with pytest.raises(ValidationError, match="schedule 2 must end on or after schedule 1"):
        product.validate()


@pytest.mark.parametrize("engine", tuple(MC_ENGINES))
def test_every_knock_in_reads_v1_at_the_exercise(engine):
    """Every knocked-in path, the ones knocking in on schedule 1's last record (1.0 == schedule 2's end) included,
    pays V1 of the spot at the exercise node (1.5)."""
    make, market = MC_ENGINES[engine]
    env = market(90.0)
    call, ref, v1, exercise, column = _v1_at_exercise(make, valid_native_absolute, env)
    assert exercise == PAIR_MATURITY and column > _column(call["grid"], 1.0)
    assert (ref["ki_time"][ref["v1"]] == 1.0).any()
    _assert_v1_at_exercise(valid_native_absolute(), env, call, ref, v1)


#: 16 independent PSEUDO runs
TRACKER_SEEDS, TRACKER_PATHS = tuple(range(501, 517)), 262_144


def test_the_tracker_copy_agrees_with_pde_within_four_standard_errors():
    """MC vs PDE 1D on the tracker's ABSOLUTE copy of an early knock-in (the valid shape: schedule 2 ends with
    schedule 1)."""
    runs = [float(SnowballMCEngine(params=MCParams(num_paths=TRACKER_PATHS, seed=s)).price(*tracker_copy()))
            for s in TRACKER_SEEDS]
    mc = float(np.mean(runs))
    se = float(np.std(runs, ddof=1) / np.sqrt(len(runs)))
    pde = float(KOResetSnowballPDESolver(params=PDEParams()).price(*tracker_copy()))
    assert abs(mc - pde) <= 4.0 * se, (mc, se, pde, (mc - pde) / se)


# --- the statistics follow the price --------------------------------------------------------------------------------


@pytest.mark.parametrize("contract", ("pair_absolute", "pair_carried", "rebased_early"))
def test_statistics_follow_the_knocked_in_readout(contract):
    """``npv`` is ``price()`` bitwise (D23) and the knocked-in settlement cash is V1 at the exercise summed over the
    knocked-in paths (decided by the reference), each discounted from the terminal payment; the knocked-in settlement
    row sits at the terminal."""
    make_product, env = {
        "pair_absolute": (lambda: pair("absolute"), cases.flat_env(PAIR_SPOT)),
        "pair_carried": (lambda: pair("absolute", knocked_in=True), cases.flat_env(PAIR_SPOT)),
        "rebased_early": (rebased_early, cases.flat_env(90.0)),
    }[contract]

    def make():
        return SnowballMCEngine(params=MCParams(**_MC))

    result = make().price_with_events(make_product(), env)
    price = float(make().price(make_product(), env))
    assert result.npv == price
    _, ref, v1, _, _ = _v1_at_exercise(make, make_product, env)
    df = np.array([env.rate_curve.get_discount_factor(float(t)) for t in ref["settle"]], dtype=float)
    expected = float(np.sum(np.where(ref["v1"], v1 * df, 0.0)) / ref["n"])
    stats = make().calculate_event_stats(make_product(), env)
    assert stats.pv == price
    cash = float(stats.ki_settlement.discounted_cash)
    assert abs(cash - expected) <= ROUND_OFF * abs(expected), (cash, expected)
    terminal = resolve_terminal_timing(make_product(), env)
    assert stats.ki_settlement.time == float(terminal.determination_time)
    assert stats.ki_settlement.payment_time == float(terminal.payment_time)
