"""The Monte Carlo time grids decide "the same instant" by the contract clock's rule (T7; spec D2, AC6).

* Two contract instants one second apart are two grid nodes, and each record maps to its own node (AC6).
* D2: a node the engine adds itself (a continuous-KI ``linspace`` node, a curve knot) a hair after a contract
  instant snaps onto that instant: the contract's float is the node, never the engine's spelling.
* The barrier / sharkfin / accumulator grids union their observation times with maturity through
  ``merge_time_nodes(..., pin=T)``: a last observation one ulp from maturity is one node, maturity bit for bit,
  and every observation maps to its node by ``node_index``. Grids without such a pair are unchanged byte for byte.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import (
    AccumulatorMCEngine,
    BarrierOptionMCEngine,
    DoubleSharkfinOptionMCEngine,
    PhoenixMCEngine,
    SingleSharkfinOptionMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.mc.ki_bridge import continuous_ki_step_count
from quantark.asset.equity.engine.mc.term_inputs import make_df_fn
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import (
    BarrierOption,
    DoubleSharkfinOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.numerical import Tolerance

SECOND = 1.0 / (365.0 * 86400.0)
UP = lambda t: float(np.nextafter(t, np.inf))      # noqa: E731
DOWN = lambda t: float(np.nextafter(t, -np.inf))   # noqa: E731


def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2), rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 9, 11),
    )


def _wired(engine, e):
    """The context ``price`` sets before it builds a grid."""
    engine._term_ctx = (e, 100.0)
    engine._df = make_df_fn(e)
    return engine


def _mc():
    return MCParams(seed=7, num_paths=2048)


def snowball(ko_times, ki_times=None, continuous=False):
    ki = (dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True) if continuous
          else dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ki_times)))
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=ko_times[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(ko_times),
                                     ki_barrier=75.0, **ki),
    )


def phoenix(ko_times):
    """Non-memory Phoenix, continuous KI 70, KO 103 on ``ko_times`` (one coupon per KO observation)."""
    p = create_standard_phoenix(100.0, 100.0, ko_times[-1], ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                coupon_rate=0.01, num_observations=len(ko_times), memory_coupon=False,
                                is_annualized=False)
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None, ko_observation_schedule=ObservationSchedule(
        records=[ObservationRecord(observation_time=t, barrier=103.0) for t in ko_times]))
    return p


def ko_reset(pre_times, post_times):
    """ABSOLUTE KO-reset with a continuous knock-in over schedule 1."""
    def schedule(times, level):
        return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])

    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=post_times[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=schedule(pre_times, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=schedule(post_times, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)


def _linspace_node(engine, e, horizon):
    """A continuous-KI ``linspace`` node in the second quarter of the horizon (not a dyadic fraction of it)."""
    n = continuous_ki_step_count(e.bus_days_in_year, horizon, engine._ki_bridge_nodes_per_day())
    nodes = np.linspace(0, horizon, n + 1)[1:]
    j = n // 3
    assert DOWN(nodes[j]) != nodes[j]
    return float(nodes[j])


# ======================================================================================================================
# AC6: two instants one second apart are two nodes
# ======================================================================================================================

class TestIntradayInstants:
    KO = [0.25, 0.5, 0.75, 1.0]
    KI = [0.5 + SECOND]

    def test_grid_holds_both_instants_and_each_record_maps_to_its_own_node(self):
        e = env()
        p = snowball(self.KO, self.KI)
        p.validate()
        eng = _wired(SnowballMCEngine(params=_mc()), e)
        all_times, dt, ko_idx, ki_idx = eng._build_time_grid(p, e, p.get_maturity(e))
        assert 0.5 in all_times and (0.5 + SECOND) in all_times
        assert dt.min() > 0.0
        assert all_times[ko_idx[1]] == 0.5
        assert all_times[ki_idx[0]] == 0.5 + SECOND
        assert ko_idx[1] + 1 == ki_idx[0]

    def test_prices_finitely(self):
        e = env()
        price = SnowballMCEngine(params=_mc()).price(snowball(self.KO, self.KI), e)
        assert np.isfinite(price)


# ======================================================================================================================
# D2: a contract instant is never displaced by an engine-added node
# ======================================================================================================================

class TestContractInstantsAnchorTheGrid:
    def test_snowball_linspace_node_one_ulp_after_a_ko_date_keeps_the_ko_float(self):
        e = env()
        eng = _wired(SnowballMCEngine(params=_mc()), e)
        node = _linspace_node(eng, e, 1.0)
        ko = DOWN(node)
        p = snowball([0.25, ko, 0.75, 1.0], continuous=True)
        p.validate()
        assert p.barrier_config.ko_observation_dates[1] == ko
        all_times, dt, ko_idx, _ = eng._build_time_grid(p, e, p.get_maturity(e))
        assert ko in all_times and node not in all_times
        assert all_times[ko_idx[1]] == ko
        assert all_times[-1] == 1.0 and dt.min() >= Tolerance.ZERO

    def test_phoenix_linspace_node_one_ulp_after_a_ko_date_keeps_the_ko_float(self):
        e = env()
        eng = _wired(PhoenixMCEngine(params=_mc()), e)
        node = _linspace_node(eng, e, 1.0)
        ko = DOWN(node)
        p = phoenix([0.25, ko, 0.75, 1.0])
        p.validate()
        all_times, _, ko_idx, _ = eng._build_time_grid(p, e, p.get_maturity(e))
        assert ko in all_times and node not in all_times
        assert all_times[ko_idx[1] - 1] == ko            # Phoenix indices are path columns (node + 1)

    def test_ko_reset_linspace_node_one_ulp_after_a_pre_ko_date_keeps_the_ko_float(self):
        e = env()
        eng = _wired(SnowballMCEngine(params=_mc()), e)
        node = _linspace_node(eng, e, 0.5)
        ko = DOWN(node)
        p = ko_reset([0.125, ko, 0.5], [0.75, 1.0])
        p.validate()
        grid = eng._build_time_grid_ko_reset(p, e, p.get_max_maturity_time(e))
        all_times = grid["all_times"]
        assert ko in all_times and node not in all_times
        assert all_times[grid["pre_indices"][1]] == ko

    def test_every_contract_time_is_its_own_node(self):
        """The event-statistics ledgers key cash on RECORD times (KO / schedule-2 / maturity times): with the anchors,
        every contract time is a grid node bit for bit, so those keys are node times."""
        e = env()
        eng = _wired(SnowballMCEngine(params=_mc()), e)
        node = _linspace_node(eng, e, 0.5)
        p = ko_reset([0.125, DOWN(node), 0.5], [0.75, 1.0])
        p.validate()
        grid = eng._build_time_grid_ko_reset(p, e, p.get_max_maturity_time(e))
        at = grid["all_times"]
        np.testing.assert_array_equal(at[grid["pre_indices"]], grid["pre_times"])
        np.testing.assert_array_equal(at[grid["post_indices"]], grid["post_times"])
        assert at[grid["pre_maturity_idx"]] == p.get_pre_maturity_time(e)

    def test_continuous_snowball_without_a_collision_is_unchanged(self):
        """No engine node within Tolerance.ZERO of a contract instant: the anchors change nothing."""
        e = env()
        eng = _wired(SnowballMCEngine(params=_mc()), e)
        p = snowball([0.25, 0.5, 0.75, 1.0], continuous=True)
        p.validate()
        all_times, _, ko_idx, _ = eng._build_time_grid(p, e, 1.0)
        n = continuous_ki_step_count(e.bus_days_in_year, 1.0, eng._ki_bridge_nodes_per_day())
        legacy = np.array(sorted({0.25, 0.5, 0.75, 1.0} | set(np.linspace(0, 1.0, n + 1)[1:])), dtype=float)
        assert all_times.tobytes() == legacy.tobytes()
        np.testing.assert_array_equal(ko_idx, [int(np.searchsorted(legacy, t)) for t in (0.25, 0.5, 0.75, 1.0)])


# ======================================================================================================================
# Barrier / sharkfin / accumulator: observation times unioned with maturity, one node per instant
# ======================================================================================================================

def _barrier(times):
    return BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT, rebate=1.0,
        maturity=times[-1], observation_type=ObservationType.DISCRETE,
        observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_time=t) for t in times]))


def _single_sharkfin(times):
    return SingleSharkfinOption(
        strike=100.0, option_type=OptionType.CALL, barrier=120.0, knock_out_rebate=1.0, maturity=times[-1],
        observation_type=ObservationType.DISCRETE,
        observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_time=t) for t in times]))


def _double_sharkfin(times):
    return DoubleSharkfinOption(
        strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=70.0, knock_out_rebate=1.0,
        maturity=times[-1], observation_type=ObservationType.DISCRETE,
        observation_schedule=ObservationSchedule(records=[
            ObservationRecord(observation_time=t, upper_barrier=130.0, lower_barrier=70.0) for t in times]))


def _discrete_grid(kind, times, T):
    """(all_times, dt_array, obs_indices) of the engine's discrete grid for observation ``times`` and maturity ``T``."""
    e = env()
    if kind == "accumulator":
        return AccumulatorMCEngine(params=_mc())._observation_grid(np.asarray(times, dtype=float), T)
    if kind == "barrier":
        p = _barrier(times)
        p.validate()
        out = BarrierOptionMCEngine(params=_mc())._build_discrete_grid(p, e, T)
        return out[0], out[1], out[2]
    if kind == "single_sharkfin":
        p = _single_sharkfin(times)
        p.validate()
        dt, idx = SingleSharkfinOptionMCEngine(params=_mc())._build_discrete_grid(p, e, T)[:2]
        return np.cumsum(dt), dt, idx
    p = _double_sharkfin(times)
    p.validate()
    out = DoubleSharkfinOptionMCEngine(params=_mc())._build_discrete_grid(p, e, T)
    return out[0], out[1], out[2]


KINDS = ["accumulator", "barrier", "single_sharkfin", "double_sharkfin"]


@pytest.mark.parametrize("kind", KINDS)
def test_last_observation_one_ulp_from_maturity_is_one_node_maturity_exactly(kind):
    times = [0.25, 0.5, 0.75, 1.0]
    T = UP(1.0)                                       # maturity spelled one ulp past the last observation
    all_times, dt, idx = _discrete_grid(kind, times, T)
    if kind == "single_sharkfin":                     # this builder returns no node array: rebuild it exactly
        assert dt.size == 4
    else:
        assert all_times.size == 4 and all_times[-1] == T
    np.testing.assert_array_equal(idx, [0, 1, 2, 3])


@pytest.mark.parametrize("kind", KINDS)
def test_maturity_inside_the_retired_relative_width_is_its_own_node(kind):
    """5e-10 y is two instants (Tolerance.ZERO 1e-10); the relative 1e-9 rule merged them."""
    times = [0.25, 0.5, 0.75, 1.0]
    T = 1.0 + 5e-10
    _, dt, idx = _discrete_grid(kind, times, T)
    assert dt.size == 5 and dt[-1] > 0.0
    np.testing.assert_array_equal(idx, [0, 1, 2, 3])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("last", [1.0, 0.9])
def test_grid_without_a_pair_is_byte_identical_to_the_legacy_construction(kind, last):
    times = [1.0 / 12.0 * k for k in range(1, 11)] + [last]
    times = sorted(set(times))
    T = 1.0
    obs = np.asarray(times, dtype=float)
    legacy = obs.copy() if obs[-1] == T else np.concatenate([obs, [T]])
    all_times, dt, idx = _discrete_grid(kind, times, T)
    assert dt.tobytes() == np.diff(np.concatenate([[0.0], legacy])).tobytes()
    if kind != "single_sharkfin":
        assert all_times.tobytes() == legacy.tobytes()
    np.testing.assert_array_equal(idx, np.searchsorted(legacy, obs))
