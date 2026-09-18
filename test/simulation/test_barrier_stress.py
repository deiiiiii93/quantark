"""The gate's deterministic barrier stress set.

The visited-state reservoir samples what a run happened to touch, so
whether it ever lands beside the knock-in barrier is a matter of luck.
The stress set is designed instead: fixed distances either side of the
barrier, both lifecycle states, on the knock-in observation dates
themselves, with the parameters pushed to the worst corner of their
buckets.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.config import GateConfig
from quantark.backtest.simulation.paths.market_path import trading_calendar
from quantark.backtest.simulation.pricing.base import GateScale, bucket_centre
from quantark.backtest.simulation.pricing.surface import LifeSurfacePricer

from .conftest import INCEPTION, RATE, SPOT, pde_engine_config

DATES = trading_calendar(pd.Timestamp("2024-01-02").date(), 8)
#: The shared short fixture puts its knock-in 25% away, which six days of
#: 22% vol cannot reach, so its life surface never spans the barrier. A
#: stress set has to sit where the surface actually lives.
KI_BARRIER = 0.97 * SPOT


def near_barrier_snowball(ko_days=(2, 5), ki_days=(1, 3)):
    from quantark.asset.equity.product.option import create_standard_snowball
    from quantark.util.enum import ObservationType

    product = create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=6 / 365.0, contract_multiplier=1.0,
        ko_barrier=1.03 * SPOT, ki_barrier=KI_BARRIER, ko_rate=0.20,
        num_observations=len(ko_days), ko_observation_dates=[d / 365.0 for d in ko_days],
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[d / 365.0 for d in ki_days], include_principal=True,
    )
    # The annualized KO coupon needs an inception to accrue from, or ageing
    # would shift the observation time it uses as its accrual factor.
    product.initial_date = INCEPTION
    return product
SCALE = GateScale(unit_notional=SPOT, hands_per_unit_delta=5.0)
VOL, Q = 0.22, 0.05
LOOSE = GateConfig(sample_states=0, pv_tolerance_bp=1.0e6, delta_tolerance_hands=1.0e6)


def _pricer(gate=None, vol_step=0.01, q_step=0.0025):
    return LifeSurfacePricer(
        near_barrier_snowball(), engine_config=pde_engine_config(), start_date=DATES[0], dates=DATES,
        underlying="CSI1000", vol_step=vol_step, q_step=q_step, surface_cache_bytes=200_000_000,
        gate=gate if gate is not None else LOOSE,
    )


def test_the_stress_set_brackets_the_knock_in_barrier():
    cases = _pricer().barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    assert cases, "the stress set must not be empty for a product with a knock-in"
    distances = [c.distance for c in cases]
    assert min(distances) < 0.0 < max(distances)
    for case in cases:
        assert float(case.states.spot[0]) == pytest.approx(KI_BARRIER * (1.0 + case.distance))
        assert len(case.states) == 1


def test_no_designed_state_is_one_the_lifecycle_cannot_produce():
    """A run steps the lifecycle before it prices.

    So on a knock-in observation date a path below the barrier is already
    marked knocked in, and an alive state there is unreachable. Gating on
    an unreachable state measures nothing, and near expiry it measures a
    very large nothing.
    """
    cases = _pricer().barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    corners = {(c.distance > 0.0, bool(c.states.knocked_in[0])) for c in cases}
    assert corners == {(True, True), (True, False), (False, True)}
    assert (False, False) not in corners


def test_the_attribution_carries_a_relative_scale_beside_the_absolute_one():
    """Two hands means one thing at 2,600 hands of delta and another at 38,000."""
    pricer = _pricer()
    report = pricer.verify_stress(pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q),
                                  LOOSE, SCALE, attribute=True)
    for row in report.attribution:
        assert row["delta_gap_rel"] == pytest.approx(
            row["delta_gap_hands"] / abs(row["exact_delta_hands"]))


def test_the_stress_set_sits_on_the_knock_in_observation_dates():
    pricer = _pricer()
    product = pricer.product
    observations = {round(float(t), 12) for t in product.barrier_config.ki_observation_dates}
    for case in pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q):
        assert round(pricer.elapsed(case.states.day_index), 12) in observations


def test_the_stress_set_is_the_same_every_time():
    a = _pricer().barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    b = _pricer().barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    assert [c.label for c in a] == [c.label for c in b]
    assert [float(c.states.spot[0]) for c in a] == [float(c.states.spot[0]) for c in b]


def test_the_parameters_sit_at_the_worst_corner_of_their_buckets():
    """A state sitting at a bucket centre would report no bucketing error at all."""
    pricer = _pricer(vol_step=0.01, q_step=0.0025)
    for case in pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q):
        for values, step in ((case.states.vol, pricer.vol_step), (case.states.q_T, pricer.q_step)):
            offset = abs(float(values[0]) - float(bucket_centre(values, step)[0]))
            assert offset >= 0.49 * step, (offset, step)


def test_the_attribution_splits_the_gap_without_losing_any_of_it():
    pricer = _pricer()
    cases = pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    report = pricer.verify_stress(cases, LOOSE, SCALE, attribute=True)
    assert report.mode == "life_surface_barrier"
    assert report.sampled == len(cases) and len(report.attribution) == len(cases)
    for row in report.attribution:
        parts = ("bucket", "propagation", "readout")
        assert sum(row[f"{p}_pv_bp"] for p in parts) == pytest.approx(row["pv_gap_bp"], abs=1e-9)
        assert sum(row[f"{p}_delta_hands"] for p in parts) == pytest.approx(
            row["delta_gap_hands"], abs=1e-9)


def test_the_report_is_the_worst_designed_state_not_the_average():
    pricer = _pricer()
    cases = pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    report = pricer.verify_stress(cases, LOOSE, SCALE, attribute=True)
    assert report.max_pv_gap_bp == pytest.approx(
        max(abs(r["pv_gap_bp"]) for r in report.attribution))
    assert report.max_delta_gap_hands == pytest.approx(
        max(abs(r["delta_gap_hands"]) for r in report.attribution))


def test_a_designed_state_outside_the_budget_fails_the_gate():
    pricer = _pricer()
    cases = pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    tight = GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0)
    assert not pricer.verify_stress(cases, tight, SCALE).passed
    assert pricer.verify_stress(cases, LOOSE, SCALE).passed


def test_the_stress_set_does_not_feed_the_visited_reservoir():
    """The gate must not certify itself on states it invented."""
    pricer = _pricer(gate=GateConfig(sample_states=8, pv_tolerance_bp=1e6,
                                     delta_tolerance_hands=1e6))
    pricer.verify_stress(pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q), LOOSE, SCALE)
    assert pricer.sample_visited() == []


def test_the_stress_set_can_be_turned_off():
    """An empty offset list is how a caller says it does not want the cost."""
    off = GateConfig(sample_states=0, pv_tolerance_bp=1e6, delta_tolerance_hands=1e6,
                     barrier_offsets=())
    assert _pricer(gate=off).barrier_stress_cases(vol=VOL, rate=RATE, q=Q) == []


def test_the_day_zero_gate_runs_the_designed_states_too():
    """A run must not start until the surface holds its budget at the barrier."""
    from .conftest import ensemble_config, make_market_path, surface_pricing
    from quantark.backtest.replay import ReplayProduct
    from quantark.backtest.simulation.engine import EnsembleBacktestEngine
    from quantark.backtest.simulation.pricing.base import GateFailure

    paths = make_market_path(n_paths=2, n_days=6)
    book = [ReplayProduct(product=near_barrier_snowball(), quantity=-1000.0,
                          position_id=1, has_lifecycle=True)]
    impossible = GateConfig(sample_states=0, pv_tolerance_bp=0.0,
                            delta_tolerance_hands=0.0, barrier_offsets=(0.002,))
    pricing = surface_pricing()
    cfg = ensemble_config(products=book,
                          pricing=dataclasses.replace(pricing, gate=impossible))
    with pytest.raises(GateFailure) as raised:
        EnsembleBacktestEngine(cfg).run(paths)
    assert "life_surface" in str(raised.value)


def test_a_scaled_time_grid_fails_the_surface_closed():
    """The gate compares the surface against exact solves that are mostly
    under the cap, so a surface whose fill was cut measures the cap, not
    itself.  It refuses, naming the knob."""
    from quantark.asset.equity.engine.pde.grid import GridConfig
    from quantark.asset.equity.param import PDEParams
    from quantark.util.exceptions import ValidationError

    pricer = LifeSurfacePricer(
        near_barrier_snowball(), engine_config=pde_engine_config(pde_params=PDEParams(grid=GridConfig(max_steps=10))),
        start_date=DATES[0], dates=DATES, underlying="CSI1000", vol_step=0.01, q_step=0.0025,
        surface_cache_bytes=200_000_000, gate=LOOSE,
    )
    with pytest.raises(ValidationError, match="max_steps"):
        pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q)


def test_the_attribution_rows_carry_the_market_each_leg_priced_at():
    """The exact leg prices at the state's own corner values and the surface
    at the bucket centre; a reader building an outside reference from a row
    must be able to see both."""
    pricer = _pricer(vol_step=0.01, q_step=0.0025)
    cases = pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    report = pricer.verify_stress(cases, LOOSE, SCALE)
    for row, case in zip(report.attribution, cases):
        state = case.states
        assert row["vol"] == float(state.vol[0])
        assert row["q"] == float(state.q_T[0])
        assert row["rate"] == float(state.rate[0])
        assert row["surface_vol"] == float(bucket_centre(state.vol, pricer.vol_step)[0])
        assert row["surface_q"] == float(bucket_centre(state.q_T, pricer.q_step)[0])
        assert row["surface_rate"] == float(bucket_centre(state.rate, pricer.q_step)[0])
        assert abs(row["vol"] - row["surface_vol"]) >= 0.49 * pricer.vol_step


def test_a_designed_state_is_judged_on_its_own_delta_scale():
    """Beside the barrier the delta runs an order of magnitude above a
    typical state, so the same absolute hand budget is a far tighter rule
    there.  The relative allowance is what lets the two be judged alike."""
    from dataclasses import replace

    pricer = _pricer(vol_step=0.01, q_step=0.0025)
    cases = pricer.barrier_stress_cases(vol=VOL, rate=RATE, q=Q)
    strict = GateConfig(sample_states=0, pv_tolerance_bp=1e9, delta_tolerance_hands=0.0,
                        barrier_offsets=LOOSE.barrier_offsets, barrier_dates=LOOSE.barrier_dates)
    report = pricer.verify_stress(cases, strict, SCALE)
    assert not report.passed
    worst = max(abs(row["delta_gap_rel"]) for row in report.attribution
                if np.isfinite(row["delta_gap_rel"]))

    assert pricer.verify_stress(cases, replace(strict, delta_tolerance_rel=worst * 1.01), SCALE).passed
    assert not pricer.verify_stress(cases, replace(strict, delta_tolerance_rel=worst * 0.99), SCALE).passed
