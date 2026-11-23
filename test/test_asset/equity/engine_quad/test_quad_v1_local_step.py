"""QUAD V1 prices a step its lattice cannot resolve with a local moment-matched stencil (sub-resolution run S3a).

The lattice is sized for the whole maturity; a step shorter than a cell (a knock-in minutes after a knock-out, two
knock-in dates seconds apart) used to alias its sampled kernel (contract clock: refused). Such a step is now applied as
``V'_i = disc * (w_- V_{i-1} + w_0 V_i + w_+ V_{i+1})`` with weights matching the step's mass, mean and variance
(spec "Decisions after S1", C1). Only steps the engine refused before take this path, so every contract it priced
before is bit-identical (bitwise priority). Chains of local steps whose accumulated error could exceed the accuracy
target are refused (AC5d); the first step from the valuation instant stays refused until S3b.

References: ``data/subresolution_v1_fixture.json`` (S1 demos): the increment reference -- the same engine at the
coincident booking and at resolved gaps, fitted in the gap -- with its half-width.
"""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from quantark.asset.equity.engine.quad import SnowballQuadEngine, UnderResolvedStepError
from quantark.asset.equity.engine.quad.local_step import (
    local_bridge_hit,
    local_step_apply,
    local_step_transpose,
    local_step_weights,
)
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.contract_clock import schedule_times
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

FIXTURE = json.loads((Path(__file__).parent / "data" / "subresolution_v1_fixture.json").read_text())
SECOND = 1.0 / (365.0 * 86400.0)
MINUTE = 60.0 * SECOND


def env(vol=0.2, spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(vol), rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 9, 11),
    )


def snowball(ko_times, ki_times, *, continuous=False, maturity=None):
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0,
        maturity=ko_times[-1] if maturity is None else maturity,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(ko_times), ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS if continuous else ObservationType.DISCRETE,
            ki_observation_dates=None if continuous else list(ki_times),
        ),
    )


QUARTERS = (0.25, 0.5, 0.75, 1.0)


def two_event(gap):
    return snowball(QUARTERS, (0.25, 0.5 + gap, 0.75, 1.0))


def two_event_rungs():
    return [r for r in FIXTURE["rungs"] if r["case"] == "two_event" and r["v1_today"] != "priced"]


def within(value, ref, hw, rel, floor):
    return abs(value - ref) <= rel * abs(ref) + hw + floor


class TestStencil:
    @pytest.mark.parametrize("alpha, beta, tau", [(0.0, 0.0, 1e-6), (-0.6, 0.4, 3e-7), (0.9, 1.1, 2e-6)])
    def test_weights_carry_the_mass_and_the_mean_of_the_step(self, alpha, beta, tau):
        h = 4e-3
        w = np.array(local_step_weights(h, alpha, beta, tau))
        disc = math.exp(-(beta - alpha * alpha) * tau)
        offsets = np.array([-1.0, 0.0, 1.0]) * h
        assert w.sum() == pytest.approx(disc, rel=1e-14)
        assert (w * offsets).sum() / disc == pytest.approx(2.0 * alpha * tau, rel=1e-12, abs=1e-18)
        assert (w * offsets ** 2).sum() / disc == pytest.approx(2.0 * tau, rel=1e-12)

    def test_negative_weight_is_refused_never_applied(self):
        with pytest.raises(UnderResolvedStepError):
            local_step_weights(1e-3, 0.0, 0.0, 1e-6)              # s/h = 1.41: not a sub-resolution step

    @pytest.mark.parametrize("bridge", [False, True])
    def test_transpose_is_the_adjoint(self, bridge):
        rng = np.random.default_rng(7)
        grid = np.linspace(-1.0, 1.0, 201)
        h = float(grid[1] - grid[0])
        w = local_step_weights(h, 0.3, 0.2, 1e-6)
        coeffs = local_bridge_hit(grid, -0.05, 2e-6, False) if bridge else None
        v, d = rng.normal(size=(2, 201))
        assert np.dot(local_step_apply(v, w, coeffs), d) == pytest.approx(
            np.dot(v, local_step_transpose(d, w, coeffs)), rel=1e-13)

    def test_rows_are_independent(self):
        """Stacked statistics rows must equal the 1-D price bitwise (pv == price())."""
        rng = np.random.default_rng(3)
        rows = rng.normal(size=(4, 101))
        w = local_step_weights(0.02, 0.1, 0.05, 1e-5)
        stacked = local_step_apply(rows, w)
        for i in range(4):
            assert np.array_equal(stacked[i], local_step_apply(rows[i], w))


class TestTwoEventLadder:
    """AC2 (V1): every gap the engine refused prices within 1e-4 + half-width of the increment reference."""

    @pytest.mark.parametrize("rung", two_event_rungs(), ids=lambda r: f"{r['vol']}-{r['rung']}")
    def test_price_matches_the_increment_reference(self, rung):
        tol = FIXTURE["tolerance"]
        value = SnowballQuadEngine(QuadParams()).price(two_event(rung["x_years"]), env(rung["vol"]))
        assert within(value, rung["ref_price"], rung["hw_price"], 1e-4, 1e-8 * 100.0), (value, rung["ref_price"])
        assert tol  # fixture carries its own tolerance block

    @pytest.mark.parametrize("rung", [r for r in two_event_rungs() if r["delta_gated"]],
                             ids=lambda r: f"{r['vol']}-{r['rung']}")
    def test_delta_matches_the_increment_reference(self, rung):
        """Every vol the fixture gates (5%, 20%, 60%): central relative 1% spot bumps. The reference is the fixture's
        delta increment over its 2e-10 y rung, anchored on the engine's own coincident-booking delta (gap 0, no local
        step, bitwise today's operator): at vol 5% the fit's intercept over 1-8 d gaps (s/h 0.87 at 1 d, barely
        resolved) is 6.9e-4 from that exact gap-0 value (1.3e-3 relative); at 20% and 60% within 4.1e-12."""
        e = SnowballQuadEngine(QuadParams())
        spot = rung["spot"]

        def delta(product):
            up = e.price(product, env(rung["vol"], spot * 1.01))
            return (up - e.price(product, env(rung["vol"], spot * 0.99))) / (0.02 * spot)

        base = next(r for r in two_event_rungs() if r["vol"] == rung["vol"] and r["rung"] == "2e-10y")
        ref = delta(two_event(0.0)) + (rung["ref_delta"] - base["ref_delta"])
        value = delta(two_event(rung["x_years"]))
        assert within(value, ref, rung["hw_delta"] + base["hw_delta"], 1e-3, 1e-6 * 100.0), (value, ref)

    def test_resolved_gap_is_bitwise_unchanged(self):
        """8.8 hours apart at vol 20% resolves (s/h ~1.05): today's operator, untouched -- the value the engine gave
        before the sub-resolution run (aa6d5a2f), bit for bit."""
        p = two_event(1e-3)
        e = SnowballQuadEngine(QuadParams())
        assert e.price(p, env()) == 98.91842991352458
        assert not getattr(e, "_last_local_steps", ())

    def test_event_statistics_pv_is_the_price(self):
        p = two_event(5 * MINUTE)
        e = SnowballQuadEngine(QuadParams())
        stats = e.calculate_event_stats(p, env())
        assert stats.pv == e.price(p, env())

    def test_two_knock_in_dates_seconds_apart_price_within_the_sizing_cap(self):
        """Rule (i): sizing for a 1 s knock-in spacing would exceed ``max_adaptive_grid_points``; the short step is
        priced locally on a grid sized from the other steps instead of refused. The two dates observe different levels
        (74 then 75): the same level twice is refused (``TestSameBarrier``)."""
        dates = (0.25, 0.5, 0.5 + SECOND, 0.75, 1.0)
        p = SnowballOption(
            initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=1.0,
            barrier_config=BarrierConfig(
                ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(QUARTERS),
                ki_barrier=[75.0, 74.0, 75.0, 75.0, 75.0], ki_observation_type=ObservationType.DISCRETE,
                ki_observation_dates=list(dates)),
        )
        e = SnowballQuadEngine(QuadParams())
        value = e.price(p, env())
        assert e._last_local_steps
        assert value == pytest.approx(e.price(snowball(QUARTERS, (0.25, 0.5, 0.75, 1.0)), env()), rel=1e-4)


class TestSameBarrier:
    """CP1: two observations of the SAME barrier a local step apart. The crossing mass between them lives in a layer
    of width s at the barrier, which the stencil cannot carry (measured against grids that resolve the step: knock-out
    pair, vol 5%, 5 min: knock-out probability increment off by 1.07e-3; vol 60%: price 1.5e-4; knock-in pair, vol 60%:
    knock-in probability 3.0e-4). Refused, never mispriced; MC, the PDE or ``auto_converge`` price them."""

    @pytest.mark.parametrize("pair", ["ko", "ki"])
    def test_a_local_step_between_two_observations_of_one_barrier_is_refused(self, pair):
        dates = (0.25, 0.5, 0.5 + 5 * MINUTE, 0.75, 1.0)
        p = snowball(dates, QUARTERS) if pair == "ko" else snowball(QUARTERS, dates)
        engine = SnowballQuadEngine(QuadParams())
        with pytest.raises(UnderResolvedStepError, match="same barrier"):
            engine.price(p, env())
        with pytest.raises(UnderResolvedStepError, match="same barrier"):
            engine.calculate_event_stats(p, env())

    def test_two_knock_out_levels_within_the_crossing_layer_are_one_barrier(self):
        """CP2: levels a hair apart (103 / 103.001 at vol 60%, 5 min) cut the same layer of width ~ s."""
        from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule

        g = 5 * MINUTE
        sched = ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=b) for t, b in
                                             ((0.25, 103.0), (0.5, 103.0), (0.5 + g, 103.001), (0.75, 103.0),
                                              (1.0, 103.0))])
        p = SnowballOption(
            initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=1.0,
            barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_schedule=sched,
                                         ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
                                         ki_observation_dates=list(QUARTERS)))
        with pytest.raises(UnderResolvedStepError, match="same barrier"):
            SnowballQuadEngine(QuadParams()).price(p, env(0.6))

    def test_auto_converge_prices_it_on_a_rung_that_resolves_the_step(self):
        p = snowball((0.25, 0.5, 0.5 + 5 * MINUTE, 0.75, 1.0), QUARTERS)
        value = SnowballQuadEngine(QuadParams(auto_converge=True)).price(p, env())
        ref = SnowballQuadEngine(QuadParams(grid_points=16001)).price(p, env())
        assert within(value, ref, 0.0, 1e-4, 1e-6), (value, ref)


class TestBackstops:
    def test_a_chain_of_local_steps_is_refused(self):
        """AC5d: daily knock-in dates over five years at vol 5% leave thousands of local steps (S1: C1 -5.9e-4)."""
        p = snowball(schedule_times(5.0, 20), schedule_times(5.0, 5 * 365))
        with pytest.raises(UnderResolvedStepError, match="local steps"):
            SnowballQuadEngine(QuadParams()).price(p, env(0.05))

    def test_unresolved_first_step_is_priced_by_the_exact_first_step(self):
        """S3b prices the valuation -> first-event step exactly at the spot (``first_step``) instead of refusing it;
        the local stencil is never the first step's price."""
        p = snowball((5 * MINUTE,) + QUARTERS, QUARTERS)
        e = SnowballQuadEngine(QuadParams())
        value = e.price(p, env(0.2, 103.0))
        assert e._last_first_step is not None
        ref = next(r for r in FIXTURE["rungs"]
                   if r["case"] == "first_ko" and r["vol"] == 0.2 and r["rung"] == "5min" and r["dist"] == "on")
        assert within(value, ref["ref_price"], ref["hw_price"], 1e-4, 1e-6), (value, ref["ref_price"])

    @pytest.mark.parametrize("gap, allowed", [(1e-3, False), (5 * MINUTE, False), (SECOND, True)])
    def test_auto_converge_statistics_run_the_ladder_that_priced(self, gap, allowed):
        """The statistics re-solve on the accepted grid with the accepted ladder's local-step switch: the first ladder
        (no local steps) for a contract it converges, the second for one every first-ladder rung refused."""
        p = two_event(gap)
        e = SnowballQuadEngine(QuadParams(auto_converge=True))
        stats = e.calculate_event_stats(p, env())
        assert e._last_convergence_info["local_steps_allowed"] is allowed
        assert stats.pv == e.price(p, env())
        assert abs(stats.reconciliation_error) <= 1e-12 * stats.pv

    def test_auto_converge_keeps_today_ladder_and_prices_what_it_refused(self):
        """Coarse rungs still skip an unresolved step (contracts that converge today are bit-identical); a contract
        every rung refused is priced by a second ladder with local steps."""
        p = two_event(5 * MINUTE)
        value = SnowballQuadEngine(QuadParams(auto_converge=True)).price(p, env())
        ref = next(r for r in two_event_rungs() if r["vol"] == 0.2 and r["rung"] == "5min")
        assert within(value, ref["ref_price"], ref["hw_price"], 1e-4, 1e-6), (value, ref["ref_price"])
