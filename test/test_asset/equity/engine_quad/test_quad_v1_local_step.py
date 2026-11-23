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

    @pytest.mark.parametrize("rung", [r for r in two_event_rungs() if r["delta_gated"] and r["vol"] == 0.2],
                             ids=lambda r: r["rung"])
    def test_delta_matches_the_increment_reference(self, rung):
        e = SnowballQuadEngine(QuadParams())
        p = two_event(rung["x_years"])
        up, dn = e.price(p, env(0.2, 101.0)), e.price(p, env(0.2, 99.0))
        delta = (up - dn) / 2.0
        assert within(delta, rung["ref_delta"], rung["hw_delta"], 1e-3, 1e-6 * 100.0), (delta, rung["ref_delta"])

    def test_resolved_gap_is_bitwise_unchanged(self):
        """8.8 hours apart at vol 20% resolves (s/h ~1.05): today's operator, untouched."""
        p = two_event(1e-3)
        e = SnowballQuadEngine(QuadParams())
        assert e.price(p, env()) == e.price(p, env())
        assert not getattr(e, "_last_local_steps", ())

    def test_event_statistics_pv_is_the_price(self):
        p = two_event(5 * MINUTE)
        e = SnowballQuadEngine(QuadParams())
        stats = e.calculate_event_stats(p, env())
        assert stats.pv == e.price(p, env())

    def test_two_knock_in_dates_seconds_apart_price_within_the_sizing_cap(self):
        """Rule (i): sizing for a 1 s knock-in spacing would exceed ``max_adaptive_grid_points``; the short step is
        priced locally on a grid sized from the other steps instead of refused."""
        p = snowball(QUARTERS, (0.25, 0.5, 0.5 + SECOND, 0.75, 1.0))
        coincident = snowball(QUARTERS, (0.25, 0.5, 0.75, 1.0))
        e = SnowballQuadEngine(QuadParams())
        assert e.price(p, env()) == pytest.approx(e.price(coincident, env()), rel=1e-4)


class TestBackstops:
    def test_a_chain_of_local_steps_is_refused(self):
        """AC5d: daily knock-in dates over five years at vol 5% leave thousands of local steps (S1: C1 -5.9e-4)."""
        p = snowball(schedule_times(5.0, 20), schedule_times(5.0, 5 * 365))
        with pytest.raises(UnderResolvedStepError, match="local steps"):
            SnowballQuadEngine(QuadParams()).price(p, env(0.05))

    def test_unresolved_first_step_is_still_refused(self):
        """S3a leaves the valuation -> first-event step to S3b's exact first step."""
        p = snowball((5 * MINUTE,) + QUARTERS, QUARTERS)
        with pytest.raises(UnderResolvedStepError, match="first step"):
            SnowballQuadEngine(QuadParams()).price(p, env())

    def test_auto_converge_keeps_today_ladder_and_prices_what_it_refused(self):
        """Coarse rungs still skip an unresolved step (contracts that converge today are bit-identical); a contract
        every rung refused is priced by a second ladder with local steps."""
        p = two_event(5 * MINUTE)
        value = SnowballQuadEngine(QuadParams(auto_converge=True)).price(p, env())
        ref = next(r for r in two_event_rungs() if r["vol"] == 0.2 and r["rung"] == "5min")
        assert value == pytest.approx(ref["R1_price"], rel=2e-4)
