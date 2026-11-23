"""QUAD V1 decides "the same instant" by the contract clock's rule (T5; spec AC6, AC9).

V1 used to merge event times 1e-6 years apart (~31 s) and to look records up at the same width: intraday events one
second apart were one step, and once the merge tightens, a 1e-6 lookup would apply one record at two steps. Every V1
time-identity test is now ``same_instant`` (Tolerance.ZERO) or ``merge_time_nodes``.
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import NumericalError

SECOND = 1.0 / (365.0 * 86400.0)
UP = lambda t: float(np.nextafter(t, np.inf))      # noqa: E731


def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2), rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 9, 11),
    )


def snowball(ko_times, ki_times):
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=ko_times[-1],
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(ko_times),
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ki_times),
        ),
    )


def engine():
    return SnowballQuadEngine(QuadParams())


class TestMergeTimes:
    @pytest.mark.parametrize("gap", [1e-8, SECOND])
    def test_distinct_instants_stay_two_steps(self, gap):
        assert engine()._merge_times([0.5, 1.0], [0.5 + gap, 1.0], 1.0) == [0.5, 0.5 + gap, 1.0]

    def test_float_noise_is_one_step_maturity_pinned(self):
        assert engine()._merge_times([0.5, UP(1.0)], [UP(0.5)], 1.0) == [UP(0.5), 1.0]

    def test_engine_knot_never_displaces_a_contract_instant(self):
        """D2: a continuous-KI knot a hair after a contract instant snaps onto it."""
        assert engine()._merge_times([0.25, UP(0.5), 0.5, 1.0], (), 1.0, anchors=[0.5, 1.0]) == [0.25, 0.5, 1.0]

    def test_record_lookup_is_one_to_one(self):
        """AC9: with the KO at 0.5 and the KI at 0.5 + 1e-8, the step at the KI time finds no KO record."""
        records = [ObservationRecord(observation_time=0.5), ObservationRecord(observation_time=1.0)]
        assert engine()._match_record(0.5 + 1e-8, records) is None
        assert engine()._match_record(UP(0.5), records) is records[0]

    def test_settlement_node_a_hair_after_its_observation_is_inserted(self):
        records = [ObservationRecord(observation_time=0.5, settlement_time=0.5 + 1e-8)]
        assert engine()._insert_settlement_times([0.5, 1.0], records, 1.0) == [0.5, 0.5 + 1e-8, 1.0]


class TestKernelResolution:
    """V1 samples each step's Gaussian kernel on a lattice sized for the whole maturity (refined only for the knock-in
    spacing, ``_resolve_grid_points``). A step whose kernel spans too few cells loses mass by aliasing,
    ``2 exp(-2 pi^2 (s/h)^2)`` (measured: s/h 0.722 -> 6.7e-5, 0.331 -> 0.23 and a 5504 price): such a step is
    refused, never priced, above the engine's own accuracy target ``convergence_rel_tol`` (owner ruling 2026-11-23,
    contract clock run). MC, PDE and QUAD V2 resolve it."""

    def test_step_within_the_accuracy_target_still_prices(self):
        """365 KI dates a year beside monthly KO dates leave steps of a fraction of a day (s/h ~0.72, mass error
        ~6.7e-5 < 1e-4): priced as before."""
        from quantark.asset.equity.product.option.contract_clock import schedule_times

        p = snowball(schedule_times(1.0, 12), schedule_times(1.0, 365))
        assert np.isfinite(engine().price(p, env()))

    def test_a_tighter_accuracy_target_tightens_the_guard(self):
        from quantark.asset.equity.product.option.contract_clock import schedule_times

        p = snowball(schedule_times(1.0, 12), schedule_times(1.0, 365))
        with pytest.raises(NumericalError, match="under-resolved"):
            SnowballQuadEngine(QuadParams(convergence_rel_tol=1e-6)).price(p, env())

    @pytest.mark.parametrize("gap", [1e-8, SECOND, 1e-4])
    def test_under_resolved_step_is_refused(self, gap):
        times = (0.25, 0.5, 0.75, 1.0)
        split = snowball(times, (0.25, 0.5 + gap, 0.75, 1.0))
        with pytest.raises(NumericalError, match=r"under-resolved"):
            engine().price(split, env())

    def test_resolved_short_step_prices_like_the_pde(self):
        """8.8 hours apart: s/h ~ 1.05, mass error ~ 8e-10 -- priced, and as the PDE prices it."""
        from quantark.asset.equity.engine.pde import SnowballPDESolver
        from quantark.asset.equity.param import PDEParams

        times = (0.25, 0.5, 0.75, 1.0)
        split = snowball(times, (0.25, 0.5 + 1e-3, 0.75, 1.0))
        assert engine().price(split, env()) == pytest.approx(SnowballPDESolver(PDEParams()).price(split, env()), abs=1e-3)

    @pytest.mark.parametrize("s_over_h, measured", [(1.047, 7.97e-10), (0.722, 6.70e-5), (0.574, 3.03e-3), (0.331, 0.230)])
    def test_mass_error_bound_matches_the_measured_aliasing(self, s_over_h, measured):
        assert SnowballQuadEngine._kernel_mass_error_bound(s_over_h) == pytest.approx(measured, rel=0.05)

    def test_float_noise_pair_prices_like_the_equal_booking(self):
        times = (0.25, 0.5, 0.75, 1.0)
        noisy = snowball(times, (0.25, UP(0.5), 0.75, 1.0))
        equal = snowball((0.25, UP(0.5), 0.75, 1.0), (0.25, UP(0.5), 0.75, 1.0))
        assert engine().price(noisy, env()) == engine().price(equal, env())

    def test_mc_prices_the_split_contract(self):
        """The instants V1 refuses are priced by MC (one node each, W17a)."""
        times = (0.25, 0.5, 0.75, 1.0)
        mc = SnowballMCEngine(params=MCParams(seed=11, num_paths=50_000))
        assert np.isfinite(mc.price(snowball(times, (0.25, 0.5 + SECOND, 0.75, 1.0)), env()))


class TestCheckpointTwoFixes:
    def test_auto_converge_refines_past_an_unresolved_rung(self):
        """CP2 #1: a KI 2e-4 y after a KO is refused at 1001 points but resolved at 2001/4001; the auto_converge
        ladder moves on instead of failing at its first rung, and lands next to the PDE."""
        from quantark.asset.equity.engine.pde import SnowballPDESolver
        from quantark.asset.equity.param import PDEParams

        times = (0.25, 0.5, 0.75, 1.0)
        split = snowball(times, (0.25, 0.5 + 2e-4, 0.75, 1.0))
        with pytest.raises(NumericalError, match="under-resolved"):
            engine().price(split, env())
        converged = SnowballQuadEngine(QuadParams(auto_converge=True)).price(split, env())
        assert converged == pytest.approx(SnowballPDESolver(PDEParams()).price(split, env()), abs=1e-3)

    def test_simpson_bound_adds_the_half_frequency_alias(self):
        """CP2 #4: measured 1.9e-2 at s/h 0.85 with Simpson weights against 1.3e-6 with trapezoid."""
        assert SnowballQuadEngine._kernel_mass_error_bound(0.85, "simpson") == pytest.approx(1.9e-2, rel=0.05)
        assert SnowballQuadEngine._kernel_mass_error_bound(0.85, "trapezoid") == pytest.approx(1.3e-6, rel=0.1)
