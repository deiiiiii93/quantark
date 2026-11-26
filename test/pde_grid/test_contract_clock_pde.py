"""The PDE decides "the same instant" by the contract clock's rule (T6; spec AC6, AC9, R6).

The PDE used to deduplicate event times with a RELATIVE ``is_close`` (1e-9 of the time: 9e-10 y at 0.9), to refuse an
event within that width of ``tau`` as the terminal endpoint, to look its nodes and records up at the same width (or
at 1e-6 y, ``_aligned_time_index``), and to read a settlement within ``np.isclose`` (rel 1e-5) of its observation as
immediate. Every one of those time-identity tests is now ``same_instant`` / ``merge_time_nodes`` at
``Tolerance.ZERO`` (1e-10 y), absolute. The 0 side stays EXACT: only t == 0.0 is the valuation instant, and a lone
event a millisecond ahead is a positive event (review 2026-09-21 R1).
"""
from datetime import datetime
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import _settlement_scale
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, request_geometry
from quantark.asset.equity.engine.pde.core.statistics import adjoint as adjoint_stats
from quantark.asset.equity.engine.pde.core.statistics import terminal as terminal_stats
from quantark.asset.equity.engine.pde.core.statistics.ko_reset import KOResetStatistics
from quantark.asset.equity.engine.pde.core.statistics.snowball import SnowballStatistics
from quantark.asset.equity.engine.pde.dcn_statistics import DCNForwardResult
from quantark.asset.equity.engine.pde.grid import GridRequest, build_time, resolve_config
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import BarrierOption, ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.execution.errors import CapabilityError
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierType, ObservationAggregation, ObservationType, OptionType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_close, merge_time_nodes

SECOND = 1.0 / (365.0 * 86400.0)
MILLISECOND = SECOND / 1000.0                      # 3.17e-11 y: below Tolerance.ZERO
UP = lambda t: float(np.nextafter(t, np.inf))      # noqa: E731
DOWN = lambda t: float(np.nextafter(t, -np.inf))   # noqa: E731
GAP = 5e-10                                        # beyond Tolerance.ZERO, inside the old relative width at 0.9


def request(event_times=(0.25, 0.5, 0.75), extra_times=(), tau=1.0):
    return GridRequest(tau=tau, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None, hard_upper=None,
                       event_times=tuple(event_times), extra_times=tuple(extra_times), day_count=252)


def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2), rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 9, 11),
    )


def snowball(ko_times, ki_times, maturity=None):
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0,
        maturity=ko_times[-1] if maturity is None else maturity,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(ko_times),
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ki_times),
        ),
    )


def records(*times):
    return [SimpleNamespace(observation_time=t) for t in times]


def test_gap_is_inside_the_retired_relative_width():
    """The pairs below are ones the old rule merged (precondition of every 'now two' test)."""
    assert is_close(0.9, 0.9 + GAP) and is_close(1.0 - GAP, 1.0)


class TestGridRequest:
    def test_pair_the_relative_rule_merged_stays_two_events(self):
        assert request(event_times=(0.9, 0.9 + GAP)).event_times == (0.9, 0.9 + GAP)

    def test_one_ulp_pair_is_one_event_the_last_of_its_run(self):
        assert request(event_times=(0.25, 0.5, UP(0.5))).event_times == (0.25, UP(0.5))

    def test_dedup_is_the_shared_function(self):
        times = (0.25, 0.5, UP(0.5), 0.9, 0.9 + GAP, UP(UP(0.75)), 0.75)
        assert request(event_times=times).event_times == tuple(float(t) for t in merge_time_nodes(times))

    def test_event_beyond_zero_below_tau_is_interior(self):
        assert request(event_times=(0.5, 1.0 - GAP)).event_times == (0.5, 1.0 - GAP)

    def test_event_one_ulp_below_tau_is_the_terminal_endpoint(self):
        with pytest.raises(ValidationError, match="strictly inside"):
            request(event_times=(0.5, DOWN(1.0)))

    def test_zero_side_stays_exact(self):
        """Only t == 0.0 is the valuation instant; a lone event 1 ms ahead is a positive event (review R1)."""
        assert request(event_times=(MILLISECOND,)).event_times == (MILLISECOND,)
        assert request(event_times=(UP(0.0),)).event_times == (UP(0.0),)
        with pytest.raises(ValidationError, match="strictly inside"):
            request(event_times=(0.0, 0.5))

    def test_extra_within_zero_of_an_event_is_served_by_the_event(self):
        r = request(event_times=(0.9,), extra_times=(UP(0.9), 0.9 + GAP))
        assert r.extra_times == (0.9 + GAP,)

    def test_extras_dedup_by_the_shared_function(self):
        assert request(event_times=(), extra_times=(0.3, UP(0.3), 0.3 + GAP)).extra_times == (UP(0.3), 0.3 + GAP)


class TestStepAt:
    def test_a_time_within_zero_finds_its_node_and_beyond_zero_does_not(self):
        layout = build_time(request(event_times=(0.5,)), resolve_config("standard", None))
        assert layout.step_at(DOWN(0.5)) == layout.step_at(0.5)
        with pytest.raises(KeyError):
            layout.step_at(0.5 + 2e-10)                 # relative width at 0.5 was 5e-10

    def test_two_events_beyond_zero_are_two_nodes(self):
        layout = build_time(request(event_times=(0.9, 0.9 + GAP)), resolve_config("standard", None))
        first, second = layout.step_at(0.9), layout.step_at(0.9 + GAP)
        assert second > first and layout.t[first] == 0.9 and layout.t[second] == 0.9 + GAP


class TestLookups:
    def test_aligned_time_index_is_one_to_one(self):
        t_vec = np.array([0.0, 0.5, 0.5 + 1e-8, 1.0])
        solver = SnowballPDESolver(PDEParams())
        assert solver._aligned_time_index(t_vec, 0.5 + 1e-8, "KI") == 2
        assert solver._aligned_time_index(t_vec, UP(0.5), "KO") == 1
        with pytest.raises(ValidationError, match="does not align"):
            solver._aligned_time_index(t_vec, 0.5 + 5e-9, "KI")

    def test_find_record_at_time(self):
        recs = records(0.9, 0.9 + GAP)
        assert SnowballPDESolver._find_record_at_time(recs, 0.9 + GAP) is recs[1]
        assert SnowballPDESolver._find_record_at_time(recs, DOWN(0.9)) is recs[0]

    def test_terminal_statistics_position(self):
        recs = records(0.9, 0.9 + GAP)
        assert terminal_stats._position(recs, recs[1]) == 1

    def test_snowball_statistics_column(self):
        # _shared_cols: same-instant groups (0.16.0 D14) — two sub-resolution records are NOT one instant
        fake = SimpleNamespace(_col_by_time={}, _shared_cols={}, ko_records=records(0.9, 0.9 + GAP),
                               layout=SimpleNamespace(ko=[7, 8]))
        assert SnowballStatistics.column_of_record(fake, 0.9 + GAP) == 8
        assert SnowballStatistics.column_of_record(fake, UP(0.9)) == 7

    def test_ko_reset_statistics_column(self):
        fake = SimpleNamespace(ko_records=records(0.9, 0.9 + GAP), post_records=records(),
                               layout=SimpleNamespace(ko=[7, 8], post_ko=[]))
        # the real same-instant lookup (0.16.0 D14) runs on the fake: two sub-resolution records stay two instants
        fake._shared_instants = MethodType(KOResetStatistics._shared_instants, fake)
        assert KOResetStatistics.column_of_record(fake, 0.9 + GAP) == 8

    def test_adjoint_column_tau_side(self):
        layout = SimpleNamespace(time=SimpleNamespace(actual_steps=99, step_at=lambda t: 42))
        assert adjoint_stats._column_of(layout, 1.0 - GAP, 1.0) == 42
        assert adjoint_stats._column_of(layout, DOWN(1.0), 1.0) == 99
        assert adjoint_stats._column_of(layout, 0.0, 1.0) == 0

    def test_dcn_ki_row(self):
        fake = SimpleNamespace(rows=SimpleNamespace(ki_times=np.array([0.9, 1.0])))
        assert DCNForwardResult.ki_row_at(fake, DOWN(0.9)) == 0
        with pytest.raises(PricingError, match="not a daily KI date"):
            DCNForwardResult.ki_row_at(fake, 0.9 - 3e-10)


class TestBarrierVolSettlement:
    @staticmethod
    def barrier(settlement):
        schedule = ObservationSchedule(
            records=[ObservationRecord(observation_time=0.5, settlement_time=settlement, barrier=130.0, payoff=10.0)],
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT)
        return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                             barrier_type=BarrierType.UP_OUT, rebate=10.0, pay_at_hit=True, maturity=1.0,
                             observation_type=ObservationType.DISCRETE, observation_schedule=schedule)

    def test_settlement_at_the_observation_instant_is_immediate(self):
        assert _settlement_scale(SnowballPDESolver(), self.barrier(UP(0.5)), env()) == 1.0

    def test_settlement_beyond_zero_after_the_observation_is_delayed(self):
        with pytest.raises(CapabilityError, match="first-hit"):
            _settlement_scale(SnowballPDESolver(), self.barrier(0.5 + GAP), env())


class TestPricing:
    def test_ko_beyond_zero_below_maturity_is_an_interior_event(self):
        """R6: a KO 5e-10 y before maturity used to be refused as the terminal endpoint; it is an interior event
        now, and prices as the KO booked at maturity does (one 5e-10 y step apart)."""
        solver = SnowballPDESolver(PDEParams())
        split = solver.price(snowball([0.25, 0.5, 0.75, 1.0 - GAP], [0.5], maturity=1.0), env())
        at_maturity = solver.price(snowball([0.25, 0.5, 0.75, 1.0], [0.5]), env())
        assert np.isfinite(split) and split == pytest.approx(at_maturity, rel=1e-6)

    def test_intraday_and_sub_second_ki_price_against_mc(self):
        """AC6 / AC9 (PDE part): KO at 0.5 with discrete KI 1e-8 y and 1 s after it. The grid request carries both
        KI instants as events, and the PDE agrees with MC within 3 standard errors."""
        ki = [0.5 + 1e-8, 0.5 + SECOND]
        product = snowball([0.25, 0.5, 0.75, 1.0], ki)
        solver = SnowballPDESolver(PDEParams())
        grid_request, _ = request_geometry(solver, product, env(), OutputRequest(Operation.PRICE))
        assert {0.5, *ki} <= set(grid_request.event_times)

        pde = solver.price(snowball([0.25, 0.5, 0.75, 1.0], ki), env())
        mc_engine = SnowballMCEngine(params=MCParams(seed=11, num_paths=200_000))
        mc = mc_engine.price(snowball([0.25, 0.5, 0.75, 1.0], ki), env())
        assert np.isfinite(pde)
        assert abs(pde - mc) < 3.0 * mc_engine.get_last_result().std_error
