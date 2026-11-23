"""Every engine-side "same instant" test on TIMES uses the contract clock's rule (spec 2026-11-23, AC9; T5-T7).

Two parts:

1. Behaviour of the T7 sites outside the MC grids (cash legs, theta, SA-CVA exposure surfaces, the backtest life
   surface, settlement support): two times closer than ``Tolerance.ZERO`` are one instant; two times further apart
   are two, however close (the retired widths were relative 1e-9, abs 1e-9, abs 1e-6 and ``Tolerance.PROBABILITY``).
2. The AC9 site list: each (file, function) that T5 (QUAD V1), T6 (PDE) and T7 (MC + rest) moved onto the shared
   rule is read from source and must not contain the retired time tolerance it used to hold. A SITE list, not a
   regex over whole files: QUAD V1 still uses ``Tolerance.PRECISION`` for barrier LEVELS and BGK spacing gates.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError

GAP = 5e-10        # two instants: beyond Tolerance.ZERO (1e-10), inside every retired width at t ~ 1
NOISE = 5e-11      # one instant: inside Tolerance.ZERO

ROOT = Path(__file__).resolve().parents[1]


# ======================================================================================================================
# 1. Behaviour
# ======================================================================================================================

class TestAutocallableLegRebasedMatch:
    """``AutocallableCashLeg._per_event_contingent``: each REBASED event row matches exactly one schedule entry."""

    @staticmethod
    def _leg(obs):
        from quantark.cashleg.autocallable_leg import AutocallableCashLeg, AutocallableLegType, PvFormula
        from quantark.cashleg.base import LegDirection

        n = len(obs)
        return AutocallableCashLeg(
            direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.MARGIN, notional=1.0, rate=1.0,
            observation_schedule=tuple(obs), accrual_factors=tuple(float(k + 1) for k in range(n)),
            settlement_schedule=tuple(obs), terminal_accrual_factor=1.0, terminal_settlement_time=obs[-1],
            pv_formula=PvFormula.NORMAL,
        )

    @staticmethod
    def _dist(times, probs):
        from quantark.cashleg.event_distribution import EventType

        return EventType.KO, SimpleNamespace(stream_times={EventType.KO: np.asarray(times, dtype=float)},
                                             probabilities={EventType.KO: np.asarray(probs, dtype=float)})

    def test_two_entries_beyond_zero_apart_are_two_candidates(self):
        leg = self._leg([0.5, 0.5 + GAP, 1.0])
        event, dist = self._dist([0.5], [1.0])
        env = SimpleNamespace(get_discount_factor=lambda t: 1.0)
        assert leg._per_event_contingent(dist, event, env) == 1.0       # entry 0's accrual factor

    def test_a_row_within_zero_of_an_entry_matches_it(self):
        leg = self._leg([0.5, 1.0])
        event, dist = self._dist([0.5 + NOISE], [1.0])
        env = SimpleNamespace(get_discount_factor=lambda t: 1.0)
        assert leg._per_event_contingent(dist, event, env) == 1.0

    def test_a_row_beyond_zero_from_every_entry_is_refused(self):
        leg = self._leg([0.5, 1.0])
        event, dist = self._dist([0.5 + GAP], [1.0])
        env = SimpleNamespace(get_discount_factor=lambda t: 1.0)
        with pytest.raises(ValidationError, match="no schedule entry"):
            leg._per_event_contingent(dist, event, env)


class TestThetaShift:
    BUMP = 1.0 / 365.0

    def test_cash_paid_within_zero_of_the_new_as_of_is_still_in_the_value(self):
        """A relative 1e-9 width at t = 1/365 is 2.7e-12: a 5e-11 residue read as 'already paid'."""
        from quantark.asset.equity.riskmeasures.greeks.theta_shift import _carried_to_t1

        flat = SimpleNamespace(get_discount_factor=lambda t: 1.0)
        assert _carried_to_t1(2.0, self.BUMP - NOISE, flat, flat, self.BUMP) == 2.0
        assert _carried_to_t1(2.0, self.BUMP - GAP, flat, flat, self.BUMP) == 0.0

    def test_ko_reset_schedule_one_ending_within_zero_of_the_bump_is_not_over(self):
        from quantark.asset.equity.riskmeasures.greeks.theta_shift import _ko_reset_schedule_1_end
        from test_asset.equity.engine_mc.test_contract_clock_mc import env, ko_reset

        e = env()
        p = ko_reset([0.5, 1.0], [1.5, 2.0])
        p.validate()
        assert _ko_reset_schedule_1_end(p, e, 1.0 + NOISE) is None
        assert _ko_reset_schedule_1_end(p, e, 1.0 + GAP) is not None


class TestSacvaSurfaces:
    @pytest.mark.parametrize("module", ["snowball_surface", "phoenix_surface"])
    def test_grid_index_is_one_to_one_on_instants(self, module):
        import importlib

        grid_index = importlib.import_module(f"quantark.sacva.exposure.{module}")._grid_index
        times = np.array([0.0, 0.5, 0.5 + GAP, 1.0])
        assert grid_index(times, 0.5) == 1
        assert grid_index(times, 0.5 + GAP) == 2
        assert grid_index(times, 0.5 + NOISE) == 1
        with pytest.raises(ValidationError, match="not a unique node"):
            grid_index(times, 0.75)

    @staticmethod
    def _records(*pairs):
        from quantark.asset.equity.product.option.observation_schedule import ObservationRecord

        return [ObservationRecord(observation_time=o, settlement_time=s, barrier=103.0, payoff=1.0) for o, s in pairs]

    def test_snowball_settlement_beyond_zero_after_its_observation_is_delayed(self):
        """QUAD V1 inserts a settlement node beyond Tolerance.ZERO (T5); the surface must index it the same way."""
        from quantark.sacva.exposure.snowball_surface import _index_ko_leg

        settle = 0.5 + 1e-7
        times = np.array([0.0, 0.5, settle, 1.0])
        _, _, settle_idx = _index_ko_leg(self._records((0.5, settle), (1.0, 1.0)), times, 1.0)
        assert settle_idx == [2, 3]
        _, _, settle_idx = _index_ko_leg(self._records((0.5, 0.5 + NOISE)), np.array([0.0, 0.5, 1.0]), 1.0)
        assert settle_idx == [1]

    def test_snowball_settlement_bounds_are_instants(self):
        from quantark.sacva.exposure.snowball_surface import _validate_ko_leg

        _validate_ko_leg(self._records((0.5, 0.5 - NOISE), (1.0, 1.0 + NOISE)), "t", 1.0)
        with pytest.raises(ValidationError, match="precedes observation"):
            _validate_ko_leg(self._records((0.5, 0.5 - 1e-7)), "t", 1.0)
        with pytest.raises(ValidationError, match="post-maturity"):
            _validate_ko_leg(self._records((1.0, 1.0 + 1e-7)), "t", 1.0)

    def test_phoenix_settlement_beyond_zero_is_delayed_and_refused(self):
        from dataclasses import replace

        from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
        from quantark.asset.equity.param import QuadParams
        from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
        from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
        from quantark.sacva.exposure.phoenix_surface import build_phoenix_surface
        from test_asset.equity.engine_mc.test_contract_clock_mc import env

        p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                    coupon_rate=0.01, num_observations=2, memory_coupon=False)
        p.barrier_config = replace(p.barrier_config, ko_observation_dates=None, ko_observation_schedule=(
            ObservationSchedule(records=self._records((0.5, 0.5 + 1e-7), (1.0, 1.0)))))
        p.validate()
        trade = SimpleNamespace(product=p, engine=PhoenixQuadEngine(QuadParams()), env=env(), trade_id="t")
        with pytest.raises(ValidationError, match="immediate KO settlement"):
            build_phoenix_surface(trade)


class TestBacktestLifeSurface:
    @staticmethod
    def _surface():
        return SimpleNamespace(step_of={0.25: 1, 0.75: 2}, t=np.array([0.0, 0.25, 0.75, 1.0]))

    def test_node_matches_within_zero_only(self):
        from quantark.backtest.simulation.pricing.surface import LifeSurface

        s = self._surface()
        assert LifeSurface.node(s, 0.75) == 2
        assert LifeSurface.node(s, 0.75 + NOISE) == 2
        assert LifeSurface.node(s, 1.0 - NOISE) == 3
        with pytest.raises(ValidationError, match="no surface node"):
            LifeSurface.node(s, 0.75 + GAP)
        with pytest.raises(ValidationError, match="no surface node"):
            LifeSurface.node(s, 1.0 - GAP)

    def test_read_columns_stops_beyond_maturity_by_the_instant_rule(self):
        from quantark.backtest.simulation.pricing.surface import LifeSurface, LifeSurfacePricer

        s = self._surface()
        s.node = lambda elapsed: LifeSurface.node(s, elapsed)
        # a day past maturity ends the read; a later (out-of-order) day shows whether it did
        for past, columns in ((1.0 + NOISE, [0, 1, 2, 3]), (1.0 + GAP, [0, 2, 3])):
            elapsed = [0.0, 0.75, past, 0.25]
            pricer = SimpleNamespace(dates=elapsed, elapsed=lambda d, e=elapsed: e[d])
            assert LifeSurfacePricer._read_columns(pricer, s) == columns

    def test_stress_days_are_knock_in_dates_by_the_instant_rule(self):
        from quantark.backtest.simulation.pricing.surface import LifeSurfacePricer

        elapsed = [0.0, 0.25 + NOISE, 0.75 + GAP, 0.9]
        pricer = SimpleNamespace(gate=SimpleNamespace(barrier_dates=10), dates=elapsed, _maturity=1.0,
                                 elapsed=lambda d: elapsed[d], _ki_observation_times=lambda: [0.25, 0.75, 0.9])
        assert LifeSurfacePricer._stress_days(pricer) == [1, 3]


class TestSettlementSupport:
    @staticmethod
    def _ko_reset_stub(pre_maturity):
        return SimpleNamespace(get_pre_maturity_time=lambda env: pre_maturity, settlement_date=None,
                               first_schedule_settlement_time=None)

    @staticmethod
    def _terminal():
        from quantark.asset.equity.settlement import CashflowKind, ResolvedPaymentTiming

        return ResolvedPaymentTiming(kind=CashflowKind.TERMINAL, determination_date=None, determination_time=1.0,
                                     payment_date=None, payment_time=1.0 + 2.0 / 365.0, determination_df=0.97,
                                     payment_df=0.9698, delay_df=0.9698 / 0.97)

    def test_first_schedule_payment_one_maturity_is_one_instant(self):
        from quantark.asset.equity.engine.settlement_support import first_schedule_payment_time

        terminal = self._terminal()
        assert first_schedule_payment_time(self._ko_reset_stub(1.0 - NOISE), None, terminal) == terminal.payment_time
        two = first_schedule_payment_time(self._ko_reset_stub(1.0 - 1e-7), None, terminal)
        assert two == (1.0 - 1e-7) + (terminal.payment_time - 1.0)

    def test_first_schedule_settlement_one_maturity_is_one_instant(self):
        from quantark.asset.equity.engine.settlement_support import first_schedule_settlement

        terminal = self._terminal()
        stub = self._ko_reset_stub(1.0 - 1e-7)
        stub.first_schedule_settlement_time = 1.5
        # schedule 1 ends beyond Tolerance.ZERO before the product: its explicit payment governs
        from unittest import mock

        timing = SimpleNamespace(payment_time=1.5, payment_date=None, delay_df=0.99)
        with mock.patch("quantark.asset.equity.engine.settlement_support.first_schedule_terminal_timing",
                        return_value=timing):
            assert first_schedule_settlement(stub, None, terminal).settlement_date_governs
            stub.get_pre_maturity_time = lambda env: 1.0 - NOISE
            assert not first_schedule_settlement(stub, None, terminal).settlement_date_governs

    def test_american_exercise_grid_ends_on_maturity_by_the_instant_rule(self):
        from datetime import datetime

        from quantark.asset.equity.engine.settlement_support import build_american_exercise_date_grid
        from test_asset.equity.engine_mc.test_contract_clock_mc import env

        e = env()
        exercise = datetime(2027, 9, 11)
        last = 365.0 / 365.0
        for maturity, ok in ((last + NOISE, True), (last + GAP, False)):
            product = SimpleNamespace(exercise_date=exercise, get_maturity=lambda env, m=maturity: m)
            if ok:
                _, times = build_american_exercise_date_grid(product, e)
                assert times[-1] == maturity
            else:
                with pytest.raises(ValidationError, match="inconsistent with product maturity"):
                    build_american_exercise_date_grid(product, e)


# ======================================================================================================================
# 2. AC9: the site list
# ======================================================================================================================

QUAD = "quantark/asset/equity/engine/quad/"
PDE = "quantark/asset/equity/engine/pde/"
MC = "quantark/asset/equity/engine/mc/"
V1_ABS = ("abs_tol=Tolerance.PRECISION",)          # QUAD V1 / PDE: the retired 1e-6 (~31 s) absolute width
VOL_TAU = ("is_close(float(tau), float(T))",)      # the vol-model solvers' relative at-valuation test

#: (file, function qualname, the retired time tolerances that site held; none may come back). A nested function is
#: dotted (``adi_forward_pass.column_of``), a method is ``Class.method``.
SITES = [
    # ---- T5: QUAD V1 (5583399e) -------------------------------------------------------------------------------------
    (QUAD + "snowball_quad_engine.py", "SnowballQuadEngine._merge_times", V1_ABS),
    (QUAD + "snowball_quad_engine.py", "SnowballQuadEngine._insert_settlement_times", ("Tolerance.PRECISION",)),
    (QUAD + "snowball_quad_engine.py", "SnowballQuadEngine._match_record", V1_ABS),
    (QUAD + "snowball_quad_engine.py", "SnowballQuadEngine._compute_event_stats", V1_ABS),
    (QUAD + "snowball_quad_engine.py", "SnowballQuadEngine._forward_event_quantities", V1_ABS),
    (QUAD + "snowball_quad_engine.py", "SnowballQuadEngine._fold_in_life", V1_ABS),
    (QUAD + "snowball_quad_engine.py", "SnowballQuadEngine._reachable_ko_observations", ("atol=Tolerance.PRECISION",)),
    (QUAD + "ko_reset_snowball_quad_engine.py", "KOResetSnowballQuadEngine._compute_event_stats", V1_ABS),
    (QUAD + "ko_reset_snowball_quad_engine.py", "KOResetSnowballQuadEngine._price_once", V1_ABS),
    (QUAD + "phoenix_quad_engine.py", "PhoenixQuadEngine._price_once", V1_ABS),
    (QUAD + "v1_statistics.py", "knock_in_table", ("<= Tolerance.PRECISION",)),
    # ---- T6: PDE (0b03738b, f0bc062c) -------------------------------------------------------------------------------
    (PDE + "grid/request.py", "_dedup_sorted_interior", ("is_close(t, tau)", "is_close(out[-1], t)")),
    (PDE + "grid/request.py", "GridRequest.__post_init__", ("is_close(t, e)",)),
    (PDE + "grid/time.py", "TimeLayout.step_at", ("is_close(key, obs_time)",)),
    (PDE + "base_pde_solver.py", "BasePDESolver._setup_observation_indices",
     ("is_close(rec.observation_time, tau)", "is_close(obs_time, tau)")),
    (PDE + "snowball_pde_solver.py", "SnowballPDESolver._populate_observation_maps", ("is_close(obs_time, tau)",)),
    (PDE + "snowball_pde_solver.py", "SnowballPDESolver._aligned_time_index", V1_ABS),
    (PDE + "snowball_pde_solver.py", "SnowballPDESolver._find_record_at_time",
     ("is_close(rec.observation_time, target_time)",)),
    (PDE + "snowball_pde_solver.py", "SnowballPDESolver._knock_in_side_row", ("is_greater_than_or_close(",)),
    (PDE + "snowball_pde_solver.py", "SnowballPDESolver._next_ko_record_at_time", ("is_greater_than_or_close(",)),
    (PDE + "ko_reset_snowball_pde_solver.py", "KOResetSnowballPDESolver._register_post_ko_observations",
     ("is_close(obs_time, tau)",)),
    (PDE + "phoenix_pde_solver.py", "PhoenixPDESolver._register_coupon_observations", ("is_close(obs_time, tau)",)),
    (PDE + "barrier_pde_solver.py", "BarrierPDESolver._observation_timeline", ("is_close(t, self._total_tau)",)),
    (PDE + "barrier_vol_pde_solvers.py", "_settlement_scale",
     ("np.isclose(rec.settlement_time, rec.observation_time)",)),
    (PDE + "core/adapters/snowball.py", "SnowballAdapter._event_schedule", ("is_close(t_obs, tau)",)),
    (PDE + "core/adapters/ko_reset.py", "KOResetAdapter.terminal", V1_ABS),
    (PDE + "core/statistics/adjoint.py", "_column_of", ("is_close(t, tau)",)),
    (PDE + "core/statistics/terminal.py", "_position", ("is_close(float(rec.observation_time), t)",)),
    (PDE + "core/statistics/snowball.py", "SnowballStatistics.column_of_record",
     ("is_close(float(rec.observation_time), t)",)),
    (PDE + "core/statistics/ko_reset.py", "KOResetStatistics.column_of_record",
     ("is_close(float(rec.observation_time), t)",)),
    (PDE + "core/statistics/ko_reset.py", "KOResetStatistics._maturity_timing",
     ("is_close(float(last_pre.observation_time), pre_maturity)",)),
    (PDE + "core/statistics/ko_reset.py", "KOResetStatistics._probability_rows", V1_ABS),
    (PDE + "core/statistics/ko_reset.py", "KOResetStatistics.terminal", V1_ABS),
    (PDE + "core/statistics/phoenix.py", "PhoenixStatistics.__init__", ("is_close(float(own.observation_time)",)),
    (PDE + "dcn_statistics.py", "DCNForwardResult.ki_row_at", ("is_close(float(self.rows.ki_times[k]), t)",)),
    (PDE + "vol_model_adjoint.py", "adi_forward_pass.column_of", ("is_close(x, tau)",)),
    (PDE + "snowball_vol_pde_solvers.py", "_Heston2DSnowballPDEBase._apply_ko", VOL_TAU),
    (PDE + "snowball_vol_pde_solvers.py", "_Heston2DSnowballPDEBase._capture_t0_pre_event_surface", VOL_TAU),
    (PDE + "snowball_vol_pde_solvers.py", "_Heston2DSnowballPDEBase._v0_hook", VOL_TAU),
    (PDE + "phoenix_vol_pde_solvers.py", "_Heston2DPhoenixPDEBase._apply_coupon", VOL_TAU),
    (PDE + "phoenix_vol_pde_solvers.py", "_Heston2DPhoenixPDEBase._apply_ko", VOL_TAU),
    (PDE + "phoenix_vol_pde_solvers.py", "_Heston2DPhoenixPDEBase._capture_t0_pre_event_surface", VOL_TAU),
    (PDE + "phoenix_vol_pde_solvers.py", "_Heston2DPhoenixPDEBase._hook_coupon_is_joint", VOL_TAU),
    (PDE + "phoenix_vol_pde_solvers.py", "_Heston2DPhoenixPDEBase._v0_hook", VOL_TAU),
    # ---- T7: MC + rest ------------------------------------------------------------------------------------------------
    (MC + "snowball_mc_engine.py", "SnowballMCEngine._build_time_grid_ko_reset", ("is_close(time, pre_maturity)",)),
    (MC + "accumulator_mc_engine.py", "AccumulatorMCEngine._price_mc",
     ("is_close(float(obs_times[-1]), maturity)", "np.searchsorted(all_times, obs_times)")),
    (MC + "barrier_option_mc_engine.py", "BarrierOptionMCEngine._build_discrete_grid",
     ("is_close(sorted_obs[-1], T)", "np.searchsorted(all_times, obs_times)")),
    (MC + "single_sharkfin_option_mc_engine.py", "SingleSharkfinOptionMCEngine._build_discrete_grid",
     ("is_close(sorted_obs[-1], maturity)", "np.searchsorted(all_times, obs_times)")),
    (MC + "double_sharkfin_option_mc_engine.py", "DoubleSharkfinOptionMCEngine._build_discrete_grid",
     ("is_close(sorted_obs[-1], maturity)", "np.searchsorted(all_times, obs_times)")),
    ("quantark/cashleg/autocallable_leg.py", "AutocallableCashLeg._per_event_contingent",
     ("<= Tolerance.PROBABILITY",)),
    ("quantark/cashleg/autocallable_leg.py", "AutocallableCashLeg._scheduled_contingent",
     ("tol=Tolerance.PROBABILITY",)),
    ("quantark/asset/equity/riskmeasures/greeks/theta_shift.py", "_carried_to_t1",
     ("is_close(payment_time, time_bump)",)),
    ("quantark/asset/equity/riskmeasures/greeks/theta_shift.py", "_ko_reset_schedule_1_end",
     ("is_close(t, time_bump)",)),
    ("quantark/sacva/exposure/snowball_surface.py", "_grid_index", ("<= _TOL",)),
    ("quantark/sacva/exposure/snowball_surface.py", "_validate_ko_leg", ("_SETTLE_TOL",)),
    ("quantark/sacva/exposure/snowball_surface.py", "_index_ko_leg", ("_SETTLE_TOL",)),
    ("quantark/sacva/exposure/snowball_surface.py", "build_snowball_surface", ("_SETTLE_TOL",)),
    ("quantark/sacva/exposure/phoenix_surface.py", "_grid_index", ("<= _TOL",)),
    ("quantark/sacva/exposure/phoenix_surface.py", "build_phoenix_surface", ("> Tolerance.PRECISION",)),
    ("quantark/backtest/simulation/pricing/surface.py", "LifeSurface.node",
     ("is_close(key, elapsed)", "is_close(elapsed, tau)")),
    ("quantark/backtest/simulation/pricing/surface.py", "LifeSurfacePricer._read_columns",
     ("is_close(elapsed, tau)",)),
    ("quantark/backtest/simulation/pricing/surface.py", "LifeSurfacePricer._stress_days",
     ("is_close(self.elapsed(d), t)",)),
    ("quantark/backtest/simulation/pricing/surface.py", "LifeSurfacePricer._ki_barrier_on", ("is_close(elapsed, t)",)),
    ("quantark/asset/equity/engine/settlement_support.py", "build_american_exercise_date_grid",
     ("is_close(resolved_times[-1], maturity)",)),
    ("quantark/asset/equity/engine/settlement_support.py", "first_schedule_settlement", V1_ABS),
    ("quantark/asset/equity/engine/settlement_support.py", "first_schedule_payment_time", V1_ABS),
    # final-review findings: the KO-reset product's own time-identity checks (is_close keeps its default rel 1e-9)
    ("quantark/asset/equity/product/option/ko_reset_snowball_option.py",
     "KnockOutResetSnowballOption._validate_first_schedule_settlement_range",
     ("abs_tol=Tolerance.PRECISION", "is_close(value, pre_end", "is_close(value, float(terminal.payment_time)")),
    ("quantark/asset/equity/product/option/ko_reset_snowball_option.py",
     "KnockOutResetSnowballOption._validate_schedule_two_ends_last", ("is_close(end, pre_end",)),
    ("quantark/asset/equity/engine/quad/v2/context.py", "_shortest_interval", ()),
    # CP2 findings
    ("quantark/asset/equity/engine/mc/barrier_vol_mc_engines.py", "_settlement_scale",
     ("np.isclose(rec.settlement_time, rec.observation_time)",)),
    ("quantark/cashleg/event_distribution.py", "EventDistribution._survival_axis", ("tol=Tolerance.PROBABILITY",)),
    ("quantark/cashleg/event_distribution.py", "EventDistribution.survival_at", ("tol=Tolerance.PROBABILITY",)),
    ("quantark/asset/equity/engine/quad/ko_reset_snowball_quad_engine.py", "KOResetSnowballQuadEngine._reset_rows",
     ("is_close(float(last_pre.observation_time), pre_maturity)",)),
    ("quantark/cashleg/folding.py", "_ko_reset_structure", ("- Tolerance.PROBABILITY",)),
    ("quantark/sacva/exposure/value_surface.py", "GridValueSurface.__init__", ("<= 1e-9",)),
    ("quantark/sacva/exposure/value_surface.py", "GridValueSurface.value_at", ("<= 1e-9",)),
    ("quantark/asset/equity/engine/pde/barrier_pde_solver.py", "BarrierPDESolver._next_observation",
     ("is_greater_than_or_close(t, current_time)",)),
]

#: D2: the MC grids that union contract instants with engine-added nodes merge against the contract instants
ANCHORED = [
    (MC + "snowball_mc_engine.py", "SnowballMCEngine._build_time_grid"),
    (MC + "snowball_mc_engine.py", "SnowballMCEngine._build_time_grid_ko_reset"),
    (MC + "phoenix_mc_engine.py", "PhoenixMCEngine._build_time_grid"),
]


def function_source(source: str, qualname: str) -> str:
    """The source of the function ``qualname`` (``Class.method``, ``outer.inner``) in a module's ``source``."""
    node: ast.AST = ast.parse(source)
    for part in qualname.split("."):
        children = [child for child in ast.iter_child_nodes(node)
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == part]
        if len(children) != 1:
            raise LookupError(f"{qualname}: {len(children)} definitions named {part!r}")
        node = children[0]
    return ast.get_source_segment(source, node)


def retired_patterns_in(source: str, qualname: str, retired) -> list:
    """The retired patterns the function ``qualname`` of ``source`` still holds."""
    body = function_source(source, qualname)
    return [pattern for pattern in retired if pattern in body]


@pytest.mark.parametrize("path,qualname,retired", SITES, ids=[f"{p.rsplit('/', 1)[-1]}::{q}" for p, q, _ in SITES])
def test_site_holds_no_retired_time_tolerance(path, qualname, retired):
    assert retired_patterns_in((ROOT / path).read_text(), qualname, retired) == []


@pytest.mark.parametrize("path,qualname", ANCHORED, ids=[q for _, q in ANCHORED])
def test_engine_added_nodes_are_merged_against_contract_anchors(path, qualname):
    assert "anchors=anchors" in function_source((ROOT / path).read_text(), qualname)


def test_function_source_reaches_nested_functions_and_refuses_a_missing_one():
    source = "class A:\n    def f(self):\n        def g():\n            return 1\n        return g\n"
    assert function_source(source, "A.f.g").strip().startswith("def g")
    with pytest.raises(LookupError):
        function_source(source, "A.h")
