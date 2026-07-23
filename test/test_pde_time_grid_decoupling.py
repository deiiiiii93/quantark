"""Tests for the decoupled PDE time-grid seam (_time_grid_spec / _resolve_time_grid).

Covers plan Tasks 1.2–1.4: the base seam default, the auto_grid=False param-drop
fix (root cause 2), market-independence of the grid (§4.5 / gate 6.6), and the
KI-regime-aware split of align vs monitor times.
"""

import numpy as np

from quantark.asset.equity.engine.pde.base_pde_solver import TimeGridSpec
from quantark.asset.equity.engine.pde.european_pde_solver import EuropeanPDESolver


# ---------------------------------------------------------------------------
# Task 1.2 — base seam default
# ---------------------------------------------------------------------------
def test_base_time_grid_spec_defaults(vanilla_barrier_product):
    solver = EuropeanPDESolver()
    spec = solver._time_grid_spec(vanilla_barrier_product, tau=1.0)
    assert isinstance(spec, TimeGridSpec)
    assert spec.monitor_times == []  # base: no KI monitor concept
    assert spec.steps_per_day == solver.params.event_steps_per_day
    # align times = interior discrete observation dates
    assert all(0.0 < t < 1.0 for t in spec.align_times)
    assert len(spec.align_times) == 11  # 11 interior monthly dates


# ---------------------------------------------------------------------------
# Task 1.3 — _resolve_time_grid uses the seam (fixes auto_grid=False param drop)
# ---------------------------------------------------------------------------
def test_auto_grid_false_passes_resolution_params(
    snowball_daily_ki, snowball_pde_solver_factory
):
    # auto_grid=False previously dropped steps_per_day/day_count -> ~10x
    # inflation (root cause 2). Now build_mandatory receives them.
    solver = snowball_pde_solver_factory(
        auto_grid=False, event_steps_per_day=1, time_grid_type="event_aligned"
    )
    t_vec, _ = solver._resolve_time_grid(
        snowball_daily_ki, tau=1.0, barriers=solver._get_barriers(snowball_daily_ki)
    )
    # ~1 step/day, not ~10/day
    assert (len(t_vec) - 1) < 400


def test_time_grid_market_independent(snowball_daily_ki, snowball_pde_solver_factory):
    # spot/vol/rate bumps must not change the time grid [§4.5 / gate 6.6]
    solver = snowball_pde_solver_factory(auto_grid=True, event_steps_per_day=1)
    barriers = solver._get_barriers(snowball_daily_ki)
    t0, _ = solver._resolve_time_grid(snowball_daily_ki, 1.0, barriers)
    t1, _ = solver._resolve_time_grid(snowball_daily_ki, 1.0, barriers)
    assert np.array_equal(t0, t1)


# ---------------------------------------------------------------------------
# Task 1.4 — KI-regime-aware _ki_monitor_times + autocallable _time_grid_spec
# ---------------------------------------------------------------------------
def test_ki_monitor_daily_discrete(snowball_daily_ki, snowball_solver):
    times = snowball_solver._ki_monitor_times(snowball_daily_ki, tau=1.0)
    assert len(times) > 100  # ~daily interior dates


def test_ki_monitor_european_is_empty(snowball_euro_ki, snowball_solver):
    assert snowball_solver._ki_monitor_times(snowball_euro_ki, tau=1.0) == []


def test_ki_monitor_continuous_is_empty(snowball_cont_ki, snowball_solver):
    assert snowball_solver._ki_monitor_times(snowball_cont_ki, tau=1.0) == []


def test_ki_monitor_no_ki_is_empty(snowball_no_ki, snowball_solver):
    assert snowball_solver._ki_monitor_times(snowball_no_ki, tau=1.0) == []


def test_align_times_are_ko_only(snowball_daily_ki, snowball_solver):
    align = snowball_solver._ko_coupon_align_times(snowball_daily_ki, tau=1.0)
    assert len(align) < 40  # monthly-ish KO, not daily
    assert all(0.0 < t < 1.0 for t in align)


def test_time_grid_spec_splits_align_and_monitor(snowball_daily_ki, snowball_solver):
    spec = snowball_solver._time_grid_spec(snowball_daily_ki, tau=1.0)
    assert len(spec.align_times) < 40  # KO only
    assert len(spec.monitor_times) > 100  # daily KI monitor
    assert spec.steps_per_day == snowball_solver.params.event_steps_per_day


def test_get_event_times_preserves_union(snowball_daily_ki, snowball_solver):
    # Back-compat: _get_event_times still returns sorted(align ∪ monitor) for
    # Rannacher/grid-cache consumers.
    union = snowball_solver._get_event_times(snowball_daily_ki, tau=1.0)
    align = set(snowball_solver._ko_coupon_align_times(snowball_daily_ki, 1.0))
    monitor = set(snowball_solver._ki_monitor_times(snowball_daily_ki, 1.0))
    assert set(union) == align | monitor
    assert union == sorted(union)


# ---------------------------------------------------------------------------
# Task 1.5 — KO-reset pre+post-KI KO alignment [§11.7]
# ---------------------------------------------------------------------------
def test_ko_reset_aligns_pre_and_post_ki_ko(ko_reset_product, ko_reset_solver):
    tau = 2.0  # total maturity (maturity_post)
    align = ko_reset_solver._ko_coupon_align_times(ko_reset_product, tau=tau)
    post_ki_ko_times = ko_reset_solver._post_ki_ko_times(ko_reset_product, tau=tau)
    assert len(post_ki_ko_times) > 0  # guard against a vacuous test
    for t in post_ki_ko_times:
        assert any(abs(t - a) < 1e-9 for a in align), f"post-KI KO {t} not aligned"


def test_ko_reset_reconciliation_gate(ko_reset_product, pricing_env):
    """[Self-review #6b] KO-reset event-stats delegate to QUAD with a residual
    adjustment against the PDE price. Phase 1 changed the PDE grid, so confirm:

    * the decoupled grid still yields a grid-converged PDE price (agrees with
      the pre-change baseline of ~98.028 measured on `main`), and
    * ``calculate_event_stats`` still reconciles its reported ``pv`` to the PDE
      price exactly (so the residual/distribution stay consistent).
    """
    from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
        KOResetSnowballPDESolver,
    )
    from quantark.asset.equity.param import PDEParams

    # Pinned to the legacy event discretization: the 98.028 baseline was
    # captured on main BEFORE the event-projection default flip (2026-07-23),
    # and this gate certifies the Phase-1 TIME-grid decoupling at fixed event
    # semantics. (Under the corrected default the converged value reprices to
    # ~97.976 — covered by test_pde_event_projection.py.)
    legacy = dict(event_projection="nodal", event_rannacher_steps=1)
    coarse = KOResetSnowballPDESolver(PDEParams(grid_size=120, time_steps=60, **legacy))
    fine = KOResetSnowballPDESolver(
        PDEParams(grid_size=240, time_steps=120, event_steps_per_day=8, **legacy)
    )
    pv_coarse = float(coarse.price(ko_reset_product, pricing_env))
    pv_fine = float(fine.price(ko_reset_product, pricing_env))

    # Grid-converged on the decoupled grid.
    assert abs(pv_coarse - pv_fine) / abs(pv_fine) < 1e-3
    # Agrees with the pre-change baseline captured on main (98.0279).
    assert abs(pv_fine - 98.028) < 0.05

    # Reconciliation intact: reported pv == PDE price.
    stats = coarse.calculate_event_stats(ko_reset_product, pricing_env)
    assert abs(float(stats.pv) - pv_coarse) < 1e-6

    # The reconciliation invariant must also hold under the corrected default
    # event semantics (projection + 2 damping steps).
    coarse_default = KOResetSnowballPDESolver(PDEParams(grid_size=120, time_steps=60))
    pv_default = float(coarse_default.price(ko_reset_product, pricing_env))
    stats_default = coarse_default.calculate_event_stats(ko_reset_product, pricing_env)
    assert abs(float(stats_default.pv) - pv_default) < 1e-6
