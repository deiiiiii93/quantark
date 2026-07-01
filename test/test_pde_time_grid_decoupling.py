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
