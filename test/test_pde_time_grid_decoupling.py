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
