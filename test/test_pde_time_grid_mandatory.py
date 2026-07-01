"""Tests for TimeGrid.build_mandatory — mandatory nodes + resolution fill.

The build_mandatory builder decouples the three orthogonal time-grid concerns:
  * alignment   — nodes that MUST land exactly (KO/coupon, daily-KI monitor)
  * resolution  — fill density between nodes (steps_per_day), KI-independent
  * cap policy  — max_steps_total caps FILL only; mandatory nodes are inviolable

See spec 2026-07-01-pde-autocallable-event-distribution-redesign-design.md, and
plan Task 1.1 ([§11.5] mandatory nodes inviolable).
"""

import numpy as np

from quantark.asset.equity.engine.pde.time_grid import TimeGrid
from quantark.util.numerical import is_close


def _contains(t_vec, t):
    return bool(np.any(np.abs(t_vec - t) < 1e-9))


def test_mandatory_nodes_land_exactly():
    tau = 1.0
    mand = [0.25, 0.5, 0.75]
    t_vec, dt_vec = TimeGrid.build_mandatory(tau, mand, steps_per_day=1.0, day_count=252)
    for t in mand + [0.0, tau]:
        assert _contains(t_vec, t), f"{t} missing from grid"
    assert is_close(t_vec[0], 0.0) and is_close(t_vec[-1], tau)
    assert np.all(np.diff(t_vec) > 0)  # strictly increasing
    assert len(dt_vec) == len(t_vec) - 1


def test_daily_nodes_no_inflation():
    # ~252 daily mandatory nodes at 1 step/day => ~252 steps, NOT ~2520
    tau = 1.0
    daily = list(np.linspace(1 / 252, 251 / 252, 251))
    t_vec, _ = TimeGrid.build_mandatory(tau, daily, steps_per_day=1.0, day_count=252)
    assert 250 <= len(t_vec) - 1 <= 300


def test_sparse_monthly_gets_resolution_fill():
    # 12 monthly nodes, no daily => ~1 step/day between them => ~252 steps
    tau = 1.0
    monthly = list(np.linspace(1 / 12, 11 / 12, 11))
    t_vec, _ = TimeGrid.build_mandatory(tau, monthly, steps_per_day=1.0, day_count=252)
    assert 230 <= len(t_vec) - 1 <= 300
    for t in monthly:
        assert _contains(t_vec, t)


def test_mandatory_inviolable_when_cap_too_small():
    # 260 mandatory nodes, cap=100: mandatory wins; nodes preserved; cap
    # exceeded (logged), NOT moved.  [§11.5]
    tau = 1.0
    mand = list(np.linspace(0.001, 0.999, 260))
    t_vec, _ = TimeGrid.build_mandatory(
        tau, mand, steps_per_day=1.0, day_count=252, max_steps_total=100
    )
    for t in mand:
        assert _contains(t_vec, t)
    assert len(t_vec) - 1 >= 260  # cap exceeded, never dropped a node


def test_duplicate_and_out_of_range_mandatory_ignored():
    t_vec, _ = TimeGrid.build_mandatory(
        1.0, [0.5, 0.5, -0.1, 1.5, 0.0, 1.0], steps_per_day=1.0, day_count=252
    )
    assert _contains(t_vec, 0.5)
    assert np.all((t_vec >= 0.0) & (t_vec <= 1.0))
