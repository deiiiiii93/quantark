"""Regression tests for P&L percentage sign convention."""

from dynamicscenario.base import BaseScenarioResults
from dynamicscenario.results.dynamic_results import DynamicScenarioResults
from util.numerical import pnl_pct_of_abs_baseline


def test_pnl_pct_positive_baseline_unchanged():
    assert pnl_pct_of_abs_baseline(25.0, 100.0) == 25.0
    assert pnl_pct_of_abs_baseline(-25.0, 100.0) == -25.0


def test_pnl_pct_negative_baseline_uses_absolute_exposure():
    assert pnl_pct_of_abs_baseline(25.0, -100.0) == 25.0
    assert pnl_pct_of_abs_baseline(-25.0, -100.0) == -25.0


def test_pnl_pct_zero_baseline_returns_zero():
    assert pnl_pct_of_abs_baseline(25.0, 0.0) == 0.0


def test_dynamic_results_derives_pct_from_absolute_baseline():
    results = DynamicScenarioResults(
        path_name="Negative Baseline",
        baseline_value=-100.0,
        final_value=-75.0,
        day_results=[],
        total_pnl=25.0,
    )

    assert results.total_pnl == 25.0
    assert results.total_pnl_pct == 25.0


def test_base_scenario_results_derives_pct_from_absolute_baseline():
    results = BaseScenarioResults(
        path_name="Negative Baseline",
        baseline_value=-100.0,
        final_value=-125.0,
        day_results=[],
        total_pnl=-25.0,
    )

    assert results.total_pnl == -25.0
    assert results.total_pnl_pct == -25.0
