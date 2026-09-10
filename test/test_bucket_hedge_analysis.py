"""Acceptance rules that must refuse a misleading success.

Every test here feeds a synthetic panel with a known answer, so the rules are
checked against arithmetic rather than against a run.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "example" / "snowball_q_term_structure"


def _load(name: str, filename: str):
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = _load("_bucket_analysis", "_bucket_analysis.py")


# ---------------------------------------------------------------------------
# Ratios and denominators
# ---------------------------------------------------------------------------


def test_a_ratio_compares_like_with_like():
    result = A.rms_ratio([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], budget=1e-6)
    assert result.numerator == pytest.approx(1.0)
    assert result.denominator == pytest.approx(2.0)
    assert result.ratio == pytest.approx(0.5)
    assert result.coverage == 3
    assert result.within_absolute_budget is None
    assert result.improved is True


def test_a_numerically_zero_denominator_becomes_an_absolute_check():
    """Dividing by it would manufacture infinity or an accidental pass."""
    result = A.rms_ratio([1e-9, 1e-9], [0.0, 0.0], budget=1e-6)
    assert math.isnan(result.ratio)
    assert result.within_absolute_budget is True
    assert result.improved is True
    # The same zero denominator with a REAL net exposure fails the budget.
    big = A.rms_ratio([5.0, 5.0], [0.0, 0.0], budget=1e-6)
    assert math.isnan(big.ratio)
    assert big.within_absolute_budget is False
    assert big.improved is False
    assert not math.isinf(big.ratio)


def test_no_matched_observations_reports_no_coverage():
    result = A.rms_ratio([float("nan")], [float("nan")], budget=1e-6)
    assert result.coverage == 0
    assert result.improved is None


def test_gross_measures_compare_against_gross_denominators():
    # Two nodes of +50 and -50: the parallel sum is zero, the gross is 100.
    parallel = A.rms_ratio([0.0, 0.0], [0.0, 0.0], budget=1e-6)
    gross = A.rms_ratio([100.0, 100.0], [50.0, 50.0], budget=1e-6)
    assert parallel.within_absolute_budget is True
    assert gross.ratio == pytest.approx(2.0)
    assert gross.improved is False


def test_maxima_and_date_level_breaches_are_reported():
    breach = A.exceedances([0.001, 0.02, -0.05], 0.01)
    assert breach["dates"] == 3
    assert breach["exceedances"] == 2
    assert breach["max_abs"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# The parallel column alone proves nothing
# ---------------------------------------------------------------------------


def test_zero_parallel_rhoq_with_large_opposite_nodes_does_not_pass_broader():
    spot = A.rms_ratio([0.0, 0.0], [10.0, 10.0], budget=1e-6)
    parallel = A.rms_ratio([0.0, 0.0], [5.0, 5.0], budget=1e-6)
    joint = A.joint_mitigation(spot, parallel)
    assert joint.status == "supported"
    # ... and the gross nodal exposure DOUBLED, so the broader claim fails.
    gross = A.rms_ratio([100.0, 100.0], [50.0, 50.0], budget=1e-6)
    broader = A.broader_carry_mitigation(joint, gross, scenario_losses={})
    assert broader.status == "unsupported"
    assert any("gross nodal" in r for r in broader.reasons)


def test_a_worse_unquoted_scenario_blocks_the_broader_claim():
    spot = A.rms_ratio([0.0], [10.0], budget=1e-6)
    parallel = A.rms_ratio([0.0], [5.0], budget=1e-6)
    joint = A.joint_mitigation(spot, parallel)
    gross = A.rms_ratio([1.0], [5.0], budget=1e-6)
    ok = A.broader_carry_mitigation(
        joint,
        gross,
        scenario_losses={"tail+0.01": {"candidate_max_loss": 4.0, "control_max_loss": 5.0}},
    )
    assert ok.status == "supported"
    worse = A.broader_carry_mitigation(
        joint,
        gross,
        scenario_losses={"tail+0.01": {"candidate_max_loss": 9.0, "control_max_loss": 5.0}},
    )
    assert worse.status == "unsupported"
    unmeasured = A.broader_carry_mitigation(
        joint,
        gross,
        scenario_losses={
            "tail+0.01": {"candidate_max_loss": float("nan"), "control_max_loss": 5.0}
        },
    )
    assert unmeasured.status == "inconclusive"


# ---------------------------------------------------------------------------
# Numerical validity
# ---------------------------------------------------------------------------


def test_a_failed_audit_blocks_validity_however_quiet_the_pnl():
    coverage = {
        "audit_coverage": "measured",
        "dates": 10,
        "measured": 10,
        "by_status": {"pass": 9, "fail": 1, "not_measured": 0, "inconclusive": 0},
    }
    result = A.numerical_validity(coverage, requested_mode="daily")
    assert result.status == "unsupported"
    assert any("failed audit" in r for r in result.reasons)


def test_an_unmeasured_run_is_not_valid():
    coverage = {
        "audit_coverage": "not_measured",
        "dates": 10,
        "measured": 0,
        "by_status": {"pass": 0, "fail": 0, "not_measured": 10, "inconclusive": 0},
    }
    assert A.numerical_validity(coverage, requested_mode="daily").status == "unsupported"


def test_a_sampled_run_cannot_claim_daily_coverage():
    coverage = {
        "audit_coverage": "measured",
        "dates": 10,
        "measured": 3,
        "by_status": {"pass": 3, "fail": 0, "not_measured": 7, "inconclusive": 0},
    }
    daily = A.numerical_validity(coverage, requested_mode="daily")
    assert daily.status == "unsupported"
    assert any("daily coverage" in r for r in daily.reasons)
    # The SAME run is valid when only sampled coverage was requested.
    sampled = A.numerical_validity(coverage, requested_mode="sampled")
    assert sampled.status == "supported"


def test_an_inconclusive_audit_is_inconclusive_not_a_failure():
    coverage = {
        "audit_coverage": "measured",
        "dates": 4,
        "measured": 4,
        "by_status": {"pass": 3, "fail": 0, "not_measured": 0, "inconclusive": 1},
    }
    assert A.numerical_validity(coverage, requested_mode="daily").status == "inconclusive"


def test_a_legacy_run_reports_no_audit_rather_than_a_failure():
    result = A.numerical_validity({"audit_coverage": "not_available"}, requested_mode="none")
    assert result.status == "not_available"


# ---------------------------------------------------------------------------
# Actual versus ideal
# ---------------------------------------------------------------------------


def test_execution_error_is_visible_next_to_the_ideal_target():
    result = A.objective_achieved(
        delta_residual_hands=[0.4, -0.3],
        parallel_residual_bp=[0.001, 0.002],
        delta_budget=0.01,
        rhoq_budget=0.01,
        ideal_delta_hands=[0.0, 0.0],
        ideal_parallel_bp=[0.0, 0.0],
    )
    assert result.status == "unsupported"
    assert result.evidence["ideal"]["delta"]["exceedances"] == 0
    assert result.evidence["actual_delta"]["exceedances"] == 2
    assert result.evidence["actual_delta"]["max_abs"] == pytest.approx(0.4)


def test_meeting_both_budgets_is_supported():
    result = A.objective_achieved(
        delta_residual_hands=[0.001, -0.002],
        parallel_residual_bp=[0.001, 0.0],
        delta_budget=0.01,
        rhoq_budget=0.01,
    )
    assert result.status == "supported"


# ---------------------------------------------------------------------------
# Paired resampling
# ---------------------------------------------------------------------------


def test_identical_panels_give_exactly_zero_and_a_zero_width_interval():
    values = np.arange(60.0).reshape(30, 2)
    panel = A.PairedPanel(
        dates=tuple(range(30)),
        inceptions=(0, 1),
        candidate=values,
        control=values.copy(),
        valid=np.ones_like(values, dtype=bool),
    )
    report = A.paired_report(panel, A.matched_rms, block_length=10, replicates=50)
    assert report["point"] == 0.0
    assert report["interval"] == [0.0, 0.0]
    assert report["excludes_zero"] is False


def test_every_inception_column_shares_the_same_sampled_blocks():
    rng = np.random.default_rng(1)
    idx = A.moving_block_indices(30, 10, rng)
    assert idx.shape == (30,)
    # A block is contiguous, so consecutive differences inside one are +1.
    assert all(idx[i + 1] - idx[i] == 1 for i in range(0, 9))

    # Applying one index vector to a whole panel keeps the columns aligned.
    panel = np.arange(60.0).reshape(30, 2)
    resampled = panel[idx]
    assert np.all(resampled[:, 1] - resampled[:, 0] == 1.0)


def test_repeated_sampled_dates_are_not_deduplicated():
    rng = np.random.default_rng(0)
    idx = A.moving_block_indices(10, 10, rng)
    assert len(idx) == 10
    forced = np.concatenate([idx, idx])
    assert len(forced) == 20
    assert len(set(forced.tolist())) < len(forced)


def test_a_block_longer_than_the_window_reports_that_it_cannot_estimate():
    values = np.arange(10.0).reshape(5, 2)
    panel = A.PairedPanel(
        dates=tuple(range(5)),
        inceptions=(0, 1),
        candidate=values,
        control=values.copy(),
        valid=np.ones_like(values, dtype=bool),
    )
    report = A.paired_report(panel, A.matched_rms, block_length=20, replicates=10)
    assert report["interval"] is None
    assert "shorter than the requested" in report["interval_status"]
    # It never silently shortens the block and claims an answer.
    assert report["block_length"] == 20


def test_the_interval_settings_and_matched_counts_are_persisted():
    values = np.random.default_rng(3).normal(size=(40, 3))
    panel = A.PairedPanel(
        dates=tuple(range(40)),
        inceptions=(0, 1, 2),
        candidate=values,
        control=values + 0.5,
        valid=np.ones_like(values, dtype=bool),
    )
    report = A.paired_report(panel, A.matched_rms, block_length=20, replicates=100)
    assert report["matched_observations"] == 120
    assert report["dates"] == 40
    assert report["inceptions"] == 3
    assert report["seed"] == A.DEFAULT_SEED
    assert report["quantiles"] == [0.025, 0.975]
    assert "calendar day" in report["resampling_unit"]
    assert set(report["block_sensitivity"]) <= {"10", "40"}


def test_the_panel_only_keeps_observations_both_sides_have():
    candidate = {"a": {1: 1.0, 2: 2.0, 3: 3.0}, "b": {1: 1.0}}
    control = {"a": {1: 1.0, 2: 2.0}, "b": {1: 2.0}}
    panel = A.build_panel(candidate, control)
    assert panel.inceptions == ("a", "b")
    # Date 3 is candidate-only everywhere and drops out; date 2 stays because
    # inception "a" has it on both sides.
    assert panel.dates == (1, 2)
    # Three matched CELLS: (1,a), (2,a) and (1,b).  Inception "b" simply has
    # no observation on date 2, which is a hole in the grid, not a dropped row.
    assert panel.matched == 3
    assert math.isnan(panel.candidate[1, 1])


def test_an_invalid_run_is_masked_not_quietly_dropped():
    candidate = {"a": {1: 1.0, 2: 5.0}}
    control = {"a": {1: 1.0, 2: 1.0}}
    valid = {"a": {1: True, 2: False}}
    panel = A.build_panel(candidate, control, valid_rows=valid)
    assert panel.dates == (1, 2)
    assert panel.matched == 1
    masked_candidate, _ = panel.masked()
    assert math.isnan(masked_candidate[1, 0])
    # The masked observation is absent from the statistic, and the DATE row
    # is still there, so the block structure is unchanged.
    assert masked_candidate.shape == (2, 1)


# ---------------------------------------------------------------------------
# Economic comparison
# ---------------------------------------------------------------------------


def test_lower_risk_with_higher_cost_can_fail_economics_without_failing_math():
    reports = {
        "daily_pnl_sd": {"interval": [-2.0, -1.0], "point": -1.5},
        "net_cost": {"interval": [0.5, 1.5], "point": 1.0},
    }
    result = A.economic_comparison(reports)
    assert result.status == "unsupported"
    assert any("net_cost" in r for r in result.reasons)
    # The implementation was never asked about; the two claims are separate.
    validity = A.numerical_validity(
        {
            "audit_coverage": "measured",
            "dates": 5,
            "measured": 5,
            "by_status": {"pass": 5, "fail": 0, "not_measured": 0, "inconclusive": 0},
        },
        requested_mode="daily",
    )
    assert validity.status == "supported"


def test_an_interval_spanning_zero_is_inconclusive():
    result = A.economic_comparison({"daily_pnl_sd": {"interval": [-1.0, 1.0]}})
    assert result.status == "inconclusive"
    assert any("spans zero" in r for r in result.reasons)


def test_an_unestimated_interval_is_inconclusive_not_a_pass():
    result = A.economic_comparison(
        {"daily_pnl_sd": {"interval": None, "interval_status": "window too short"}}
    )
    assert result.status == "inconclusive"


def test_every_improvement_is_supported():
    result = A.economic_comparison(
        {
            "daily_pnl_sd": {"interval": [-2.0, -1.0]},
            "tail_loss": {"interval": [-3.0, -0.5]},
        }
    )
    assert result.status == "supported"


def test_the_five_conclusions_serialise_in_order():
    payload = A.conclusions_payload(
        [
            A.Conclusion("numerical_validity", "supported"),
            A.Conclusion("objective_achieved", "supported"),
            A.Conclusion("joint_mitigation", "unsupported", ("spot did not fall",)),
            A.Conclusion("broader_carry_mitigation", "unsupported"),
            A.Conclusion("economic_comparison", "inconclusive"),
        ]
    )
    assert list(payload) == [
        "numerical_validity",
        "objective_achieved",
        "joint_mitigation",
        "broader_carry_mitigation",
        "economic_comparison",
    ]
    assert payload["joint_mitigation"]["reasons"] == ["spot did not fall"]
    with pytest.raises(ValueError):
        A.Conclusion("x", "probably")


def test_the_statistic_for_variability_is_the_mean_of_inception_sds():
    panel = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
    # Column 0 has sd 2.0; column 1 is constant, sd 0.
    assert A.mean_of_inception_sds(panel) == pytest.approx(1.0)
    assert A.matched_rms(np.array([3.0, 4.0])) == pytest.approx(3.5355339, rel=1e-6)


# ---------------------------------------------------------------------------
# Report integration
# ---------------------------------------------------------------------------

AGG = _load("aggregate_stage", "03_aggregate_and_report.py")


def test_a_run_without_risk_frames_reports_unavailable_not_zeros():
    row = AGG.carry_rows({"run_format": "legacy"}, {"notional": 5e7})
    assert row["audit_coverage"] == "not_available"
    assert row["audits_passed"] is None
    for key, value in row.items():
        if key.endswith("_rms") or key.endswith("_mean") or key.endswith("_total"):
            assert math.isnan(value), key


def test_cells_with_no_risk_frames_are_named_not_dropped():
    rows = [
        {"cell": "flat_from_hedge__front", "audit_coverage": "not_available"},
        {"cell": "flat_from_hedge__front", "audit_coverage": "not_available"},
    ]
    summary = AGG.carry_audit_summary(rows)
    assert summary["by_cell"] == {}
    assert summary["risk_audit_unavailable"] == ["flat_from_hedge__front"]


def test_a_measured_cell_reports_its_ratios_and_conclusions():
    rows = [
        {
            "cell": "term_flat_q__buckets_spot_parallel",
            "audit_coverage": "measured",
            "audits_passed": True,
            "net_spot_1pct_bp_rms": 1.0,
            "product_spot_1pct_bp_rms": 10.0,
            "net_parallel_rhoq_bp_rms": 0.5,
            "product_parallel_rhoq_bp_rms": 5.0,
            "net_gross_rhoq_bp_rms": 8.0,
            "product_gross_rhoq_bp_rms": 4.0,
            "gross_contracts_mean": 3.0,
            "turnover_contracts_total": 12.0,
        }
    ]
    summary = AGG.carry_audit_summary(rows)
    cell = summary["by_cell"]["term_flat_q__buckets_spot_parallel"]
    assert cell["spot_ratio"] == pytest.approx(0.1)
    assert cell["parallel_ratio"] == pytest.approx(0.1)
    # Spot and parallel both fell, so the joint claim holds ...
    assert cell["joint_mitigation"] == "supported"
    # ... but the GROSS nodal exposure doubled, so the broader one does not.
    assert cell["gross_ratio"] == pytest.approx(2.0)
    assert cell["broader_carry_mitigation"] == "unsupported"
    assert summary["risk_audit_unavailable"] == []
