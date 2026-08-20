"""Tests for the MO calibration stability artifact generator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "example/mo_volmodels/15_calibration_stability_report.py"
SPEC = importlib.util.spec_from_file_location("mo_calibration_stability_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def calibration(tag: str, *, rmse: float, kappa: float = 2.0) -> dict:
    heston = {
        "v0": 0.05,
        "kappa": kappa,
        "theta": 0.04,
        "sigma": 0.4,
        "rho": -0.5,
        "overall_rmse_iv": rmse,
        "feller_ratio": 1.0,
        "feller_margin": 1e-5,
        "feller_satisfied": True,
        "nfev": 25,
        "bound_hits": {
            parameter: {"lower": False, "upper": False}
            for parameter in reporter.PARAMETERS
        },
    }
    return {
        "date": tag,
        "status": "ok",
        "elapsed_seconds": 1.5,
        "variants": {
            "localvol": {
                "status": "ok",
                "record": {"lv_min": 0.1, "lv_max": 0.4},
            },
            "heston": {"status": "ok", "record": heston},
            "heston_slv": {
                "status": "ok",
                "record": {
                    "leverage_min": 0.5,
                    "leverage_mean": 1.0,
                    "leverage_max": 1.5,
                    "max_negative_mass": 1e-6,
                    "n_clipped": 0,
                },
            },
        },
    }


def test_report_preserves_exclusions_and_computes_coverage() -> None:
    report = reporter.build_report(
        {
            "records": [
                {"date": "20260102", "status": "ok"},
                {
                    "date": "20260105",
                    "status": "excluded",
                    "reason": "static_arbitrage",
                },
                {"date": "20260106", "status": "ok"},
            ]
        },
        {
            "config": {"heston_max_nfev": 200},
            "records": [
                calibration("20260102", rmse=0.01),
                calibration("20260106", rmse=0.015, kappa=2.1),
            ],
        },
        start="20260102",
        end="20260106",
        source_hashes={"surface_manifest": "a", "calibration_manifest": "b"},
    )
    assert report["coverage"] == {
        "surface_decisions": 3,
        "surface_admitted": 2,
        "surface_excluded": 1,
        "calibration_ok": 2,
        "calibration_failed": 0,
        "calibration_missing": 0,
        "coverage_ratio": 1.0,
    }
    assert report["surface_exclusions"][0]["date"] == "20260105"
    assert report["metrics"]["heston_rmse_iv"]["p95"] == 0.01475
    assert report["overall_assessment"] == "WATCH"
    assert report["domain_assessments"] == {
        "pipeline_reliability": "PASS",
        "heston_fit_quality": "PASS",
        "heston_parameter_stability": "WATCH",
        "slv_numerical_health": "PASS",
    }


def test_missing_admitted_calibration_fails_coverage_gate() -> None:
    report = reporter.build_report(
        {
            "records": [
                {"date": "20260102", "status": "ok"},
                {"date": "20260105", "status": "ok"},
            ]
        },
        {"records": [calibration("20260102", rmse=0.01)]},
        start="20260102",
        end="20260105",
        source_hashes={"surface_manifest": "a", "calibration_manifest": "b"},
    )
    assert report["coverage"]["calibration_missing"] == 1
    coverage_gate = next(
        gate for gate in report["gates"] if gate["name"] == "Calibration coverage"
    )
    assert coverage_gate["status"] == "FAIL"
    assert report["overall_assessment"] == "FAIL"


def test_rendered_html_has_charts_and_evidence_links() -> None:
    report = reporter.build_report(
        {"records": [{"date": "20260102", "status": "ok"}]},
        {
            "config": {
                "heston_max_nfev": 200,
                "slv_eta": 1.0,
                "slv_n_steps": 40,
                "slv_n_x": 161,
                "slv_n_z": 81,
            },
            "records": [calibration("20260102", rmse=0.01)],
        },
        start="20260102",
        end="20260102",
        source_hashes={"surface_manifest": "a", "calibration_manifest": "b"},
    )
    rendered = reporter.render_html(report, "evidence.json", "daily.csv")
    assert "<svg" in rendered
    assert "evidence.json" in rendered
    assert "daily.csv" in rendered
    assert "Hard-Feller Ratio (log10 scale)" in rendered


def test_temporal_report_includes_raw_and_ewma_evidence() -> None:
    record = calibration("20260102", rmse=0.01)
    heston = record["variants"]["heston"]["record"]
    heston["temporal_penalty_cost"] = 0.002
    record["temporal_scheme"] = {
        "name": "daily_v0_structural_ewma",
        "slv_heston_feller_ratio": 1.1,
        "slv_heston_feller_satisfied": True,
        "raw_heston": {**heston, "overall_rmse_iv": 0.012},
    }
    report = reporter.build_report(
        {"records": [{"date": "20260102", "status": "ok"}]},
        {
            "config": {
                "temporal_scheme": {
                    "name": "daily_v0_structural_ewma",
                    "structural_ewma_span": 5,
                    "structural_ewma_alpha": 1.0 / 3.0,
                    "heston_temporal_regularization": 0.1,
                }
            },
            "records": [record],
        },
        start="20260102",
        end="20260102",
        source_hashes={"surface_manifest": "a", "calibration_manifest": "b"},
    )
    assert report["temporal"]["enabled"] is True
    assert report["metrics"]["temporal_heston_raw_rmse_iv"]["median"] == 0.012
    rendered = reporter.render_html(report, "evidence.json", "daily.csv")
    assert "Temporal calibration evidence" in rendered
    assert "EWMA span 5" in rendered
