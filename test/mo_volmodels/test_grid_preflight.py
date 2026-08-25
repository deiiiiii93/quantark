"""Tests for example/mo_volmodels/11c_grid_preflight.py (Gate G5).

The sweep itself needs the uncommitted surface history, so the tests that
touch it skip without one. Everything about the ARTIFACT CONTRACT -- the
shape the dashboard's headline_g5 reads -- is checked from synthetic data and
runs everywhere.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "example/mo_volmodels/11c_grid_preflight.py"
HISTORY_DIR = ROOT / "example/mo_volmodels/data/history"

spec = importlib.util.spec_from_file_location("grid_preflight_11c", MODULE_PATH)
g5 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g5
spec.loader.exec_module(g5)


def test_artifact_carries_the_two_keys_the_dashboard_reads():
    """headline_g5 requires a positive n_operating_points and a list."""
    doc = g5.build_artifact(points=[], failures=[], scope={}, config={})
    assert doc["n_operating_points"] == 0
    assert doc["under_resolved"] == []


def test_a_failing_point_is_recorded_not_swallowed():
    failures = [
        {"variant": "flat_bsm", "inception": "2023-05-04", "date": "2024-01-02",
         "tau_years": 2.34, "error": "ValidationError: spatial grid ..."}
    ]
    doc = g5.build_artifact(points=["a", "b"], failures=failures, scope={}, config={})
    assert doc["n_operating_points"] == 2
    assert len(doc["under_resolved"]) == 1
    assert doc["under_resolved"][0]["error"].startswith("ValidationError")


def test_scope_is_recorded_so_the_artifact_cannot_overclaim():
    """A pre-flight that covers less than everything must say so."""
    doc = g5.build_artifact(points=[], failures=[], scope={"not_covered": ["x"]},
                            config={})
    assert doc["scope"]["not_covered"] == ["x"]


@pytest.mark.skipif(
    not (HISTORY_DIR / "surface_manifest.json").exists(),
    reason="IV surface history not built",
)
def test_one_real_operating_point_builds_its_grid():
    """The seam works against a real inception: build only, no solve."""
    error = g5.probe_one_inception(
        inception="2023-05-04", limit_days=3, history_dir=HISTORY_DIR
    )
    assert error == [], f"grid build failed on a live operating point: {error}"
