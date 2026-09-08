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


# ---------------------------------------------------------------------------
# The QUAD arm
#
# flat_bsm_quad is 27 of the fleet's 162 cells and routes to the quadrature
# engine, whose own under-resolution guard is _resolve_grid_points: it raises
# NumericalError when resolving the shortest diffusion step would need more
# than max_adaptive_grid_points.  That is the same class of mid-fleet failure
# G5 exists to find before ~143 CPU-hours are spent, so it belongs in the
# sweep rather than in scope.not_covered.
#
# The probe intercepts the engine's OWN method on the instance.  Rebuilding
# the quadrature prologue outside price() to call it with hand-made arguments
# would validate a setup the fleet never performs -- the same trap the PDE arm
# avoids by taking solver.grid_binder instead of constructing one.
# ---------------------------------------------------------------------------

class _FakeQuadEngine:
    """Stands in for SnowballQuadEngine's grid-resolution contract."""

    def __init__(self, *, used, requested=1001, cap=5001, raises=None):
        self._used, self._raises = used, raises
        class _P:
            grid_points = requested
            max_adaptive_grid_points = cap
        self.params = _P()
        self.priced = False

    def _resolve_grid_points(self, maturity, vol, times):
        if self._raises is not None:
            raise self._raises
        return self._used

    def price(self, product, env):
        # Reached only if the probe failed to stop; the real call here is the
        # stacked FFT recursion the sweep must never pay for.
        self._resolve_grid_points(3.0, 0.2, [0.5, 1.0])
        self.priced = True
        return 1.0


def test_the_quad_probe_stops_before_the_expensive_recursion():
    engine = _FakeQuadEngine(used=1001)
    result = g5.quad_grid_probe(engine, product=None, env=None)
    assert result["error"] is None
    assert result["stopped_at"] == "resolve_grid_points"
    assert engine.priced is False, "the probe must not pay for the solve"


def test_a_quad_point_that_exceeds_the_adaptive_cap_is_recorded():
    from quantark.util.exceptions import NumericalError
    engine = _FakeQuadEngine(
        used=None,
        raises=NumericalError(
            "Observation intervals require at least 7001 quadrature grid "
            "points ... exceeding max_adaptive_grid_points=5001"
        ),
    )
    result = g5.quad_grid_probe(engine, product=None, env=None)
    assert result["error"] is not None
    assert result["error"].startswith("NumericalError")
    assert "5001" in result["error"]


def test_an_adapted_grid_reports_the_headroom_it_left():
    """used > requested means adaptation fired, and then used IS required.

    Headroom is derived from the return value, never by recomputing the
    engine's formula -- a second copy of that arithmetic would drift.
    """
    engine = _FakeQuadEngine(used=4801, requested=1001, cap=5001)
    result = g5.quad_grid_probe(engine, product=None, env=None)
    assert result["adapted"] is True
    assert result["grid_points"] == 4801
    assert result["headroom"] == 200


def test_an_unadapted_grid_claims_no_headroom_measurement():
    """No adaptation means the required count was never revealed."""
    engine = _FakeQuadEngine(used=1001, requested=1001, cap=5001)
    result = g5.quad_grid_probe(engine, product=None, env=None)
    assert result["adapted"] is False
    assert result["headroom"] is None


def test_the_artifact_no_longer_disclaims_quad_GRID_coverage():
    """The grid disclaimer must go; the barrier-filter one must stay.

    A stale not_covered entry understates the gate as badly as a missing one
    does. But filter_unreachable_barriers SUPPRESSES a KO trigger rather than
    raising, so it is a policy choice this gate cannot fail on -- dropping
    that line would claim coverage the sweep does not have.
    """
    joined = " ".join(g5.NOT_COVERED).lower()
    assert "grid_points" not in joined, "QUAD grid resolution is covered now"
    barrier = [n for n in g5.NOT_COVERED if "unreachable" in n.lower()]
    assert len(barrier) == 1
    assert "suppress" in barrier[0].lower()


def test_the_artifact_records_which_arms_ran():
    doc = g5.build_artifact(points=[], failures=[], scope={}, config={},
                            arms={"flat_bsm": 10, "flat_bsm_quad": 10})
    assert doc["arms"] == {"flat_bsm": 10, "flat_bsm_quad": 10}


def test_a_quad_failure_is_attributed_to_the_quad_arm():
    failures = [
        {"variant": "flat_bsm_quad", "inception": "2023-05-04",
         "date": "2024-01-02", "tau_years": 2.34,
         "error": "NumericalError: ... exceeding max_adaptive_grid_points=5001"}
    ]
    doc = g5.build_artifact(points=["a"], failures=failures, scope={}, config={})
    assert doc["under_resolved"][0]["variant"] == "flat_bsm_quad"
