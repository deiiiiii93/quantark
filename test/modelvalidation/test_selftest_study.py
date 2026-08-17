"""End-to-end integration test over real pricing engines.

The candidate is closed-form Black-Scholes, so the framework must admit it. A
failure here means the certification machinery is broken -- not the engine.
This is the module's own calibration check, and it exercises every stage:
YAML loading, benchmark sampling, gating, evidence, resume, amendment, anchors.
"""

from pathlib import Path

import pytest

from quantark.modelvalidation.amendment import amend
from quantark.modelvalidation.anchors import assert_anchors, extract_anchors
from quantark.modelvalidation.evidence import atomic_write_json
from quantark.modelvalidation.pipeline import certify
from quantark.modelvalidation.yaml_loader import load_study

STUDY_PATH = Path("example/modelvalidation/european_selftest.yaml")

CANDIDATE = "equity.european.analytical"


@pytest.fixture(scope="module")
def study():
    return load_study(STUDY_PATH)


@pytest.fixture(scope="module")
def certified(tmp_path_factory, study):
    out = tmp_path_factory.mktemp("selftest")
    return certify(study, out_dir=out), out


def test_analytically_exact_engine_is_admitted(certified):
    certificate, _ = certified
    assert certificate.payload["decisions"][CANDIDATE] == "ADMITTED"


def test_every_cell_resolved_against_a_sharp_benchmark(certified):
    """ADMITTED is only meaningful if the benchmark met its budget everywhere."""
    certificate, _ = certified
    assert all(cell["verdict"] == "PASS" for cell in certificate.payload["cells"])
    assert all(
        block["stopped_reason"] == "se_budget_met"
        for block in certificate.payload["references"].values()
    )


def test_bounds_have_real_margin(certified):
    """A test that only just passes will flap; assert the headroom explicitly."""
    certificate, _ = certified
    bound = certificate.payload["study"]["bounds"]["cell"]
    worst = max(cell["gate"]["interval_c"] for cell in certificate.payload["cells"])
    assert worst < 0.6 * bound, f"worst interval {worst:.3f} c is close to the {bound} c bound"


def test_closed_form_candidate_reports_a_zero_envelope(certified):
    certificate, _ = certified
    assert all(cell["gate"]["envelope_c"] == 0.0 for cell in certificate.payload["cells"])


def test_resume_reproduces_the_projected_digest(certified, study):
    certificate, out = certified
    again = certify(study, out_dir=out, resume=True)
    assert again.payload["projected_sha256"] == certificate.payload["projected_sha256"]


def test_amendment_carries_an_unchanged_study_forward(tmp_path, certified, study):
    certificate, _ = certified
    amended = amend(
        study,
        parent=certificate.path,
        out_dir=tmp_path,
        reason="no-op amendment (integration test)",
    )
    assert amended.payload["amendment"]["replaced_cells"] == []
    assert len(amended.payload["amendment"]["carried_cells"]) == len(
        certificate.payload["cells"]
    )
    assert amended.payload["decisions"] == certificate.payload["decisions"]


def test_anchors_round_trip_on_this_machine(tmp_path, certified, study):
    certificate, _ = certified
    path = tmp_path / "anchors.json"
    atomic_write_json(path, extract_anchors(certificate.payload, study))
    assert_anchors(path)  # exact comparison: same machine


def test_report_and_certificate_are_written(certified):
    certificate, _ = certified
    assert certificate.path.exists()
    report = (certificate.path.parent / "report.md").read_text(encoding="utf-8")
    assert "ADMITTED" in report
    assert "european-selftest" in report
