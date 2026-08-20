"""The imported ADI 2D certification: structure, honesty, and refusal.

The expensive half of this certificate -- re-running both ADI solvers over all
fourteen cells -- is already covered by ``test_banked_certificates.py``, which
globs every banked ``anchors.json``. These tests are the cheap half: they hold
the *import* itself to the rules in release procedure section 10, so a future
edit cannot quietly turn a translated certificate into one that looks native.
"""

import json
from pathlib import Path

import pytest

from quantark.modelvalidation import load_study, validate_payload
from quantark.modelvalidation.anchors import ANCHOR_SCHEMA
from quantark.util.exceptions import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = REPO_ROOT / "example/modelvalidation/adi2d_snowball_greeks.yaml"
BANKED = (
    REPO_ROOT
    / "docs/modelvalidation/certificates/adi2d-snowball-greeks/2026-08-19"
)

CANDIDATES = ("equity.snowball.heston_pde", "equity.snowball.heston_slv_pde")
CASES = (
    "ordinary_full",
    "ordinary_decayed",
    "near_ko",
    "near_ki",
    "low_feller",
    "sigma_collapse",
    "near_expiry",
)


@pytest.fixture(scope="module")
def study():
    return load_study(STUDY)


@pytest.fixture(scope="module")
def certificate():
    return json.loads((BANKED / "certificate.json").read_text())


def test_the_study_loads_with_both_adi_candidates(study):
    assert study.name == "adi2d-snowball-greeks"
    assert tuple(c.name() for c in study.candidates) == CANDIDATES
    assert tuple(c.name for c in study.cases) == CASES
    assert study.quantities == ("delta", "gamma")


def test_the_external_benchmark_refuses_to_stand_in_for_itself(study):
    """A simplified reference would make `run` appear to work while certifying
    against something the banked evidence does not describe."""
    case = study.cases[0]
    with pytest.raises(ValidationError, match="EXTERNAL benchmark"):
        study.reference.run_batch(case, 0)


def test_the_benchmark_still_describes_itself(study):
    config = study.reference.config()
    assert config["external"] is True
    assert "multilevel control-variate telescope" in config["estimator"]
    assert config["harness"].endswith("16_adi_greek_certification.py")


def test_the_study_refuses_a_quantity_this_evidence_never_certified():
    """PV was certified by the stage-11 convergence gate, not by this study."""
    text = STUDY.read_text().replace(
        "quantities: [delta, gamma]", "quantities: [pv, delta, gamma]"
    )
    from quantark.modelvalidation import load_study_text

    with pytest.raises(ValidationError, match="delta and gamma only"):
        load_study_text(text)


def test_every_case_declares_its_own_variance_regime(study):
    """The regime IS the scenario here; a case without one is a silent default."""
    for case in study.cases:
        model = dict(case.environment_params).get("heston")
        assert model, case.name
        assert set(model) == {"v0", "kappa", "theta", "sigma", "rho"}, case.name


def test_the_near_ki_case_declares_its_denser_stencil(study):
    """Declared per case, never inferred from the case name."""
    dense = [
        case.name
        for case in study.cases
        if dict(case.product_params).get("dense_ki_stencil")
    ]
    assert dense == ["near_ki"]


def test_the_banked_certificate_validates_and_admits_both_engines(certificate):
    validate_payload(certificate)
    assert certificate["decisions"] == {name: "ADMITTED" for name in CANDIDATES}
    assert len(certificate["cells"]) == len(CANDIDATES) * len(CASES) * 2
    assert all(cell["verdict"] == "PASS" for cell in certificate["cells"])


def test_the_certificate_declares_that_it_was_imported(certificate):
    """Release procedure section 10: every divergence is named, in the payload."""
    imported = certificate["imported"]
    assert imported["reason"]
    assert len(imported["gate_differences"]) >= 4
    assert imported["source_digests"]["stage16_schema"] == 13
    assert imported["source_digests"]["stage17_schema"] == 12
    for relative in imported["evidence_files"].values():
        assert (BANKED / relative).is_file(), relative


def test_the_original_payloads_are_banked_with_their_digests_intact(certificate):
    """The translation is a convenience; the original is the record."""
    digests = certificate["imported"]["source_digests"]
    stage16 = json.loads(
        (BANKED / "evidence/stage16_greek_certification.json").read_text()
    )
    stage17 = json.loads(
        (BANKED / "evidence/stage17_slv_aggregate_amendment.json").read_text()
    )
    assert stage16["evidence_sha256"] == digests["stage16_evidence_sha256"]
    assert stage17["evidence_sha256"] == digests["stage17_evidence_sha256"]
    # The amendment is byte-linked to the certification it amends.
    assert (
        stage17["parent_certificate"]["evidence_sha256"]
        == stage16["evidence_sha256"]
    )


def test_the_aggregate_edges_are_the_banked_edges(certificate):
    """mean +/- SE here is a centre and a HALF-WIDTH, so the reported edge has
    to equal the interval the certification actually defended."""
    for aggregate in certificate["aggregates"]:
        low, high = aggregate["source_bias"]["interval"]
        assert aggregate["mean_signed_bias_c"] == pytest.approx(0.5 * (low + high))
        assert aggregate["se_of_mean_c"] == pytest.approx(0.5 * (high - low))
        edge = max(abs(low), abs(high))
        assert edge < certificate["study"]["bounds"]["mean_signed_bias"]


def test_the_anchors_cover_every_certified_cell(certificate):
    anchors = json.loads((BANKED / "anchors.json").read_text())
    assert anchors["schema"] == ANCHOR_SCHEMA
    assert anchors["certificate_sha256"] == certificate["projected_sha256"]

    anchored = {(a["candidate"], a["case"]) for a in anchors["anchors"]}
    assert anchored == {(c, case) for c in CANDIDATES for case in CASES}

    banked_values = {
        (cell["candidate"], cell["case"], cell["quantity"]): cell["candidate_value"]
        for cell in certificate["cells"]
    }
    for entry in anchors["anchors"]:
        for quantity, value in entry["values"].items():
            key = (entry["candidate"], entry["case"], quantity)
            # Exact: an anchor that rounded its own source would compare a
            # different number than the certificate claims.
            assert value == banked_values[key], key


def test_the_certified_cells_are_all_discretely_monitored(study):
    """The reason main's continuous-KI work leaves this evidence standing.

    If a case ever gains continuous monitoring, the first-passage and
    Brownian-bridge machinery starts running for it and the banked numbers stop
    describing the engine -- so the property is asserted, not assumed.
    """
    from quantark.modelvalidation.builders.equity_snowball_vol import make_snowball

    for case in study.cases:
        product_spec = dict(study.candidates[0].product_params)
        product_spec.update(case.product_params)
        product = make_snowball(product_spec)
        assert product.barrier_config.ki_continuous is False, case.name
        observations = product.barrier_config.ki_observation_dates
        assert observations, case.name
        expected = max(1, round(252 * float(product_spec["maturity"])))
        assert len(observations) == expected, case.name
