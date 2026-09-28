"""Tests for deterministic anchors and their cross-architecture tolerance policy."""

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation import anchors as anchors_module
from quantark.modelvalidation.anchors import (
    DEFAULT_ABS_TOL,
    DEFAULT_REL_TOL,
    assert_anchors,
    extract_anchors,
    machine_fingerprint,
)
from quantark.modelvalidation.evidence import atomic_write_json, read_json
from quantark.modelvalidation.pipeline import certify

from conftest import CASE_MEANS_C, ExplodingCandidate, OffsetCandidate, make_study


@pytest.fixture
def certified(tmp_path):
    study = make_study()
    payload = certify(study, out_dir=tmp_path).payload
    return study, payload


def _patch_loader(monkeypatch, study):
    """Anchors reconstruct a study from its source text; fake that here so the
    anchor tests do not depend on the YAML loader."""
    monkeypatch.setattr(anchors_module, "load_study_text", lambda text: study)


def test_extract_anchors_captures_candidate_values(certified):
    study, payload = certified
    anchors = extract_anchors(payload, study)
    assert anchors["schema"] == 1
    assert anchors["study_source_text"] == study.source_text
    assert anchors["fingerprint"] == machine_fingerprint()
    assert anchors["rel_tol"] == DEFAULT_REL_TOL
    assert anchors["abs_tol"] == DEFAULT_ABS_TOL

    entries = {(a["candidate"], a["case"]): a for a in anchors["anchors"]}
    assert set(entries) == {
        ("fake.candidate", "ordinary"),
        ("fake.candidate", "near_ko"),
    }
    assert set(entries[("fake.candidate", "ordinary")]["values"]) == {
        "pv",
        "delta",
        "gamma",
    }


def test_extract_anchors_skips_errored_cells(tmp_path):
    study = make_study(
        candidates=(
            ExplodingCandidate(name="fake.exploding"),
            OffsetCandidate(name="fake.good", means_c=CASE_MEANS_C),
        )
    )
    payload = certify(study, out_dir=tmp_path).payload
    anchors = extract_anchors(payload, study)
    assert {a["candidate"] for a in anchors["anchors"]} == {"fake.good"}


def test_extract_anchors_requires_source_text(tmp_path):
    study = make_study(source_text=None)
    payload = certify(study, out_dir=tmp_path).payload
    with pytest.raises(ValidationError):
        extract_anchors(payload, study)


def test_assert_anchors_passes_on_the_banking_machine(tmp_path, certified, monkeypatch):
    study, payload = certified
    _patch_loader(monkeypatch, study)
    path = tmp_path / "anchors.json"
    atomic_write_json(path, extract_anchors(payload, study))
    assert_anchors(path)  # must not raise


def test_assert_anchors_detects_a_changed_engine(tmp_path, certified, monkeypatch):
    study, payload = certified
    path = tmp_path / "anchors.json"
    atomic_write_json(path, extract_anchors(payload, study))

    # The engine now returns a slightly different number: exactly what an
    # anchor test exists to catch.
    drifted = make_study(
        candidates=(
            OffsetCandidate(name="fake.candidate", offset_c=1e-6, means_c=CASE_MEANS_C),
        )
    )
    _patch_loader(monkeypatch, drifted)
    with pytest.raises(AssertionError) as exc:
        assert_anchors(path)
    assert "fake.candidate" in str(exc.value)


def test_same_machine_comparison_is_exact(tmp_path, certified, monkeypatch):
    """On the banking machine an anchor is bitwise; nothing is allowed to drift."""
    study, payload = certified
    anchors = extract_anchors(payload, study)
    entry = anchors["anchors"][0]
    entry["values"]["delta"] = entry["values"]["delta"] * (1 + 1e-15)
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    with pytest.raises(AssertionError):
        assert_anchors(path)


def _a_different_machine() -> dict:
    """A fingerprint that cannot be THIS machine, whatever this machine is.

    These tests used to hardcode ``{"machine": "x86_64", "system": "Linux"}``,
    which made them exercise the cross-architecture path only on machines that
    were not x86_64 Linux. CI is exactly that, so on the first run there the
    comparison went down the same-machine EXACT branch and the tests failed --
    reporting a numerics problem where there was only a bad stub.
    """
    fingerprint = dict(machine_fingerprint())
    fingerprint["machine"] = f"{fingerprint['machine']}-not-this-one"
    return fingerprint


def test_cross_architecture_uses_the_tolerance(tmp_path, certified, monkeypatch):
    """A different machine gets ULP-level slack -- and no more."""
    study, payload = certified
    anchors = extract_anchors(payload, study)
    anchors["fingerprint"] = _a_different_machine()
    for entry in anchors["anchors"]:
        entry["values"] = {
            q: v * (1 + 1e-11) for q, v in entry["values"].items()
        }
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    assert_anchors(path)  # within rel_tol


def test_cross_architecture_still_catches_real_drift(tmp_path, certified, monkeypatch):
    study, payload = certified
    anchors = extract_anchors(payload, study)
    anchors["fingerprint"] = _a_different_machine()
    for entry in anchors["anchors"]:
        entry["values"] = {q: v * (1 + 1e-6) for q, v in entry["values"].items()}
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    with pytest.raises(AssertionError):
        assert_anchors(path)


def test_assert_anchors_reports_every_mismatch(tmp_path, certified, monkeypatch):
    """A reviewer needs the whole diff, not just the first failure."""
    study, payload = certified
    anchors = extract_anchors(payload, study)
    for entry in anchors["anchors"]:
        entry["values"] = {q: v + 1.0 for q, v in entry["values"].items()}
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    with pytest.raises(AssertionError) as exc:
        assert_anchors(path)
    message = str(exc.value)
    assert message.count("ordinary") >= 1 and message.count("near_ko") >= 1


def test_assert_anchors_rejects_a_wrong_schema(tmp_path, certified, monkeypatch):
    study, payload = certified
    anchors = extract_anchors(payload, study)
    anchors["schema"] = 99
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    with pytest.raises(ValidationError):
        assert_anchors(path)


def test_anchor_file_round_trips(tmp_path, certified):
    study, payload = certified
    path = tmp_path / "anchors.json"
    anchors = extract_anchors(payload, study)
    atomic_write_json(path, anchors)
    assert read_json(path) == anchors


# --- finite-difference quantities off the banking machine ------------------
#
# A quantity formed by differencing prices, Q = sum_i w_i V_i, carries the
# prices' cross-architecture noise amplified by the stencil: about
# eps * |V| * sum_i |w_i|. Measured on the first CI run of the intraday
# certificate: a one-second desk theta moved 2.2e-8 relative while every PV it
# differences agreed to 7.5e-12 -- architecture noise divided by a one-second
# step, not a behaviour change.


class StencilCandidate(OffsetCandidate):
    """A candidate that declares how its quantities difference prices."""

    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = dict(weights)

    def anchor_noise_weights(self, case):
        return dict(self.weights)


def _stencil_study(weights):
    return make_study(candidates=(StencilCandidate(weights, name="fake.candidate", means_c=CASE_MEANS_C),))


def test_a_declared_stencil_weight_scales_the_cross_arch_tolerance_with_the_price(tmp_path, monkeypatch):
    weight = 1.0e4
    study = _stencil_study({"gamma": weight})
    anchors = extract_anchors(certify(study, out_dir=tmp_path).payload, study)
    anchors["fingerprint"] = _a_different_machine()
    for entry in anchors["anchors"]:
        values = entry["values"]
        price_noise = DEFAULT_REL_TOL * abs(values["pv"]) * weight
        assert price_noise > 10 * DEFAULT_REL_TOL * abs(values["gamma"])  # the undeclared rule would reject it
        values["gamma"] += 0.5 * price_noise
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    assert_anchors(path)  # half the propagated price noise: architecture, not a change


def test_a_stencil_weight_still_rejects_what_the_prices_would_reject(tmp_path, monkeypatch):
    weight = 1.0e4
    study = _stencil_study({"gamma": weight})
    anchors = extract_anchors(certify(study, out_dir=tmp_path).payload, study)
    anchors["fingerprint"] = _a_different_machine()
    for entry in anchors["anchors"]:
        entry["values"]["gamma"] += 2.0 * DEFAULT_REL_TOL * abs(entry["values"]["pv"]) * weight
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    with pytest.raises(AssertionError, match="gamma"):
        assert_anchors(path)


def test_a_stencil_weight_never_loosens_the_same_machine_comparison(tmp_path, monkeypatch):
    study = _stencil_study({"gamma": 1.0e12})
    anchors = extract_anchors(certify(study, out_dir=tmp_path).payload, study)
    anchors["anchors"][0]["values"]["gamma"] *= 1 + 1e-15
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    with pytest.raises(AssertionError):
        assert_anchors(path)


def test_an_undeclared_quantity_keeps_the_relative_tolerance(tmp_path, monkeypatch):
    """Weights are per quantity: declaring gamma's stencil says nothing about delta."""
    study = _stencil_study({"gamma": 1.0e12})
    anchors = extract_anchors(certify(study, out_dir=tmp_path).payload, study)
    anchors["fingerprint"] = _a_different_machine()
    for entry in anchors["anchors"]:
        entry["values"]["delta"] *= 1 + 1e-6
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    with pytest.raises(AssertionError, match="delta"):
        assert_anchors(path)


def test_anchor_tolerance_is_the_larger_of_the_value_and_the_propagated_price_noise():
    tol = anchors_module.anchor_tolerance
    assert tol(2.0, pv=None, weight=None, rel_tol=1e-9, abs_tol=1e-12) == pytest.approx(2e-9 + 1e-12)
    assert tol(2.0, pv=10.0, weight=1.0e3, rel_tol=1e-9, abs_tol=1e-12) == pytest.approx(1e-5 + 1e-12)
    assert tol(2.0, pv=10.0, weight=1.0e-3, rel_tol=1e-9, abs_tol=1e-12) == pytest.approx(2e-9 + 1e-12)
    with pytest.raises(ValidationError):
        tol(2.0, pv=10.0, weight=-1.0, rel_tol=1e-9, abs_tol=1e-12)
