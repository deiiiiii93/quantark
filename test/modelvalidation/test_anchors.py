"""Tests for deterministic anchors and their cross-architecture tolerance policy."""

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation import anchors as anchors_module
from quantark.modelvalidation.anchors import (
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
    assert anchors["rel_tol"] == 1e-12

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


def test_cross_architecture_uses_the_tolerance(tmp_path, certified, monkeypatch):
    """A different machine gets ULP-level slack -- and no more."""
    study, payload = certified
    anchors = extract_anchors(payload, study)
    anchors["fingerprint"] = {"machine": "x86_64", "system": "Linux"}
    for entry in anchors["anchors"]:
        entry["values"] = {
            q: v * (1 + 1e-13) for q, v in entry["values"].items()
        }
    path = tmp_path / "anchors.json"
    atomic_write_json(path, anchors)

    _patch_loader(monkeypatch, study)
    assert_anchors(path)  # within rel_tol


def test_cross_architecture_still_catches_real_drift(tmp_path, certified, monkeypatch):
    study, payload = certified
    anchors = extract_anchors(payload, study)
    anchors["fingerprint"] = {"machine": "x86_64", "system": "Linux"}
    for entry in anchors["anchors"]:
        entry["values"] = {q: v * (1 + 1e-9) for q, v in entry["values"].items()}
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
