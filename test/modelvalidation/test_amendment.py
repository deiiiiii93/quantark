"""Tests for the hash-chained amendment flow."""

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.amendment import amend, validate_parent
from quantark.modelvalidation.evidence import atomic_write_json, read_json
from quantark.modelvalidation.pipeline import certify

from conftest import (
    CASE_MEANS_C,
    ExplodingCandidate,
    OffsetCandidate,
    SteadyReference,
    make_study,
)


@pytest.fixture
def parent(tmp_path):
    return certify(make_study(), out_dir=tmp_path / "parent")


def test_validate_parent_accepts_a_written_certificate(parent):
    payload = validate_parent(parent.path)
    assert payload["projected_sha256"] == parent.payload["projected_sha256"]


def test_validate_parent_rejects_a_tampered_certificate(parent):
    payload = read_json(parent.path)
    payload["cells"][0]["candidate_value"] = 99.0
    atomic_write_json(parent.path, payload)
    with pytest.raises(ValidationError):
        validate_parent(parent.path)


def test_validate_parent_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValidationError):
        validate_parent(tmp_path / "absent.json")


def test_unchanged_study_carries_everything_forward(tmp_path, parent):
    reference = SteadyReference(means_c=CASE_MEANS_C)
    candidate = OffsetCandidate(means_c=CASE_MEANS_C)
    study = make_study(reference=reference, candidates=(candidate,))

    certificate = amend(
        study,
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="no-op amendment",
    )

    assert reference.calls == 0
    assert candidate.calls == 0
    assert certificate.payload["amendment"]["replaced_cells"] == []
    assert len(certificate.payload["amendment"]["carried_cells"]) == len(
        parent.payload["cells"]
    )
    assert certificate.payload["decisions"] == parent.payload["decisions"]


def test_amendment_records_its_lineage(tmp_path, parent):
    certificate = amend(
        make_study(),
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="grid refinement",
    )
    amendment = certificate.payload["amendment"]
    assert amendment["reason"] == "grid refinement"
    assert amendment["parent_projected_sha256"] == parent.payload["projected_sha256"]
    assert str(parent.path) in amendment["parent"]


def test_changed_candidate_replaces_only_its_own_cells(tmp_path, parent):
    changed = OffsetCandidate(name="fake.candidate", offset_c=0.2, means_c=CASE_MEANS_C)
    reference = SteadyReference(means_c=CASE_MEANS_C)
    certificate = amend(
        make_study(reference=reference, candidates=(changed,)),
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="candidate re-tuned",
    )

    assert changed.calls == 2  # both cases re-evaluated
    assert reference.calls == 0  # benchmark identity unchanged -> reused
    replaced = certificate.payload["amendment"]["replaced_cells"]
    assert len(replaced) == len(parent.payload["cells"])
    assert certificate.payload["amendment"]["carried_cells"] == []


def test_new_candidate_is_run_and_old_one_carried(tmp_path, parent):
    kept = OffsetCandidate(name="fake.candidate", means_c=CASE_MEANS_C)
    added = OffsetCandidate(name="fake.added", offset_c=0.1, means_c=CASE_MEANS_C)
    certificate = amend(
        make_study(candidates=(kept, added)),
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="second engine added",
    )

    assert kept.calls == 0  # unchanged -> carried
    assert added.calls == 2  # new -> evaluated
    assert set(certificate.payload["decisions"]) == {"fake.candidate", "fake.added"}


def test_amendment_cannot_silently_drop_a_parent_case(tmp_path, parent):
    from quantark.modelvalidation.study import CaseSpec

    shrunk = make_study(cases=(CaseSpec(name="ordinary"),))
    with pytest.raises(ValidationError) as exc:
        amend(
            shrunk,
            parent=parent.path,
            out_dir=tmp_path / "amended",
            reason="dropping coverage",
        )
    assert "near_ko" in str(exc.value)


def test_amendment_cannot_silently_drop_a_parent_candidate(tmp_path):
    parent = certify(
        make_study(
            candidates=(
                OffsetCandidate(name="fake.a", means_c=CASE_MEANS_C),
                OffsetCandidate(name="fake.b", means_c=CASE_MEANS_C),
            )
        ),
        out_dir=tmp_path / "parent",
    )
    with pytest.raises(ValidationError) as exc:
        amend(
            make_study(candidates=(OffsetCandidate(name="fake.a", means_c=CASE_MEANS_C),)),
            parent=parent.path,
            out_dir=tmp_path / "amended",
            reason="dropping an engine",
        )
    assert "fake.b" in str(exc.value)


def test_new_case_is_certified_and_others_carried(tmp_path, parent):
    from quantark.modelvalidation.study import CaseSpec

    means = dict(CASE_MEANS_C, near_ki=3.0)
    reference = SteadyReference(means_c=means)
    candidate = OffsetCandidate(name="fake.candidate", means_c=means)
    study = make_study(
        cases=(
            CaseSpec(name="ordinary"),
            CaseSpec(name="near_ko"),
            CaseSpec(name="near_ki"),
        ),
        reference=reference,
        candidates=(candidate,),
        means_c=means,
    )
    certificate = amend(
        study,
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="scenario added",
    )

    assert reference.calls > 0  # only the new case sampled
    assert candidate.calls == 1  # only the new case evaluated
    cases_in_cells = {c["case"] for c in certificate.payload["cells"]}
    assert cases_in_cells == {"ordinary", "near_ko", "near_ki"}


def test_amendment_requires_a_reason(tmp_path, parent):
    with pytest.raises(ValidationError):
        amend(make_study(), parent=parent.path, out_dir=tmp_path / "amended", reason="")


def test_amendment_payload_validates_and_writes(tmp_path, parent):
    certificate = amend(
        make_study(),
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="no-op",
    )
    assert certificate.path.exists()
    on_disk = read_json(certificate.path)
    validate_parent(certificate.path)  # an amendment is itself a valid parent
    assert on_disk["projected_sha256"] == certificate.payload["projected_sha256"]


def test_amendment_of_an_amendment_chains(tmp_path, parent):
    first = amend(
        make_study(), parent=parent.path, out_dir=tmp_path / "a1", reason="first"
    )
    second = amend(
        make_study(), parent=first.path, out_dir=tmp_path / "a2", reason="second"
    )
    assert (
        second.payload["amendment"]["parent_projected_sha256"]
        == first.payload["projected_sha256"]
    )


def test_carrying_forward_an_errored_cell_keeps_it_errored(tmp_path):
    parent = certify(
        make_study(candidates=(ExplodingCandidate(name="fake.exploding"),)),
        out_dir=tmp_path / "parent",
    )
    certificate = amend(
        make_study(candidates=(ExplodingCandidate(name="fake.exploding"),)),
        parent=parent.path,
        out_dir=tmp_path / "amended",
        reason="carry the failure forward",
    )
    assert all(c["verdict"] == "ERROR" for c in certificate.payload["cells"])
    assert certificate.payload["decisions"]["fake.exploding"] == "INCONCLUSIVE"
