"""Tests for the hash-chained amendment flow."""

import copy

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.amendment import amend, validate_parent
from quantark.modelvalidation.evidence import atomic_write_json, projected_sha256, read_json
from quantark.modelvalidation.pipeline import certify

from conftest import (
    CASE_MEANS_C,
    ExplodingCandidate,
    OffsetCandidate,
    SteadyReference,
    make_scale,
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


class ExternalReference(SteadyReference):
    """A banked benchmark that refuses to stand in for itself, like the ADI 2D import's."""

    def __init__(self, benchmark="fake.telescope", **kwargs):
        super().__init__(**kwargs)
        self.benchmark = benchmark

    def config(self):
        return {"benchmark": self.benchmark, "external": True}

    def run_batch(self, case, batch_index):
        self.calls += 1
        raise ValidationError("an external benchmark is banked, never run")


TOTAL_C = 0.05  # the harness's half-width around each cell's difference
AGGREGATE_HALF_C = 0.02


@pytest.fixture
def imported(tmp_path):
    """A native certificate translated the way RELEASE_PROCEDURE section 10 imports a harness's evidence."""
    payload = copy.deepcopy(certify(make_study(), out_dir=tmp_path / "native").payload)
    payload["reference_config"] = ExternalReference().config()
    scale = make_scale()
    for cell in payload["cells"]:
        k = scale.to_economic(cell["quantity"], 1.0)
        difference = (cell["candidate_value"] - cell["reference"]["value"]) * k
        cell["source_verdict"] = {
            "status": "PASS",
            "economic_bound": 0.5,
            "estimate_difference": difference,
            "interval": [difference - TOTAL_C, difference + TOTAL_C],
            "reference_standard_error": cell["reference"]["se"] * k,
            "total_uncertainty": TOTAL_C,
        }
        cell["gate"] = {**cell["gate"], "signed_err_c": difference, "interval_c": abs(difference) + TOTAL_C}
    for aggregate in payload["aggregates"]:
        centre = aggregate["mean_signed_bias_c"]
        aggregate["source_bias"] = {"status": "PASS", "economic_bound": 0.1,
                                    "interval": [centre - AGGREGATE_HALF_C, centre + AGGREGATE_HALF_C]}
    payload["imported"] = {"reason": "translated from a harness", "gate_differences": []}
    payload["projected_sha256"] = projected_sha256(payload)
    path = tmp_path / "imported" / "certificate.json"
    path.parent.mkdir(parents=True)
    atomic_write_json(path, payload)
    return path, payload


def _amend_external(tmp_path, imported, candidate, reference=None):
    path, _ = imported
    reference = reference or ExternalReference(means_c=CASE_MEANS_C)
    study = make_study(reference=reference, candidates=(candidate,))
    return amend(study, parent=path, out_dir=tmp_path / "amended", reason="engine code moved"), reference


def test_an_external_benchmark_is_never_run_and_an_unmoved_candidate_carries_verbatim(tmp_path, imported):
    candidate = OffsetCandidate(means_c=CASE_MEANS_C)
    certificate, reference = _amend_external(tmp_path, imported, candidate)
    _, parent = imported
    payload = certificate.payload

    assert reference.calls == 0
    assert candidate.calls == 2  # an imported cell's identity is the harness's: every candidate re-runs
    assert [{k: v for k, v in c.items() if k != "carried_from"} for c in payload["cells"]] == parent["cells"]
    assert {c["carried_from"] for c in payload["cells"]} == {parent["projected_sha256"]}
    for key in ("references", "reference_config", "aggregates", "decisions", "imported"):
        assert payload[key] == parent[key]
    assert payload["amendment"]["translated_cells"] == []
    validate_parent(certificate.path)


def test_an_external_benchmark_translates_a_moved_candidate_by_the_harness_arithmetic(tmp_path, imported):
    candidate = OffsetCandidate(offset_c=1e-3, means_c=CASE_MEANS_C)
    certificate, reference = _amend_external(tmp_path, imported, candidate)
    _, parent = imported
    banked = {(c["case"], c["quantity"]): c for c in parent["cells"]}

    assert reference.calls == 0
    for cell in certificate.payload["cells"]:
        old = banked[(cell["case"], cell["quantity"])]
        shift = cell["source_verdict"]["estimate_difference"] - old["source_verdict"]["estimate_difference"]
        assert shift == pytest.approx(1e-3, rel=1e-9)
        assert cell["source_verdict"]["interval"] == pytest.approx([e + shift for e in old["source_verdict"]["interval"]])
        assert cell["gate"]["signed_err_c"] == cell["source_verdict"]["estimate_difference"]
        assert cell["gate"]["interval_c"] == max(abs(e) for e in cell["source_verdict"]["interval"])
        assert cell["candidate_value"] != old["candidate_value"]
        assert cell["verdict"] == old["verdict"] == "PASS"
        assert cell["reference"] == old["reference"]
        assert cell["translated_from"] == parent["projected_sha256"]
    translated = certificate.payload["amendment"]["translated_cells"]
    assert len(translated) == len(parent["cells"])
    assert all(t["shift_c"] == pytest.approx(1e-3, rel=1e-9) for t in translated)
    assert certificate.payload["aggregates"] == parent["aggregates"]
    assert certificate.payload["decisions"] == parent["decisions"]
    validate_parent(certificate.path)


def test_an_external_benchmark_refuses_a_move_that_would_change_a_cell_verdict(tmp_path, imported):
    with pytest.raises(ValidationError, match="harness"):
        _amend_external(tmp_path, imported, OffsetCandidate(offset_c=0.46, means_c=CASE_MEANS_C))


def test_an_external_benchmark_refuses_a_move_that_would_change_an_aggregate_verdict(tmp_path, imported):
    # every cell stays inside its 0.5 bound; the aggregate's edge plus the move reaches its 0.1 bound
    with pytest.raises(ValidationError, match="aggregate"):
        _amend_external(tmp_path, imported, OffsetCandidate(offset_c=0.09, means_c=CASE_MEANS_C))


def test_an_external_benchmark_refuses_a_changed_declaration(tmp_path, imported):
    changed = ExternalReference(benchmark="fake.other_telescope", means_c=CASE_MEANS_C)
    with pytest.raises(ValidationError, match="declared"):
        _amend_external(tmp_path, imported, OffsetCandidate(means_c=CASE_MEANS_C), reference=changed)


def test_an_external_benchmark_refuses_coverage_it_never_banked(tmp_path, imported):
    path, _ = imported
    study = make_study(reference=ExternalReference(means_c=CASE_MEANS_C),
                       candidates=(OffsetCandidate(means_c=CASE_MEANS_C),
                                   OffsetCandidate(name="fake.added", means_c=CASE_MEANS_C)))
    with pytest.raises(ValidationError, match="fake.added"):
        amend(study, parent=path, out_dir=tmp_path / "amended", reason="engine added")
