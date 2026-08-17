"""End-to-end tests for the certification pipeline over deterministic fakes."""

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.evidence import SCHEMA_VERSION, projected_sha256, read_json
from quantark.modelvalidation.pipeline import (
    Certificate,
    certify,
    quick_policy,
    runtime_environment,
    validate_payload,
)

from conftest import (
    CASE_MEANS_C,
    ExplodingCandidate,
    OffsetCandidate,
    SteadyReference,
    make_study,
)


def test_admits_a_candidate_that_matches_the_benchmark(tmp_path, study):
    certificate = certify(study, out_dir=tmp_path)
    assert isinstance(certificate, Certificate)
    assert certificate.payload["decisions"] == {"fake.candidate": "ADMITTED"}
    assert certificate.path.exists()


def test_rejects_a_candidate_biased_beyond_the_bound(tmp_path):
    study = make_study(
        candidates=(
            OffsetCandidate(name="fake.biased", offset_c=2.0, means_c=CASE_MEANS_C),
        )
    )
    certificate = certify(study, out_dir=tmp_path)
    assert certificate.payload["decisions"]["fake.biased"] == "REJECTED"


def test_small_systematic_tilt_is_caught_by_the_aggregate_gate(tmp_path):
    """Inside the per-cell bound, but the same sign in every cell."""
    study = make_study(
        candidates=(
            OffsetCandidate(name="fake.tilted", offset_c=0.3, means_c=CASE_MEANS_C),
        )
    )
    certificate = certify(study, out_dir=tmp_path)
    cells = [c for c in certificate.payload["cells"] if c["candidate"] == "fake.tilted"]
    assert all(c["verdict"] == "PASS" for c in cells)
    assert certificate.payload["decisions"]["fake.tilted"] == "REJECTED"
    assert any(not a["within_bound"] for a in certificate.payload["aggregates"])


def test_multiple_candidates_share_one_reference_bank(tmp_path):
    """One benchmark run serves every candidate -- that is why identity excludes them."""
    reference = SteadyReference(means_c=CASE_MEANS_C)
    study = make_study(
        reference=reference,
        candidates=(
            OffsetCandidate(name="fake.a", means_c=CASE_MEANS_C),
            OffsetCandidate(name="fake.b", offset_c=0.1, means_c=CASE_MEANS_C),
        ),
    )
    certify(study, out_dir=tmp_path)
    # 2 cases x 2 batches (tight series meets budget at min_batches), not x2 again
    assert reference.calls == 4
    assert len({c["candidate"] for c in _cells(tmp_path, study)}) == 2


def _cells(tmp_path, study):
    payload = read_json(tmp_path / study.name / "certificate.json")
    return payload["cells"]


def test_engine_error_is_recorded_and_caps_the_decision(tmp_path):
    study = make_study(
        candidates=(
            ExplodingCandidate(name="fake.exploding"),
            OffsetCandidate(name="fake.good", means_c=CASE_MEANS_C),
        )
    )
    certificate = certify(study, out_dir=tmp_path)
    decisions = certificate.payload["decisions"]
    assert decisions["fake.exploding"] == "INCONCLUSIVE"
    # A broken candidate must not poison a healthy one.
    assert decisions["fake.good"] == "ADMITTED"

    errored = [c for c in certificate.payload["cells"] if c["candidate"] == "fake.exploding"]
    assert errored and all(c["verdict"] == "ERROR" for c in errored)
    assert all("engine blew up" in c["error"] for c in errored)


def test_payload_shape_and_hash(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["study"]["name"] == "fake-study"
    assert payload["study"]["source_text"] == "study: fake-study\n"
    assert set(payload["references"]) == {"ordinary", "near_ko"}
    assert payload["runtime"]["python"]
    assert payload["projected_sha256"] == projected_sha256(payload)
    assert len(payload["cells"]) == 2 * 3  # cases x quantities x 1 candidate


def test_cells_carry_gate_numbers(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    cell = payload["cells"][0]
    assert {"candidate", "case", "quantity", "reference", "candidate_value", "gate"} <= set(cell)
    assert "signed_err_c" in cell["gate"]
    assert cell["gate"]["envelope_c"] is not None  # fakes supply ladders


def test_resume_reproduces_the_projected_hash(tmp_path, study):
    first = certify(study, out_dir=tmp_path)
    again = certify(make_study(), out_dir=tmp_path, resume=True)
    assert again.payload["projected_sha256"] == first.payload["projected_sha256"]


def test_resume_skips_completed_work(tmp_path):
    reference = SteadyReference(means_c=CASE_MEANS_C)
    candidate = OffsetCandidate(means_c=CASE_MEANS_C)
    study = make_study(reference=reference, candidates=(candidate,))
    certify(study, out_dir=tmp_path)
    assert reference.calls > 0 and candidate.calls > 0

    reference2 = SteadyReference(means_c=CASE_MEANS_C)
    candidate2 = OffsetCandidate(means_c=CASE_MEANS_C)
    certify(
        make_study(reference=reference2, candidates=(candidate2,)),
        out_dir=tmp_path,
        resume=True,
    )
    assert reference2.calls == 0
    assert candidate2.calls == 0


def test_resume_reruns_a_changed_candidate(tmp_path):
    certify(make_study(), out_dir=tmp_path)
    changed = OffsetCandidate(name="fake.candidate", offset_c=0.2, means_c=CASE_MEANS_C)
    certify(make_study(candidates=(changed,)), out_dir=tmp_path, resume=True)
    assert changed.calls == 2  # identity changed -> re-evaluated for both cases


def test_quick_mode_shrinks_sampling(tmp_path):
    reference = SteadyReference(means_c=CASE_MEANS_C)
    study = make_study(reference=reference)
    certificate = certify(study, out_dir=tmp_path, quick=True)
    assert certificate.payload["study"]["quick"] is True
    assert certificate.payload["study"]["sampling"]["max_batches"] <= 4


def test_quick_policy_is_deterministic():
    from quantark.modelvalidation.study import SamplingPolicy

    policy = SamplingPolicy(paths_per_batch=65536, min_batches=8, max_batches=64, seed=7)
    quick = quick_policy(policy)
    assert quick.paths_per_batch == 8192
    assert quick.min_batches == 2
    assert quick.max_batches == 4
    assert quick.seed == policy.seed
    assert quick.bump == policy.bump


def test_quick_policy_never_inverts_the_batch_limits():
    from quantark.modelvalidation.study import SamplingPolicy

    policy = SamplingPolicy(paths_per_batch=64, min_batches=2, max_batches=3, seed=7)
    quick = quick_policy(policy)
    assert quick.min_batches <= quick.max_batches
    assert quick.paths_per_batch >= 1


def test_runtime_environment_reports_the_machine():
    env = runtime_environment()
    assert env["platform"] and env["machine"] and env["python"] and env["numpy"]


def test_validate_payload_rejects_a_tampered_hash(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    payload["cells"][0]["candidate_value"] = 12345.0
    with pytest.raises(ValidationError):
        validate_payload(payload)


def test_validate_payload_rejects_an_unknown_verdict(tmp_path, study):
    from quantark.modelvalidation.evidence import projected_sha256 as digest

    payload = certify(study, out_dir=tmp_path).payload
    payload["cells"][0]["verdict"] = "MAYBE"
    payload["projected_sha256"] = digest(payload)
    with pytest.raises(ValidationError):
        validate_payload(payload)


def test_validate_payload_rejects_a_wrong_schema(tmp_path, study):
    payload = certify(study, out_dir=tmp_path).payload
    payload["schema"] = 99
    with pytest.raises(ValidationError):
        validate_payload(payload)


def test_certify_refuses_a_temp_output_root(study):
    with pytest.raises(ValidationError):
        certify(study, out_dir="/tmp/modelvalidation-should-not-write")
