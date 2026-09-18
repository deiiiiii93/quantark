"""Schema-2 protocols: statuses, convergence evidence, resolved identities, targets, bound references, substreams."""
import dataclasses

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.candidate import (
    CandidateResult,
    ConvergenceAxis,
    ConvergenceLevel,
    candidate_identity,
    convergence_evidence,
    deserialize_candidate_result,
    serialize_candidate_result,
)
from quantark.modelvalidation.evidence import identity_hash
from quantark.modelvalidation.reference import BatchResult, bound_reference, reference_targets, run_reference
from quantark.modelvalidation.study import CaseSpec, GateBounds, NormalizedScale, SamplingPolicy, batch_seed

from conftest import OffsetCandidate, SteadyReference


# --- statuses ------------------------------------------------------------------------------------------------------
def test_statuses_default_to_ok_for_reported_values_and_failed_otherwise():
    result = CandidateResult(values={"pv": 1.0})
    assert result.status("pv") == "ok" and result.status("point_delta") == "failed"


def test_a_non_ok_status_cannot_carry_a_value():
    with pytest.raises(ValidationError, match="undefined"):
        CandidateResult(values={"pv": 1.0, "point_delta": 0.5}, statuses={"point_delta": "undefined"})
    with pytest.raises(ValidationError, match="statuses"):
        CandidateResult(values={"pv": 1.0}, statuses={"point_delta": "unqualified"})


# --- convergence evidence (review R5) -----------------------------------------------------------------------------
def _axis(name, values, kind="refinement", target=0):
    return ConvergenceAxis(name, kind, tuple(
        ConvergenceLevel(f"l{i}", float(2 ** i if kind == "refinement" else i), {"pv": v}, {"points": 100 * 2 ** i},
                         is_target=(i == target)) for i, v in enumerate(values)))


def test_envelope_is_the_target_to_next_finer_increment_summed_over_axes():
    result = CandidateResult(values={"pv": 1.0}, convergence=(_axis("space", [1.0, 1.04, 1.05]), _axis("time", [1.0, 0.99, 0.9875])))
    evidence = convergence_evidence(result, "pv")
    assert evidence.complete and evidence.envelope == pytest.approx(0.04 + 0.01)
    assert evidence.observed_orders["space"] == pytest.approx(2.0)      # 0.04 -> 0.01 on a doubling
    assert evidence.observed_orders["time"] == pytest.approx(2.0)
    assert evidence.non_monotone == ()


def test_a_placement_axis_contributes_its_largest_shift():
    result = CandidateResult(values={"pv": 1.0}, convergence=(_axis("placement", [1.0, 1.003, 0.998, 1.001], kind="placement"),))
    assert convergence_evidence(result, "pv").envelope == pytest.approx(0.003)


def test_fewer_than_three_levels_is_incomplete_not_acceptable():
    result = CandidateResult(values={"pv": 1.0}, convergence=(_axis("space", [1.0, 1.04]),))
    evidence = convergence_evidence(result, "pv")
    assert not evidence.complete and evidence.envelope is None and evidence.missing == ("space",)


def test_a_level_missing_the_quantity_does_not_count():
    axis = ConvergenceAxis("space", "refinement", (
        ConvergenceLevel("t", 1.0, {"pv": 1.0, "desk_gamma": 2.0}, is_target=True),
        ConvergenceLevel("x2", 2.0, {"pv": 1.01}),
        ConvergenceLevel("x4", 4.0, {"pv": 1.0125, "desk_gamma": 2.2})))
    result = CandidateResult(values={"pv": 1.0, "desk_gamma": 2.0}, convergence=(axis,))
    assert convergence_evidence(result, "pv").complete
    assert convergence_evidence(result, "desk_gamma").missing == ("space",)


def test_no_declared_evidence_is_incomplete_unless_the_candidate_is_exact():
    assert convergence_evidence(CandidateResult(values={"pv": 1.0}), "pv").missing == ("<no axis declared>",)
    exact = convergence_evidence(CandidateResult(values={"pv": 1.0}, exact=True), "pv")
    assert exact.complete and exact.envelope == 0.0


def test_an_oscillating_ladder_is_reported():
    result = CandidateResult(values={"pv": 1.0}, convergence=(_axis("space", [1.0, 1.01, 0.97]),))
    assert convergence_evidence(result, "pv").non_monotone == ("space",)


def test_an_axis_needs_exactly_one_target():
    with pytest.raises(ValidationError, match="target"):
        ConvergenceAxis("space", "refinement", (ConvergenceLevel("a", 1.0, {"pv": 1.0}),))
    with pytest.raises(ValidationError, match="kind"):
        ConvergenceAxis("space", "sideways", (ConvergenceLevel("a", 1.0, {"pv": 1.0}, is_target=True),))


# --- serialization keeps the schema-1 checkpoint shape (review R2) ------------------------------------------------
def test_a_schema_1_result_serializes_exactly_as_before():
    result = OffsetCandidate().evaluate(CaseSpec(name="ordinary"))
    assert sorted(serialize_candidate_result(result)) == ["ladders", "values"]


def test_schema_2_fields_round_trip():
    result = CandidateResult(values={"pv": 1.0}, statuses={"pv": "ok", "point_delta": "undefined"},
                             reasons={"point_delta": "spot on the barrier"}, convergence=(_axis("space", [1.0, 1.04, 1.05]),))
    back = deserialize_candidate_result(serialize_candidate_result(result))
    assert back.statuses == result.statuses and back.reasons == result.reasons
    assert back.convergence == result.convergence and back.exact is False


# --- identities (review R1, R2) ------------------------------------------------------------------------------------
class ResolvedCandidate(OffsetCandidate):
    def __init__(self, vol=0.20, **kwargs):
        super().__init__(**kwargs)
        self.vol = vol

    def resolved_inputs(self, case):
        return {"environment": {"vol": self.vol, **case.environment_params}, "context": dict(case.context_params),
                "quantities": ["pv"]}

    def fingerprint(self):
        return "fp-1"


def test_schema_1_identity_is_exactly_the_original():
    assert candidate_identity(OffsetCandidate(), CaseSpec(name="ordinary")) == {
        "candidate": "fake.candidate", "params": {"offset_c": 0.0, "envelope_c": 0.0},
        "case": {"name": "ordinary", "environment_params": {}, "product_params": {}}}


def test_schema_2_identity_moves_with_a_same_name_context_change():
    early = CaseSpec(name="near_ki", context_params={"valuation": "2026-09-10T14:00:00+08:00"})
    late = CaseSpec(name="near_ki", context_params={"valuation": "2026-09-10T14:59:59+08:00"})
    candidate = ResolvedCandidate()
    assert identity_hash(candidate_identity(candidate, early, schema=2)) != identity_hash(candidate_identity(candidate, late, schema=2))


def test_schema_2_identity_moves_with_a_study_level_market_change():
    case = CaseSpec(name="ordinary")
    assert identity_hash(candidate_identity(ResolvedCandidate(vol=0.20), case, schema=2)) != \
        identity_hash(candidate_identity(ResolvedCandidate(vol=0.35), case, schema=2))


def test_schema_2_identity_moves_with_a_semantic_expectation_and_carries_the_implementation():
    candidate = ResolvedCandidate()
    plain = candidate_identity(candidate, CaseSpec(name="a"), schema=2)
    expecting = candidate_identity(candidate, CaseSpec(name="a", expected={"point_delta": "undefined"}), schema=2)
    assert plain["implementation"] == "fp-1" and identity_hash(plain) != identity_hash(expecting)


def test_a_schema_2_candidate_must_declare_its_inputs_and_fingerprint():
    with pytest.raises(ValidationError, match="resolved_inputs"):
        candidate_identity(OffsetCandidate(), CaseSpec(name="a"), schema=2)


# --- reference targets, binding and substreams (review R3, R10) ---------------------------------------------------
def test_reference_targets_default_to_replicate_means_and_honour_a_declaration():
    assert reference_targets(SteadyReference(), ("pv",)) == {"pv": {"estimator": "replicate_mean"}}

    class Declaring(SteadyReference):
        def targets(self):
            return {"desk_delta": {"estimator": "paired_central_difference", "bump": 0.01}, "point_delta": None}

    targets = reference_targets(Declaring(), ("pv", "desk_delta", "point_delta"))
    assert targets["desk_delta"]["bump"] == 0.01 and targets["point_delta"] is None


class PolicyBound:
    """A reference that captures its policy at construction, as every real builder does."""

    seen = []

    def __init__(self, sampling):
        self.sampling = sampling

    def bind(self, policy):
        return PolicyBound(policy)

    def identity(self, case):
        return {"builder": "fake.bound", "case": case.name, "paths": self.sampling.paths_per_batch}

    def run_batch(self, case, batch_index):
        PolicyBound.seen.append((case.name, self.sampling.paths_per_batch))
        return BatchResult(index=batch_index, seed=batch_seed(self.sampling, case.name, batch_index),
                           values={"pv": 1.0 + 1e-9 * batch_index})


FULL = SamplingPolicy(paths_per_batch=65536, min_batches=2, max_batches=2, seed=20260918, seed_scheme="substream")


def test_a_bound_reference_runs_the_effective_policy():
    quick = dataclasses.replace(FULL, paths_per_batch=8192)
    bound = bound_reference(PolicyBound(FULL), quick, schema=2)
    PolicyBound.seen.clear()
    run_reference(builder=bound, case=CaseSpec(name="ordinary"), quantities=("pv",), scale=NormalizedScale(100.0, 100.0),
                  bounds=GateBounds(cell=1.0, mean_signed_bias=0.2), policy=quick)
    assert {paths for _, paths in PolicyBound.seen} == {8192}
    assert bound.identity(CaseSpec(name="ordinary"))["paths"] == 8192


def test_a_schema_2_reference_must_be_bindable_and_a_schema_1_one_passes_through():
    plain = SteadyReference()
    assert bound_reference(plain, FULL, schema=1) is plain
    with pytest.raises(ValidationError, match="bind"):
        bound_reference(plain, FULL, schema=2)


def test_a_substream_batch_with_the_sequential_seed_is_rejected():
    class Sequential(PolicyBound):
        def run_batch(self, case, batch_index):
            return BatchResult(index=batch_index, seed=self.sampling.seed + batch_index, values={"pv": 1.0})

    with pytest.raises(ValidationError, match="seed"):
        run_reference(builder=Sequential(FULL), case=CaseSpec(name="ordinary"), quantities=("pv",),
                      scale=NormalizedScale(100.0, 100.0), bounds=GateBounds(cell=1.0, mean_signed_bias=0.2), policy=FULL)
