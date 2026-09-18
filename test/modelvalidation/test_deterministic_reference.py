"""The deterministic reference kind (study revision 2026-09-18): a declared radius instead of a standard error,
linear aggregation, a typed record, and the stochastic qualifier that must agree with it case by case."""
import dataclasses
import math

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.gates import evaluate_aggregate_gate, evaluate_cell_gate
from quantark.modelvalidation.html_report import render_html
from quantark.modelvalidation.pipeline import certify, validate_payload
from quantark.modelvalidation.reference import BatchResult, DeterministicResult
from quantark.modelvalidation.report import render_markdown
from quantark.modelvalidation.study import (
    CaseSpec, CertificationStudy, GateBounds, NormalizedScale, QuantityBounds, ReferenceQualification, SamplingPolicy,
    batch_seed,
)

from test_pipeline_schema2 import Candidate, HONEST, Reference as StochasticReference

QUANTITIES = ("pv", "desk_delta", "point_delta")
BOUNDS = {"pv": QuantityBounds(1e-6), "desk_delta": QuantityBounds(1e-5, 1e-4), "point_delta": QuantityBounds(1e-5, 1e-4)}
TRUTH = {"pv": 0.5, "desk_delta": 0.3, "point_delta": 0.3}
SAMPLING = SamplingPolicy(paths_per_batch=4096, min_batches=2, max_batches=2, seed=7, seed_scheme="substream")
GATES = GateBounds(cell=1.0, mean_signed_bias=0.2, radius_budget_fraction=0.25)
SCALE = NormalizedScale(100.0, 100.0)
PV_BUDGET_RAW = 1e-4          # pv normalizes as raw / notional (100) against a 1e-6 budget
CASES = (CaseSpec(name="ordinary", context_params={"valuation": "t0"}),
         CaseSpec(name="on_barrier", context_params={"valuation": "t1"}, expected={"point_delta": "undefined"}))


class Deterministic:
    """A solved ladder: every quantity is a target, point_delta has no value on the barrier."""

    reference_kind = "deterministic"

    def __init__(self, radii=None, shift=None, quick=False):
        self.radii, self.shift, self.quick, self.solves = radii or {}, shift or {}, quick, []

    def bind(self, policy, quick=False):
        bound = Deterministic(self.radii, self.shift, quick)
        bound.solves = self.solves
        return bound

    def targets(self):
        return {q: {"estimator": "richardson_extrapolant"} for q in QUANTITIES}

    def error_model(self):
        return {"method": "fake ladder", "radius": "declared"}

    def identity(self, case):
        return {"builder": "fake.det", "case": case.name, "context": dict(case.context_params), "radii": self.radii,
                "shift": self.shift, "quick": self.quick}

    def solve(self, case):
        self.solves.append(case.name)
        undefined = {"point_delta": "valuation is on the event"} if case.name == "on_barrier" else {}
        values = {q: TRUTH[q] + self.shift.get(q, 0.0) for q in QUANTITIES if q not in undefined}
        return DeterministicResult(values=values, radii={q: self.radii.get(q, 1e-9) for q in values},
                                   evidence={"levels": [101, 201, 401, 801]}, undefined=undefined)


def make_study(**overrides):
    kwargs = dict(name="fake-det", schema=2, quantities=QUANTITIES, quantity_bounds=BOUNDS, cases=CASES, bounds=GATES,
                  scale=SCALE, reference=Deterministic(), candidates=(Candidate(statuses=HONEST),), sampling=SAMPLING,
                  source_text="study: fake-det\n")
    kwargs.update(overrides)
    return CertificationStudy(**kwargs)


def _cell(payload, case, quantity):
    return next(c for c in payload["cells"] if (c["case"], c["quantity"]) == (case, quantity))


# --- gates ---------------------------------------------------------------------------------------------------------
def test_a_radius_consumes_the_budget_as_a_bound_not_as_k_standard_errors():
    gate = evaluate_cell_gate(candidate_raw=0.5 + 0.7 * PV_BUDGET_RAW, reference_raw=0.5, reference_se_raw=None,
                              reference_radius_raw=0.2 * PV_BUDGET_RAW, quantity="pv", scale=SCALE, bounds=GATES, bound_c=1e-6)
    assert gate.radius_c == pytest.approx(0.2) and gate.se_c is None
    assert gate.interval_c == pytest.approx(0.9) and gate.lower_error_c == pytest.approx(0.5)      # 0.7 +/- 0.2, no k
    assert gate.se_budget_met and gate.interval_within_bound and gate.passed


def test_a_radius_above_its_fraction_of_the_budget_is_not_sharp_enough():
    gate = evaluate_cell_gate(candidate_raw=0.5, reference_raw=0.5, reference_se_raw=None,
                              reference_radius_raw=0.3 * PV_BUDGET_RAW, quantity="pv", scale=SCALE, bounds=GATES, bound_c=1e-6)
    assert not gate.se_budget_met and not gate.passed


def test_a_gate_takes_a_standard_error_or_a_radius_never_both_and_never_neither():
    common = dict(candidate_raw=0.5, reference_raw=0.5, quantity="pv", scale=SCALE, bounds=GATES, bound_c=1e-6)
    with pytest.raises(ValidationError, match="exactly one"):
        evaluate_cell_gate(reference_se_raw=1e-9, reference_radius_raw=1e-9, **common)
    with pytest.raises(ValidationError, match="exactly one"):
        evaluate_cell_gate(reference_se_raw=None, reference_radius_raw=None, **common)
    with pytest.raises(ValidationError, match="radius_budget_fraction"):
        evaluate_cell_gate(reference_se_raw=None, reference_radius_raw=1e-9,
                           **{**common, "bounds": GateBounds(cell=1.0, mean_signed_bias=0.2)})


def test_deterministic_radii_add_linearly_in_the_aggregate_because_they_may_share_a_sign():
    # four cells, each radius 0.04: independent noise would give 0.02, a shared discretization bias gives 0.04
    aggregate = evaluate_aggregate_gate([0.1, 0.1, 0.1, 0.1], None, GATES, radii_c=[0.04] * 4)
    assert aggregate.radius_of_mean_c == pytest.approx(0.04) and aggregate.se_of_mean_c is None
    assert aggregate.within_bound and aggregate.se_adequate and aggregate.passed          # 0.1 + 0.04 <= 0.2; 0.04 <= 0.05
    wide = evaluate_aggregate_gate([0.1] * 4, None, GATES, radii_c=[0.06] * 4)
    assert not wide.se_adequate and not wide.passed


# --- study validation ----------------------------------------------------------------------------------------------
def test_a_deterministic_reference_must_declare_its_radius_fraction_and_a_stochastic_one_must_not():
    with pytest.raises(ValidationError, match="radius_budget_fraction"):
        make_study(bounds=GateBounds(cell=1.0, mean_signed_bias=0.2))
    with pytest.raises(ValidationError, match="radius_budget_fraction"):
        make_study(reference=StochasticReference())
    with pytest.raises(ValidationError, match="schema"):
        make_study(schema=1, quantity_bounds={}, cases=(CaseSpec(name="a"),), quantities=("price", "delta", "gamma"),
                   sampling=dataclasses.replace(SAMPLING, seed_scheme="sequential"))


def test_a_qualifier_needs_a_deterministic_primary_and_a_stochastic_arm():
    with pytest.raises(ValidationError, match="deterministic"):
        make_study(reference=StochasticReference(), bounds=GateBounds(cell=1.0, mean_signed_bias=0.2),
                   qualification=ReferenceQualification(builder=StochasticReference(), max_z=4.0))
    with pytest.raises(ValidationError, match="stochastic"):
        make_study(qualification=ReferenceQualification(builder=Deterministic(), max_z=4.0))
    with pytest.raises(ValidationError, match="max_z"):
        ReferenceQualification(builder=StochasticReference(), max_z=0.0)


# --- pipeline ------------------------------------------------------------------------------------------------------
def test_a_matching_candidate_is_admitted_and_the_record_is_typed_never_a_zero_standard_error(tmp_path):
    payload = certify(make_study(), out_dir=tmp_path).payload
    assert payload["decisions"] == {"fake.cand": "ADMITTED"}
    block = payload["references"]["ordinary"]
    assert block["kind"] == "deterministic" and block["radii"]["pv"] == 1e-9 and block["evidence"] == {"levels": [101, 201, 401, 801]}
    assert not {"std_errors", "batches", "seeds"} & set(block)
    cell = _cell(payload, "ordinary", "pv")
    assert cell["reference"] == {"kind": "deterministic", "value": 0.5, "radius": 1e-9}
    assert cell["gate"]["se_c"] is None and cell["gate"]["radius_c"] == pytest.approx(1e-5)
    point = _cell(payload, "ordinary", "point_delta")                       # the deterministic reference targets it
    assert point["kind"] == "numeric" and point["verdict"] == "PASS" and payload["study"]["uncertified_quantities"] == []
    assert payload["contract"]["reference_kind"] == "deterministic"
    assert payload["contract"]["reference_error_model"] == {"method": "fake ladder", "radius": "declared"}
    validate_payload(payload)
    assert "deterministic" in render_markdown(payload) and "radius" in render_html(payload)


def test_an_undefined_reference_value_serves_a_semantic_cell_and_errors_a_numeric_one(tmp_path):
    payload = certify(make_study(), out_dir=tmp_path / "a").payload
    semantic = _cell(payload, "on_barrier", "point_delta")
    assert semantic["kind"] == "semantic" and semantic["verdict"] == "PASS" and semantic["reference"]["value"] is None
    assert payload["references"]["on_barrier"]["undefined"] == {"point_delta": "valuation is on the event"}
    undeclared = (CASES[0], CaseSpec(name="on_barrier", context_params={"valuation": "t1"}))
    payload = certify(make_study(cases=undeclared, candidates=(Candidate(),)), out_dir=tmp_path / "b").payload
    cell = _cell(payload, "on_barrier", "point_delta")
    assert cell["verdict"] == "ERROR" and "reference is undefined" in cell["error"]


def test_a_wide_or_unbounded_radius_is_unresolved(tmp_path):
    wide = certify(make_study(reference=Deterministic(radii={"pv": 0.3 * PV_BUDGET_RAW}), cases=CASES[:1]), out_dir=tmp_path / "a").payload
    assert _cell(wide, "ordinary", "pv")["verdict"] == "UNRESOLVED" and wide["decisions"] == {"fake.cand": "INCONCLUSIVE"}
    unbounded = certify(make_study(reference=Deterministic(radii={"pv": math.inf}), cases=CASES[:1]), out_dir=tmp_path / "b").payload
    cell = _cell(unbounded, "ordinary", "pv")
    assert cell["verdict"] == "UNRESOLVED" and cell["gate"] is None and "could not bound" in cell["reason"]


def test_a_candidate_beyond_the_budget_by_more_than_the_radius_is_rejected(tmp_path):
    study = make_study(candidates=(Candidate(statuses=HONEST, offsets={"pv": 1.5 * PV_BUDGET_RAW}),))
    assert certify(study, out_dir=tmp_path).payload["decisions"] == {"fake.cand": "REJECTED"}


def test_resume_reuses_a_solved_case_and_quick_is_bound_into_the_identity(tmp_path):
    reference = Deterministic()
    certify(make_study(reference=reference), out_dir=tmp_path)
    certify(make_study(reference=reference), out_dir=tmp_path, resume=True)
    assert reference.solves == ["ordinary", "on_barrier"]
    certify(make_study(reference=reference), out_dir=tmp_path, resume=True, quick=True)
    assert reference.solves == ["ordinary", "on_barrier", "ordinary", "on_barrier"]


# --- qualification -------------------------------------------------------------------------------------------------
def test_an_agreeing_qualifier_is_recorded_and_changes_nothing(tmp_path):
    qualifier = StochasticReference(jitter={q: 1e-7 for q in QUANTITIES})
    study = make_study(qualification=ReferenceQualification(builder=qualifier, max_z=4.0))
    payload = certify(study, out_dir=tmp_path).payload
    block = payload["qualification"]["ordinary"]
    assert block["qualified"] and set(block["checks"]) == {"pv", "desk_delta"}          # the qualifier targets no point greek
    assert block["checks"]["pv"]["within"] and block["batches"] == 2
    assert payload["decisions"] == {"fake.cand": "ADMITTED"}
    assert payload["contract"]["qualification"] == {"max_z": 4.0, "targets": qualifier.targets() | {"pv": {"estimator": "replicate_mean"}}}
    assert "qualif" in render_markdown(payload).lower() and "qualif" in render_html(payload).lower()


def test_a_disagreeing_qualifier_leaves_the_case_unresolved_whatever_the_candidate_says(tmp_path):
    # the deterministic value sits 50 qualifier-SEs away: nobody knows which arm is wrong, so nothing is admitted
    qualifier = StochasticReference(jitter={q: 1e-7 for q in QUANTITIES})
    study = make_study(reference=Deterministic(shift={"pv": 5e-6}), cases=CASES[:1],
                       candidates=(Candidate(offsets={"pv": 5e-6}),),
                       qualification=ReferenceQualification(builder=qualifier, max_z=4.0))
    payload = certify(study, out_dir=tmp_path).payload
    assert not payload["qualification"]["ordinary"]["qualified"]
    assert not payload["qualification"]["ordinary"]["checks"]["pv"]["within"]
    cell = _cell(payload, "ordinary", "pv")
    assert cell["verdict"] == "UNRESOLVED" and "not qualified" in cell["reason"]
    assert payload["decisions"] == {"fake.cand": "INCONCLUSIVE"}


# --- amendments ----------------------------------------------------------------------------------------------------
def test_an_amendment_carries_a_qualified_deterministic_reference_and_re_gates_a_changed_candidate(tmp_path):
    from quantark.modelvalidation.amendment import amend

    qualifier = StochasticReference(jitter={q: 1e-7 for q in QUANTITIES})
    reference = Deterministic()
    study = make_study(reference=reference, qualification=ReferenceQualification(builder=qualifier, max_z=4.0))
    parent = certify(study, out_dir=tmp_path / "parent")
    sampled = len(StochasticReference.paths_seen)
    changed = make_study(reference=reference, qualification=ReferenceQualification(builder=qualifier, max_z=4.0),
                         candidates=(Candidate(statuses=HONEST, offsets={"pv": 0.1 * PV_BUDGET_RAW}),))
    amended = amend(changed, parent.path, tmp_path / "amended", reason="candidate retuned").payload
    assert reference.solves == ["ordinary", "on_barrier"]                       # the solve was carried, never repeated
    assert len(StochasticReference.paths_seen) == sampled                       # and so was the qualifying arm
    assert amended["qualification"] == parent.payload["qualification"]
    cell = _cell(amended, "ordinary", "pv")
    assert cell["verdict"] == "PASS" and cell["gate"]["radius_c"] == pytest.approx(1e-5) and "carried_from" not in cell
    assert amended["decisions"] == {"fake.cand": "ADMITTED"}


def test_an_amendment_may_not_change_the_error_model_or_the_qualification_policy(tmp_path):
    from quantark.modelvalidation.amendment import amend

    qualifier = StochasticReference(jitter={q: 1e-7 for q in QUANTITIES})
    parent = certify(make_study(qualification=ReferenceQualification(builder=qualifier, max_z=4.0)), out_dir=tmp_path / "parent")
    looser = make_study(qualification=ReferenceQualification(builder=qualifier, max_z=6.0))
    with pytest.raises(ValidationError, match="qualification"):
        amend(looser, parent.path, tmp_path / "amended", reason="looser qualification")
