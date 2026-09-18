"""End-to-end schema-2 certification over deterministic fakes."""
import dataclasses

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.amendment import amend
from quantark.modelvalidation.anchors import assert_anchors, extract_anchors
from quantark.modelvalidation.candidate import CandidateResult, ConvergenceAxis, ConvergenceLevel
from quantark.modelvalidation.evidence import atomic_write_json
from quantark.modelvalidation.html_report import render_html
from quantark.modelvalidation.pipeline import certify, validate_payload
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.report import render_markdown
from quantark.modelvalidation.study import (
    CaseSpec, CertificationStudy, GateBounds, NormalizedScale, QuantityBounds, SamplingPolicy, batch_seed,
)

QUANTITIES = ("pv", "desk_delta", "point_delta")
BOUNDS = {"pv": QuantityBounds(1e-6), "desk_delta": QuantityBounds(1e-5, 1e-4), "point_delta": QuantityBounds(1e-5, 1e-4)}
TRUTH = {"pv": 0.5, "desk_delta": 0.3, "point_delta": 0.3}
SAMPLING = SamplingPolicy(paths_per_batch=4096, min_batches=2, max_batches=2, seed=7, seed_scheme="substream")
# pv normalizes as raw / notional (100) against a 1e-6 budget, so 1e-4 raw is one pv budget
PV_BUDGET_RAW = 1e-4


class Reference:
    """Tight replicate means on substream seeds; point_delta is recorded but declared untargeted."""

    paths_seen: list = []

    def __init__(self, sampling=SAMPLING, jitter=None, targets=None):
        self.sampling, self.jitter = sampling, jitter or {}
        self._targets = targets or {"desk_delta": {"estimator": "paired_central_difference", "bump": 0.01}, "point_delta": None}

    def bind(self, policy):
        return Reference(policy, self.jitter, self._targets)

    def targets(self):
        return dict(self._targets)

    def identity(self, case):
        return {"builder": "fake.ref", "case": case.name, "context": dict(case.context_params), "jitter": self.jitter,
                "targets": self.targets(), "paths": self.sampling.paths_per_batch}

    def run_batch(self, case, batch_index):
        Reference.paths_seen.append(self.sampling.paths_per_batch)
        sign = 1.0 if batch_index % 2 else -1.0
        values = {q: TRUTH[q] + sign * self.jitter.get(q, 1e-12) for q in QUANTITIES}
        return BatchResult(index=batch_index, seed=batch_seed(self.sampling, case.name, batch_index), values=values)


def _axes(values, envelope=0.0, levels=3):
    return (ConvergenceAxis("grid", "refinement", tuple(
        ConvergenceLevel(f"x{2 ** i}", float(2 ** i), {q: v + (envelope if i else 0.0) for q, v in values.items()},
                         {"points": 100 * 2 ** i}, is_target=(i == 0)) for i in range(levels))),)


class Candidate:
    def __init__(self, name="fake.cand", offsets=None, statuses=None, envelope=0.0, levels=3, vol=0.20):
        self._name, self.offsets, self.statuses = name, offsets or {}, statuses or {}
        self.envelope, self.levels, self.vol, self.calls, self.target_calls = envelope, levels, vol, [], 0

    def name(self):
        return self._name

    def params(self):
        return {"offsets": self.offsets}

    def resolved_inputs(self, case):
        return {"environment": {"vol": self.vol, **case.environment_params}, "context": dict(case.context_params),
                "quantities": list(QUANTITIES)}

    def fingerprint(self):
        return "fp-1"

    def _values(self, case):
        statuses = dict(self.statuses.get(case.name, {}))
        return {q: TRUTH[q] + self.offsets.get(q, 0.0) for q in QUANTITIES if statuses.get(q, "ok") == "ok"}, statuses

    def evaluate(self, case):
        self.calls.append(case.name)
        values, statuses = self._values(case)
        return CandidateResult(values=values, statuses=statuses, reasons={q: "no derivative here" for q in statuses},
                               convergence=_axes(values, self.envelope, self.levels))

    def evaluate_target(self, case):
        self.target_calls += 1
        values, statuses = self._values(case)
        return CandidateResult(values=values, statuses=statuses, exact=True)


CASES = (CaseSpec(name="ordinary", context_params={"valuation": "t0"}),
         CaseSpec(name="on_barrier", context_params={"valuation": "t1"}, expected={"point_delta": "undefined"}))
HONEST = {"on_barrier": {"point_delta": "undefined"}}


def make_study(**overrides):
    kwargs = dict(
        name="fake-s2", schema=2, quantities=QUANTITIES, quantity_bounds=BOUNDS, cases=CASES,
        bounds=GateBounds(cell=1.0, mean_signed_bias=0.2), scale=NormalizedScale(100.0, 100.0),
        reference=Reference(), candidates=(Candidate(statuses=HONEST),), sampling=SAMPLING,
        source_text="study: fake-s2\n",
    )
    kwargs.update(overrides)
    return CertificationStudy(**kwargs)


def _cell(payload, case, quantity, candidate="fake.cand"):
    return next(c for c in payload["cells"] if (c["candidate"], c["case"], c["quantity"]) == (candidate, case, quantity))


# --- decisions -----------------------------------------------------------------------------------------------------
def test_matching_candidate_is_admitted_and_point_cells_are_uncertified_not_blocking(tmp_path):
    payload = certify(make_study(), out_dir=tmp_path).payload
    assert payload["schema"] == 2 and payload["decisions"] == {"fake.cand": "ADMITTED"}
    pv = _cell(payload, "ordinary", "pv")
    assert pv["kind"] == "numeric" and pv["verdict"] == "PASS" and pv["bound_c"] == 1e-6
    assert pv["convergence"]["complete"] and pv["convergence"]["observed_orders"] == {"grid": None}
    point = _cell(payload, "ordinary", "point_delta")
    assert point["kind"] == "untargeted" and point["verdict"] == "UNRESOLVED" and point["gate"] is None
    assert "no estimator" in point["reason"] and point["candidate_value"] == pytest.approx(0.3)
    assert payload["study"]["uncertified_quantities"] == ["point_delta"]
    semantic = _cell(payload, "on_barrier", "point_delta")
    assert semantic["kind"] == "semantic" and semantic["verdict"] == "PASS"


def test_admitted_requires_every_semantic_assertion_to_hold(tmp_path):
    payload = certify(make_study(candidates=(Candidate(),)), out_dir=tmp_path).payload    # reports a number where undefined was expected
    semantic = _cell(payload, "on_barrier", "point_delta")
    assert semantic["verdict"] == "FAIL" and "expected undefined" in semantic["reason"]
    assert payload["decisions"] == {"fake.cand": "REJECTED"}


def test_unexpected_undefined_in_a_numeric_cell_is_an_error_that_blocks_admission(tmp_path):
    study = make_study(candidates=(Candidate(statuses={"ordinary": {"desk_delta": "undefined"}}),), cases=CASES[:1])
    payload = certify(study, out_dir=tmp_path).payload
    cell = _cell(payload, "ordinary", "desk_delta")
    assert cell["verdict"] == "ERROR" and "unexpected undefined" in cell["error"]
    assert payload["decisions"] == {"fake.cand": "INCONCLUSIVE"}


def test_relative_budget_uses_the_reference_magnitude(tmp_path):
    # desk_delta budget = max(1e-5, 1e-4 * |0.3 * S/N|) = 3e-5 normalized; an offset of 2e-5 passes, 4e-5 fails
    ok = certify(make_study(candidates=(Candidate(offsets={"desk_delta": 2e-5}),), cases=CASES[:1]), out_dir=tmp_path / "a").payload
    assert _cell(ok, "ordinary", "desk_delta")["bound_c"] == pytest.approx(3e-5)
    assert _cell(ok, "ordinary", "desk_delta")["verdict"] == "PASS"
    bad = certify(make_study(candidates=(Candidate(offsets={"desk_delta": 4e-5}),), cases=CASES[:1]), out_dir=tmp_path / "b").payload
    assert _cell(bad, "ordinary", "desk_delta")["verdict"] == "FAIL" and bad["decisions"] == {"fake.cand": "REJECTED"}


def test_a_cell_straddle_is_unresolved_and_inconclusive(tmp_path):
    # SE 0.2 budgets and error 0.9 budgets on pv: upper 1.3, lower 0.5
    study = make_study(reference=Reference(jitter={"pv": 0.2 * PV_BUDGET_RAW}),
                       candidates=(Candidate(offsets={"pv": 0.9 * PV_BUDGET_RAW}),), cases=CASES[:1])
    payload = certify(study, out_dir=tmp_path).payload
    assert _cell(payload, "ordinary", "pv")["verdict"] == "UNRESOLVED"
    assert payload["decisions"] == {"fake.cand": "INCONCLUSIVE"}


def test_an_aggregate_overlap_with_adequate_precision_is_inconclusive_not_rejected(tmp_path):
    # Review R4. Every cell passes on its own (0.195 + 2 * 0.005 = 0.205 of its budget), but the mean signed
    # error 0.195 +/- 2 * 0.0035 = [0.188, 0.202] overlaps the 0.2 aggregate bound with an adequate SE.
    cases = (CaseSpec(name="a"), CaseSpec(name="b"))
    study = make_study(cases=cases, reference=Reference(jitter={"pv": 0.005 * PV_BUDGET_RAW}),
                       candidates=(Candidate(offsets={"pv": 0.195 * PV_BUDGET_RAW}),))
    payload = certify(study, out_dir=tmp_path).payload
    assert all(c["verdict"] == "PASS" for c in payload["cells"] if c["quantity"] == "pv")
    aggregate = next(a for a in payload["aggregates"] if a["quantity"] == "pv")
    assert aggregate["se_adequate"] and not aggregate["within_bound"] and not aggregate["lower_exceeds_bound"]
    assert payload["decisions"] == {"fake.cand": "INCONCLUSIVE"}


def test_missing_convergence_evidence_is_unresolved_never_acceptable(tmp_path):
    # Review R5: two levels cannot show an order, so the envelope requirement is unmet, not waived
    payload = certify(make_study(candidates=(Candidate(statuses=HONEST, levels=2),)), out_dir=tmp_path).payload
    cell = _cell(payload, "ordinary", "pv")
    assert cell["verdict"] == "UNRESOLVED" and "grid" in cell["reason"] and not cell["convergence"]["complete"]
    assert payload["decisions"] == {"fake.cand": "INCONCLUSIVE"}


def test_an_envelope_beyond_its_allocation_fails(tmp_path):
    payload = certify(make_study(candidates=(Candidate(statuses=HONEST, envelope=0.6 * PV_BUDGET_RAW),), cases=CASES[:1]),
                      out_dir=tmp_path).payload
    assert _cell(payload, "ordinary", "pv")["verdict"] == "FAIL"


# --- identities and resume (review R1) ----------------------------------------------------------------------------
def test_resume_re_evaluates_a_case_whose_context_changed_under_the_same_name(tmp_path):
    candidate = Candidate(statuses=HONEST)
    certify(make_study(candidates=(candidate,)), out_dir=tmp_path)
    assert candidate.calls == ["ordinary", "on_barrier"]
    moved = (CaseSpec(name="ordinary", context_params={"valuation": "t0-later"}), CASES[1])
    certify(make_study(candidates=(candidate,), cases=moved), out_dir=tmp_path, resume=True)
    assert candidate.calls == ["ordinary", "on_barrier", "ordinary"]          # only the moved case ran again


def test_resume_re_evaluates_every_case_after_a_study_level_market_change(tmp_path):
    first = Candidate(statuses=HONEST, vol=0.20)
    certify(make_study(candidates=(first,)), out_dir=tmp_path)
    second = Candidate(statuses=HONEST, vol=0.35)
    certify(make_study(candidates=(second,)), out_dir=tmp_path, resume=True)
    assert second.calls == ["ordinary", "on_barrier"]


# --- the effective sampling policy is the one that runs (review R10) ----------------------------------------------
def test_quick_mode_runs_the_path_count_it_records(tmp_path):
    Reference.paths_seen.clear()
    payload = certify(make_study(), out_dir=tmp_path, quick=True).payload
    declared = payload["study"]["sampling"]["paths_per_batch"]
    assert declared == 512 and set(Reference.paths_seen) == {declared}


def test_stop_reason_is_truthful_under_a_frozen_budget(tmp_path):
    noisy_target = certify(make_study(reference=Reference(jitter={"pv": 0.4 * PV_BUDGET_RAW}), cases=CASES[:1]),
                           out_dir=tmp_path / "a").payload
    assert noisy_target["references"]["ordinary"]["stopped_reason"] == "max_batches"
    noisy_untargeted = certify(make_study(reference=Reference(jitter={"point_delta": 1.0}), cases=CASES[:1]),
                               out_dir=tmp_path / "b").payload
    assert noisy_untargeted["references"]["ordinary"]["stopped_reason"] == "se_budget_met"


# --- payload, reports, anchors -------------------------------------------------------------------------------------
def test_payload_records_the_schema_2_declarations_and_validates(tmp_path):
    payload = certify(make_study(), out_dir=tmp_path).payload
    validate_payload(payload)
    block = payload["study"]
    assert block["quantity_bounds"]["desk_delta"] == {"abs_floor": 1e-5, "rel": 1e-4}
    assert block["quantity_definitions"]["point_delta"]["convention"] == "point"
    assert block["scale"] == {"notional": 100.0, "spot_scale": 100.0, "point_move": 0.01}
    assert block["sampling"]["seed_scheme"] == "substream"
    assert payload["reference_targets"]["point_delta"] is None
    assert payload["contract"]["gate_policy"]["interval_k"] == 2.0 and len(payload["contract_sha256"]) == 64
    cases = {c["name"]: c for c in block["cases"]}
    assert cases["on_barrier"]["expected"] == {"point_delta": "undefined"}
    assert cases["ordinary"]["context_params"] == {"valuation": "t0"}
    markdown = render_markdown(payload)
    assert "fraction of each cell's own budget" in markdown and "Uncertified quantities" in markdown
    assert "on_barrier" in render_html(payload)


def test_anchors_skip_semantic_cells_only_and_replay_the_target_without_the_ladders(tmp_path, monkeypatch):
    study = make_study()
    payload = certify(study, out_dir=tmp_path).payload
    extracted = extract_anchors(payload, study)
    by_case = {a["case"]: a["values"] for a in extracted["anchors"]}
    assert "point_delta" in by_case["ordinary"] and "point_delta" not in by_case["on_barrier"]
    assert "pv" in by_case["on_barrier"]
    path = tmp_path / "anchors.json"
    atomic_write_json(path, extracted)
    candidate = study.candidates[0]
    monkeypatch.setattr("quantark.modelvalidation.anchors.load_study_text", lambda text: study)
    calls_before = len(candidate.calls)
    assert_anchors(path)
    assert candidate.target_calls == 2 and len(candidate.calls) == calls_before     # evaluate() never ran


# --- the amendment contract (review R9) ---------------------------------------------------------------------------
@pytest.mark.parametrize("change, match", [
    (dict(quantity_bounds={**BOUNDS, "pv": QuantityBounds(1e-5)}), "quantity_bounds"),
    (dict(bounds=GateBounds(cell=1.0, mean_signed_bias=0.2, interval_k=3.0)), "gate_policy"),
    (dict(bounds=GateBounds(cell=1.0, mean_signed_bias=0.2, envelope_fraction=0.9)), "gate_policy"),
    (dict(scale=NormalizedScale(100.0, 75.0)), "scale"),
    (dict(reference=Reference(targets={"desk_delta": {"estimator": "paired_central_difference", "bump": 0.01},
                                       "point_delta": {"estimator": "bump_limit"}})), "reference_targets"),
])
def test_an_amendment_refuses_a_changed_contract(tmp_path, change, match):
    parent = certify(make_study(), out_dir=tmp_path / "parent")
    with pytest.raises(ValidationError, match=match):
        amend(make_study(**change), parent=parent.path, out_dir=tmp_path / "amend", reason="contract change")


def test_an_amendment_refuses_a_schema_mismatch(tmp_path):
    parent = certify(make_study(), out_dir=tmp_path / "parent")
    schema1 = dataclasses.replace(make_study(), schema=1, quantities=("pv",), quantity_bounds={}, cases=(CaseSpec(name="ordinary"),),
                                  sampling=dataclasses.replace(SAMPLING, seed_scheme="sequential"))
    with pytest.raises(ValidationError, match="schema"):
        amend(schema1, parent=parent.path, out_dir=tmp_path / "amend", reason="schema mismatch")


def test_adding_an_expectation_re_evaluates_exactly_that_case(tmp_path):
    parent = certify(make_study(cases=(CaseSpec(name="ordinary", context_params={"valuation": "t0"}),
                                       CaseSpec(name="on_barrier", context_params={"valuation": "t1"}))),
                     out_dir=tmp_path / "parent")
    amended = amend(make_study(), parent=parent.path, out_dir=tmp_path / "amend", reason="declare the undefined derivative")
    info = amended.payload["amendment"]
    assert {c["case"] for c in info["replaced_cells"]} == {"on_barrier"}
    assert {c["case"] for c in info["carried_cells"]} == {"ordinary"}
    assert _cell(amended.payload, "on_barrier", "point_delta")["kind"] == "semantic"
