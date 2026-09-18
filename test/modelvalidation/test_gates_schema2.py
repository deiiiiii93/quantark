"""Schema-2 gate arithmetic: budget units, lower/upper edges on cells and aggregates, wire projections."""
import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.decisions import Decision, Verdict, decide_candidate, decide_cell
from quantark.modelvalidation.gates import (
    aggregate_gate_wire,
    cell_gate_wire,
    evaluate_aggregate_gate,
    evaluate_cell_gate,
)
from quantark.modelvalidation.study import GateBounds, NormalizedScale

SCALE = NormalizedScale(notional=100.0, spot_scale=100.0)
BOUNDS = GateBounds(cell=1.0, mean_signed_bias=0.2, se_budget_fraction=0.25, interval_k=2.0, envelope_fraction=0.5)


def _gate(err_n, se_n, bound_n=1e-5, envelope_n=None):
    """A point_delta cell with error, SE and envelope given as fractions of the budget."""
    # to_economic(point_delta, raw) = raw * S / N = raw here (S == N), so raw == normalized
    return evaluate_cell_gate(
        candidate_raw=err_n * bound_n, reference_raw=0.0, reference_se_raw=se_n * bound_n,
        quantity="point_delta", scale=SCALE, bounds=BOUNDS, bound_c=bound_n,
        envelope_raw=None if envelope_n is None else envelope_n * bound_n,
    )


def test_values_are_normalized_by_the_cell_budget():
    gate = _gate(err_n=0.4, se_n=0.1)
    assert gate.bound_c == 1e-5
    assert gate.signed_err_c == pytest.approx(0.4) and gate.se_c == pytest.approx(0.1)
    assert gate.interval_c == pytest.approx(0.6)          # upper edge: 0.4 + 2 * 0.1
    assert gate.lower_error_c == pytest.approx(0.2)       # lower edge: 0.4 - 2 * 0.1
    assert gate.se_budget_met and gate.interval_within_bound and not gate.lower_exceeds_bound
    assert decide_cell(gate, error=False, schema=2) is Verdict.PASS


def test_lower_edge_clamps_at_zero():
    assert _gate(err_n=0.1, se_n=0.2).lower_error_c == 0.0


def test_a_cell_straddle_is_unresolved_not_fail():
    gate = _gate(err_n=0.9, se_n=0.1)                     # upper 1.1 > 1, lower 0.7 <= 1
    assert gate.se_budget_met and not gate.interval_within_bound and not gate.lower_exceeds_bound
    assert decide_cell(gate, error=False, schema=2) is Verdict.UNRESOLVED
    assert decide_cell(gate, error=False, schema=1) is Verdict.FAIL   # schema 1 keeps its rule


def test_a_cell_whose_lower_edge_is_beyond_the_budget_fails():
    gate = _gate(err_n=1.5, se_n=0.1)                     # lower 1.3 > 1
    assert gate.lower_exceeds_bound and decide_cell(gate, error=False, schema=2) is Verdict.FAIL


def test_a_noisy_reference_is_unresolved_whatever_the_error():
    gate = _gate(err_n=0.0, se_n=0.3)                     # SE 0.3 > 0.25 of the budget
    assert not gate.se_budget_met and decide_cell(gate, error=False, schema=2) is Verdict.UNRESOLVED


def test_an_envelope_violation_fails_as_a_convergence_requirement():
    gate = _gate(err_n=0.1, se_n=0.05, envelope_n=0.6)    # envelope 0.6 > 0.5 of the budget
    assert not gate.envelope_within_bound and decide_cell(gate, error=False, schema=2) is Verdict.FAIL


def test_bound_must_be_positive_and_finite():
    with pytest.raises(ValidationError):
        _gate(err_n=0.1, se_n=0.1, bound_n=0.0)


# --- the aggregate gate follows the same three-way rule (review R4) ----------------------------------------------
def test_an_aggregate_straddle_is_inconclusive_in_schema_2_and_still_rejected_in_schema_1():
    # mean signed error 0.19, SE 0.01, k = 2, bound 0.20: interval [0.17, 0.21] overlaps the bound
    aggregate = evaluate_aggregate_gate([0.19], [0.01], BOUNDS)
    assert aggregate.se_adequate and not aggregate.within_bound
    assert aggregate.lower_c == pytest.approx(0.17) and not aggregate.lower_exceeds_bound
    assert decide_candidate([Verdict.PASS], [aggregate], schema=2) is Decision.INCONCLUSIVE
    assert decide_candidate([Verdict.PASS], [aggregate], schema=1) is Decision.REJECTED


def test_an_aggregate_whose_lower_edge_is_beyond_the_bound_rejects():
    aggregate = evaluate_aggregate_gate([0.25], [0.01], BOUNDS)     # interval [0.23, 0.27]
    assert aggregate.lower_exceeds_bound
    assert decide_candidate([Verdict.PASS], [aggregate], schema=2) is Decision.REJECTED


def test_an_aggregate_inside_the_bound_admits():
    aggregate = evaluate_aggregate_gate([0.05, -0.02], [0.01, 0.01], BOUNDS)
    assert aggregate.passed and decide_candidate([Verdict.PASS, Verdict.PASS], [aggregate], schema=2) is Decision.ADMITTED


def test_unknown_schema_is_rejected():
    with pytest.raises(ValidationError, match="schema"):
        decide_cell(_gate(0.1, 0.1), error=False, schema=3)
    with pytest.raises(ValidationError, match="schema"):
        decide_candidate([Verdict.PASS], [], schema=3)


# --- wire projections (review R2) ---------------------------------------------------------------------------------
def test_schema_1_wire_drops_every_field_the_old_format_did_not_have():
    gate = evaluate_cell_gate(candidate_raw=0.2, reference_raw=0.0, reference_se_raw=0.05, quantity="point_delta",
                              scale=SCALE, bounds=BOUNDS)
    assert gate.bound_c is None and gate.interval_c == pytest.approx(0.3)
    assert sorted(cell_gate_wire(gate, 1)) == [
        "envelope_c", "envelope_within_bound", "interval_c", "interval_within_bound", "passed", "se_budget_met",
        "se_c", "signed_err_c"]
    assert {"lower_error_c", "lower_exceeds_bound", "bound_c"} <= set(cell_gate_wire(gate, 2))
    aggregate = evaluate_aggregate_gate([0.05], [0.01], BOUNDS)
    assert sorted(aggregate_gate_wire(aggregate, 1)) == [
        "cells", "mean_signed_bias_c", "passed", "se_adequate", "se_of_mean_c", "within_bound"]
    assert {"lower_c", "lower_exceeds_bound"} <= set(aggregate_gate_wire(aggregate, 2))
