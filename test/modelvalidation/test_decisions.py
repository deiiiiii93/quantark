"""Tests for the verdict lattice and candidate decisions."""

import pytest

from quantark.modelvalidation.decisions import (
    Decision,
    Verdict,
    decide_candidate,
    decide_cell,
)
from quantark.modelvalidation.gates import AggregateGateResult, CellGateResult


def _cell_gate(*, passed: bool, se_budget_met: bool = True) -> CellGateResult:
    return CellGateResult(
        signed_err_c=0.1,
        se_c=0.01,
        interval_c=0.12,
        se_budget_met=se_budget_met,
        interval_within_bound=passed,
        envelope_c=None,
        envelope_within_bound=True,
        passed=passed and se_budget_met,
    )


def _aggregate(*, passed: bool, se_adequate: bool = True) -> AggregateGateResult:
    return AggregateGateResult(
        mean_signed_bias_c=0.01,
        se_of_mean_c=0.001,
        within_bound=passed,
        se_adequate=se_adequate,
        passed=passed and se_adequate,
        cells=3,
    )


def test_enums_serialize_as_their_value():
    assert Verdict.PASS.value == "PASS"
    assert Decision.ADMITTED.value == "ADMITTED"
    # str-Enum so json.dumps writes the value, not "Verdict.PASS"
    assert f"{Verdict.PASS}" in ("PASS", "Verdict.PASS")
    assert Verdict("PASS") is Verdict.PASS


def test_decide_cell_error_wins():
    assert decide_cell(None, error=True) is Verdict.ERROR
    assert decide_cell(_cell_gate(passed=True), error=True) is Verdict.ERROR


def test_decide_cell_unresolved_when_benchmark_too_noisy():
    gate = _cell_gate(passed=True, se_budget_met=False)
    assert decide_cell(gate, error=False) is Verdict.UNRESOLVED


def test_decide_cell_pass_and_fail():
    assert decide_cell(_cell_gate(passed=True), error=False) is Verdict.PASS
    assert decide_cell(_cell_gate(passed=False), error=False) is Verdict.FAIL


def test_decide_cell_requires_a_gate_when_no_error():
    with pytest.raises(Exception):
        decide_cell(None, error=False)


def test_all_pass_admits():
    verdicts = [Verdict.PASS, Verdict.PASS, Verdict.PASS]
    aggregates = [_aggregate(passed=True)]
    assert decide_candidate(verdicts, aggregates) is Decision.ADMITTED


def test_confident_cell_failure_rejects():
    verdicts = [Verdict.PASS, Verdict.FAIL]
    aggregates = [_aggregate(passed=True)]
    assert decide_candidate(verdicts, aggregates) is Decision.REJECTED


def test_error_cell_is_inconclusive_not_admitted():
    verdicts = [Verdict.PASS, Verdict.ERROR]
    aggregates = [_aggregate(passed=True)]
    assert decide_candidate(verdicts, aggregates) is Decision.INCONCLUSIVE


def test_unresolved_cell_is_inconclusive():
    verdicts = [Verdict.PASS, Verdict.UNRESOLVED]
    aggregates = [_aggregate(passed=True)]
    assert decide_candidate(verdicts, aggregates) is Decision.INCONCLUSIVE


def test_confident_aggregate_failure_rejects():
    verdicts = [Verdict.PASS, Verdict.PASS]
    aggregates = [_aggregate(passed=False, se_adequate=True)]
    assert decide_candidate(verdicts, aggregates) is Decision.REJECTED


def test_aggregate_failure_with_inadequate_se_is_inconclusive():
    """A tilt we cannot resolve is not evidence of a defect."""
    verdicts = [Verdict.PASS, Verdict.PASS]
    aggregates = [_aggregate(passed=False, se_adequate=False)]
    assert decide_candidate(verdicts, aggregates) is Decision.INCONCLUSIVE


def test_confident_failure_dominates_an_error():
    verdicts = [Verdict.ERROR, Verdict.FAIL]
    aggregates = [_aggregate(passed=True)]
    assert decide_candidate(verdicts, aggregates) is Decision.REJECTED


def test_empty_verdicts_is_inconclusive():
    assert decide_candidate([], []) is Decision.INCONCLUSIVE
