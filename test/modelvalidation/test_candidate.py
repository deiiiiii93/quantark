"""Tests for the candidate arm: ladders, envelopes, identity."""

import math

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.candidate import (
    CandidateResult,
    LadderRung,
    candidate_identity,
    envelope_from_ladders,
)
from quantark.modelvalidation.study import CaseSpec


def _rung(axis, level, delta):
    return LadderRung(axis=axis, level=level, values={"delta": delta})


def test_envelope_sums_per_axis_increments():
    ladders = [
        _rung("n_x", "target", 1.000),
        _rung("n_x", "medium", 1.004),
        _rung("n_t", "target", 1.000),
        _rung("n_t", "medium", 0.998),
    ]
    assert envelope_from_ladders(ladders, "delta") == pytest.approx(0.006)


def test_envelope_single_axis():
    ladders = [_rung("n_x", "target", 2.0), _rung("n_x", "medium", 2.5)]
    assert envelope_from_ladders(ladders, "delta") == pytest.approx(0.5)


def test_envelope_is_none_without_a_medium_rung():
    ladders = [_rung("n_x", "target", 1.0)]
    assert envelope_from_ladders(ladders, "delta") is None


def test_envelope_is_none_without_a_target_rung():
    ladders = [_rung("n_x", "medium", 1.0)]
    assert envelope_from_ladders(ladders, "delta") is None


def test_envelope_is_none_for_empty_ladders():
    assert envelope_from_ladders([], "delta") is None


def test_envelope_ignores_axes_missing_the_quantity():
    ladders = [
        LadderRung(axis="n_x", level="target", values={"delta": 1.0}),
        LadderRung(axis="n_x", level="medium", values={"delta": 1.1}),
        LadderRung(axis="n_t", level="target", values={"gamma": 5.0}),
        LadderRung(axis="n_t", level="medium", values={"gamma": 6.0}),
    ]
    assert envelope_from_ladders(ladders, "delta") == pytest.approx(0.1)


def test_envelope_is_zero_for_an_exact_engine():
    """An analytically exact candidate has no discretization error to bound."""
    ladders = [_rung("analytic", "target", 3.0), _rung("analytic", "medium", 3.0)]
    assert envelope_from_ladders(ladders, "delta") == pytest.approx(0.0)


def test_ladder_rung_rejects_unknown_level():
    with pytest.raises(ValidationError):
        LadderRung(axis="n_x", level="ultra", values={"delta": 1.0})


def test_ladder_rung_rejects_empty_axis():
    with pytest.raises(ValidationError):
        LadderRung(axis="", level="target", values={"delta": 1.0})


def test_envelope_rejects_non_finite_values():
    ladders = [
        _rung("n_x", "target", math.nan),
        _rung("n_x", "medium", 1.0),
    ]
    with pytest.raises(ValidationError):
        envelope_from_ladders(ladders, "delta")


def test_candidate_result_requires_values():
    with pytest.raises(ValidationError):
        CandidateResult(values={}, ladders=())


class FakeCandidate:
    def name(self):
        return "fake.engine"

    def params(self):
        return {"n_x": 300, "accuracy": "standard"}

    def evaluate(self, case):
        return CandidateResult(values={"delta": 1.0}, ladders=())


def test_candidate_identity_covers_name_params_and_case():
    case = CaseSpec(name="near_ko", environment_params={"spot": 102.5})
    identity = candidate_identity(FakeCandidate(), case)
    assert identity["candidate"] == "fake.engine"
    assert identity["params"] == {"n_x": 300, "accuracy": "standard"}
    assert identity["case"]["name"] == "near_ko"
    assert identity["case"]["environment_params"] == {"spot": 102.5}


def test_candidate_identity_changes_with_params():
    class Other(FakeCandidate):
        def params(self):
            return {"n_x": 600, "accuracy": "standard"}

    case = CaseSpec(name="ordinary")
    assert candidate_identity(FakeCandidate(), case) != candidate_identity(Other(), case)
