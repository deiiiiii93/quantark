"""Tests for the pure gate arithmetic."""

import math

import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.gates import (
    evaluate_aggregate_gate,
    evaluate_cell_gate,
)
from quantark.modelvalidation.study import GateBounds, HedgeContractScale


def _scale() -> HedgeContractScale:
    return HedgeContractScale(200.0, 100.0, 50_000_000.0)


def _bounds(**overrides) -> GateBounds:
    kwargs = dict(cell=0.5, mean_signed_bias=0.1)
    kwargs.update(overrides)
    return GateBounds(**kwargs)


def test_cell_gate_pass():
    dq = _scale().delta_quantum
    # 0.2 contracts of error, SE 0.05 contracts -> interval 0.3 <= 0.5
    result = evaluate_cell_gate(
        candidate_raw=0.2 * dq,
        reference_raw=0.0,
        reference_se_raw=0.05 * dq,
        quantity="delta",
        scale=_scale(),
        bounds=_bounds(),
    )
    assert result.signed_err_c == pytest.approx(0.2)
    assert result.se_c == pytest.approx(0.05)
    assert result.interval_c == pytest.approx(0.3)
    assert result.se_budget_met
    assert result.interval_within_bound
    assert result.passed


def test_cell_gate_fails_when_interval_exceeds_bound():
    dq = _scale().delta_quantum
    result = evaluate_cell_gate(
        candidate_raw=0.6 * dq,
        reference_raw=0.0,
        reference_se_raw=0.01 * dq,
        quantity="delta",
        scale=_scale(),
        bounds=_bounds(),
    )
    assert not result.interval_within_bound
    assert not result.passed


def test_cell_gate_se_budget_blocks_pass():
    """A noisy benchmark cannot discriminate, so it cannot certify."""
    dq = _scale().delta_quantum
    result = evaluate_cell_gate(
        candidate_raw=0.0,
        reference_raw=0.0,
        reference_se_raw=0.2 * dq,  # 0.2 > 0.25 * 0.5
        quantity="delta",
        scale=_scale(),
        bounds=_bounds(),
    )
    assert not result.se_budget_met
    assert not result.passed


def test_cell_gate_envelope_violation_blocks_pass():
    dq = _scale().delta_quantum
    result = evaluate_cell_gate(
        candidate_raw=0.0,
        reference_raw=0.0,
        reference_se_raw=0.01 * dq,
        quantity="delta",
        scale=_scale(),
        bounds=_bounds(envelope_fraction=0.5),
        envelope_raw=0.3 * dq,  # 0.3 > 0.5 * 0.5
    )
    assert result.envelope_c == pytest.approx(0.3)
    assert not result.envelope_within_bound
    assert not result.passed


def test_cell_gate_without_envelope_is_permitted():
    dq = _scale().delta_quantum
    result = evaluate_cell_gate(
        candidate_raw=0.0,
        reference_raw=0.0,
        reference_se_raw=0.01 * dq,
        quantity="delta",
        scale=_scale(),
        bounds=_bounds(),
        envelope_raw=None,
    )
    assert result.envelope_c is None
    assert result.envelope_within_bound
    assert result.passed


def test_cell_gate_sign_is_candidate_minus_reference():
    dq = _scale().delta_quantum
    result = evaluate_cell_gate(
        candidate_raw=-0.3 * dq,
        reference_raw=0.0,
        reference_se_raw=0.01 * dq,
        quantity="delta",
        scale=_scale(),
        bounds=_bounds(),
    )
    assert result.signed_err_c == pytest.approx(-0.3)
    assert result.interval_c == pytest.approx(0.3 + 2.0 * 0.01)


def test_cell_gate_converts_gamma_through_the_scale():
    scale = _scale()
    raw_err = 1.0
    expected_c = scale.to_economic("gamma", raw_err)
    result = evaluate_cell_gate(
        candidate_raw=raw_err,
        reference_raw=0.0,
        reference_se_raw=0.0,
        quantity="gamma",
        scale=scale,
        bounds=_bounds(cell=expected_c * 2.0),
    )
    assert result.signed_err_c == pytest.approx(expected_c)


def test_cell_gate_rejects_negative_se():
    with pytest.raises(ValidationError):
        evaluate_cell_gate(
            candidate_raw=0.0,
            reference_raw=0.0,
            reference_se_raw=-1.0,
            quantity="delta",
            scale=_scale(),
            bounds=_bounds(),
        )


def test_cell_gate_rejects_non_finite_values():
    with pytest.raises(ValidationError):
        evaluate_cell_gate(
            candidate_raw=math.nan,
            reference_raw=0.0,
            reference_se_raw=0.0,
            quantity="delta",
            scale=_scale(),
            bounds=_bounds(),
        )


def test_aggregate_gate_passes_on_small_bias():
    result = evaluate_aggregate_gate([0.02, -0.01, 0.03], [0.005, 0.005, 0.005], _bounds())
    assert result.mean_signed_bias_c == pytest.approx(0.04 / 3.0)
    assert result.se_of_mean_c == pytest.approx(math.sqrt(3 * 0.005**2) / 3.0)
    assert result.passed


def test_aggregate_gate_catches_systematic_tilt():
    """Both cells sit inside the per-cell bound, but they lean the same way."""
    result = evaluate_aggregate_gate([0.2, 0.2], [0.001, 0.001], _bounds())
    assert not result.within_bound
    assert not result.passed


def test_aggregate_gate_requires_adequate_se():
    result = evaluate_aggregate_gate([0.0, 0.0], [0.1, 0.1], _bounds())
    assert not result.se_adequate
    assert not result.passed


def test_aggregate_gate_rejects_empty_input():
    with pytest.raises(ValidationError):
        evaluate_aggregate_gate([], [], _bounds())


def test_aggregate_gate_rejects_length_mismatch():
    with pytest.raises(ValidationError):
        evaluate_aggregate_gate([0.1], [0.01, 0.02], _bounds())
