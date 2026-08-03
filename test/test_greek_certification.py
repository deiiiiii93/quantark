import numpy as np
import pytest

from quantark.validation import (
    EconomicGreekScale,
    EquivalenceStatus,
    certify_equivalence,
    certify_signed_bias_from_batches,
)


@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        (0.20, EquivalenceStatus.PASS),
        (0.80, EquivalenceStatus.FAIL),
        (0.48, EquivalenceStatus.INCONCLUSIVE),
    ],
)
def test_equivalence_gate_is_tri_state(difference, expected):
    result = certify_equivalence(
        difference,
        reference_standard_error=0.05,
        reference_degrees_of_freedom=15,
        economic_bound=0.5,
    )

    assert result.status is expected
    assert result.lower < result.upper


def test_equivalence_gate_adds_pde_refinement_envelope():
    without_envelope = certify_equivalence(0.35, 0.0, 0.5)
    with_envelope = certify_equivalence(
        0.35,
        0.0,
        0.5,
        pde_discretization_envelope=0.20,
    )

    assert without_envelope.status is EquivalenceStatus.PASS
    assert with_envelope.status is EquivalenceStatus.INCONCLUSIVE


def test_positive_reference_error_without_degrees_of_freedom_is_inconclusive():
    result = certify_equivalence(0.0, 0.1, 0.5)

    assert result.status is EquivalenceStatus.INCONCLUSIVE
    assert "degrees of freedom" in result.reason


def test_economic_scale_reports_delta_and_gamma_in_hedge_contracts():
    scale = EconomicGreekScale(
        model_spot=100.0,
        hedge_inception_spot=5_000.0,
        study_notional=50_000_000.0,
        hedge_multiplier=200.0,
    )

    assert scale.delta_quantum_per_contract == pytest.approx(0.02)
    assert scale.delta_contracts(0.01) == pytest.approx(0.5)
    assert scale.gamma_hedge_contract_change(0.01) == pytest.approx(0.5)


def test_economic_scale_keeps_normalized_model_spot_separate_from_hedge_spot():
    normalized = EconomicGreekScale(
        model_spot=100.0,
        hedge_inception_spot=5_000.0,
        study_notional=50_000_000.0,
        hedge_multiplier=200.0,
    )
    live_move = EconomicGreekScale(
        model_spot=75.0,
        hedge_inception_spot=5_000.0,
        study_notional=50_000_000.0,
        hedge_multiplier=200.0,
    )

    assert normalized.delta_quantum_per_contract == live_move.delta_quantum_per_contract
    assert normalized.delta_contracts(0.01) == live_move.delta_contracts(0.01)
    assert live_move.gamma_hedge_contract_change(0.01) == pytest.approx(0.375)


def test_signed_bias_uses_outer_batch_distribution():
    batches = np.array([0.02, 0.04, 0.03, 0.01, 0.02, 0.03])
    result = certify_signed_bias_from_batches(batches, economic_bound=0.1)

    assert result.status is EquivalenceStatus.PASS
    assert result.estimate_difference == pytest.approx(np.mean(batches))
