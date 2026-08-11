"""Sequential (anytime-valid) admission policy for Greek certification.

The fixed-allocation gate spends a declared number of batches per cell and
judges once at the end with a Student-t half-width.  A sequential rule instead
watches the gate as batches arrive and stops as soon as the answer is decided.
That is only sound if the half-width is anytime-valid and every parameter is
declared BEFORE the run: peeking at a fixed-t interval and stopping at the
first favourable look inflates the error rate.

These tests pin the properties that make the policy adoptable rather than the
particular savings it produces.
"""

import math

import pytest
from scipy.stats import t as student_t

from quantark.validation import (
    SequentialAdmissionPolicy,
    SequentialAdmissionStatus,
    confidence_sequence_half_width,
    scan_admission_stream,
    sequential_admission,
)


# Mirrors the production certification: 14 cells x 2 greeks, the economic bound
# in contracts, and the fixed-gate confidence the frozen rule uses.
FAMILY_ALPHA = 0.05
TESTS = 26
BOUND = 0.5
FIXED_GATE_CONFIDENCE = 0.975


def _policy(**overrides) -> SequentialAdmissionPolicy:
    kwargs = dict(
        family_alpha=FAMILY_ALPHA,
        tests=TESTS,
        min_batches=16,
        aggregate_floor_batches=128,
        planned_batches=256,
        max_batches=1024,
    )
    kwargs.update(overrides)
    return SequentialAdmissionPolicy(**kwargs)


def test_alpha_is_split_across_every_test_and_component():
    """Bonferroni over the declared test count, then over the two components."""
    policy = _policy()

    assert policy.components_per_test == 2
    assert policy.alpha_per_component == pytest.approx(
        FAMILY_ALPHA / (TESTS * 2)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"family_alpha": 0.0},
        {"family_alpha": 1.0},
        {"tests": 0},
        {"min_batches": 1},
        {"max_batches": 8},  # below the floor
        {"planned_batches": 4},  # outside [first_decidable, max]
        {"aggregate_floor_batches": -1},
    ],
)
def test_policy_rejects_parameters_that_would_void_the_guarantee(overrides):
    with pytest.raises(ValueError):
        _policy(**overrides)


def test_the_decision_floor_respects_the_aggregate_common_scramble_requirement():
    """A cell may not stop below what the downstream aggregate consumes.

    The aggregate gate reads a common scramble prefix across cells, so a cell
    that stopped earlier than that prefix would leave the aggregate unable to
    form its cohort rows at all.
    """
    policy = _policy(min_batches=16, aggregate_floor_batches=128)
    assert policy.first_decidable_batch == 128

    # With no aggregate consumer the floor falls back to the per-cell minimum.
    assert _policy(aggregate_floor_batches=0).first_decidable_batch == 16


def test_rho_is_tuned_from_the_declared_plan_and_never_from_the_data():
    """rho fixes the shape of the confidence sequence and must be data-free."""
    policy = _policy()

    assert policy.rho_squared == _policy().rho_squared > 0.0
    # A different declared plan legitimately tunes elsewhere.
    assert _policy(planned_batches=512).rho_squared != policy.rho_squared


def test_half_width_shrinks_as_batches_accumulate():
    policy = _policy()
    widths = [
        confidence_sequence_half_width(
            batches,
            standard_deviation=0.4,
            rho_squared=policy.rho_squared,
            alpha=policy.alpha_per_component,
        )
        for batches in (16, 32, 64, 128, 256, 512)
    ]

    assert all(later < earlier for earlier, later in zip(widths, widths[1:]))
    assert widths[0] > 0.0


def test_zero_dispersion_collapses_the_half_width():
    policy = _policy()

    assert (
        confidence_sequence_half_width(
            64,
            standard_deviation=0.0,
            rho_squared=policy.rho_squared,
            alpha=policy.alpha_per_component,
        )
        == 0.0
    )


@pytest.mark.parametrize("batches", [16, 32, 128, 512, 2048])
def test_anytime_validity_costs_width_against_the_fixed_batch_gate(batches):
    """The sequential half-width must never be narrower than the frozen gate's.

    This is the safety property that makes early stopping defensible: because
    the confidence sequence is wider at every batch count, a sequential ADMIT
    at t implies the fixed-allocation gate would also have passed had it been
    judged at t.  Early stopping therefore trades wall-clock for width, never
    evidence for optimism.
    """
    policy = _policy()
    sd = 0.4

    sequential = confidence_sequence_half_width(
        batches,
        standard_deviation=sd,
        rho_squared=policy.rho_squared,
        alpha=policy.alpha_per_component,
    )
    fixed = float(
        student_t.ppf(0.5 + 0.5 * FIXED_GATE_CONFIDENCE, batches - 1)
    ) * sd / math.sqrt(batches)

    assert sequential > fixed


def test_admission_needs_the_gap_the_pde_envelope_and_the_bias_envelope_together():
    """All four terms share one budget, exactly as the frozen gate does."""
    policy = _policy()
    common = dict(
        policy=policy,
        batches_used=256,
        reference_gap=0.05,
        greek_batch_standard_deviation=0.2,
        economic_bound=BOUND,
    )

    admitted = sequential_admission(
        **common, pde_discretization_envelope=0.01, substep_bias_mean=0.01,
        substep_batch_standard_deviation=0.05,
    )
    assert admitted.status is SequentialAdmissionStatus.ADMIT
    assert admitted.total_uncertainty < BOUND

    # Same MC evidence, but a bias mean that eats the budget: no admission.
    blocked = sequential_admission(
        **common, pde_discretization_envelope=0.01, substep_bias_mean=0.40,
        substep_batch_standard_deviation=0.05,
    )
    assert blocked.status is SequentialAdmissionStatus.CONTINUE


def test_no_admission_before_the_declared_floor_however_clean_the_stream():
    """An overwhelming pass at batch 20 still may not stop before the floor."""
    policy = _policy(min_batches=16, aggregate_floor_batches=128)

    early = sequential_admission(
        policy=policy,
        batches_used=20,
        reference_gap=0.0,
        greek_batch_standard_deviation=1e-6,
        pde_discretization_envelope=0.0,
        economic_bound=BOUND,
    )
    assert early.status is SequentialAdmissionStatus.CONTINUE
    assert "floor" in early.reason


def test_a_conclusive_rejection_grants_the_bias_envelope_nothing():
    """FAIL must be provable from the gap alone, never from the bias term.

    The bias envelope only widens the interval, so counting it towards a
    rejection would let an unresolved bias masquerade as a failed cell.
    """
    policy = _policy()

    rejected = sequential_admission(
        policy=policy,
        batches_used=256,
        reference_gap=2.0,
        greek_batch_standard_deviation=0.05,
        pde_discretization_envelope=0.01,
        economic_bound=BOUND,
        substep_bias_mean=5.0,
        substep_batch_standard_deviation=0.05,
    )
    assert rejected.status is SequentialAdmissionStatus.REJECT


def test_the_cap_exhausts_rather_than_admitting_on_a_technicality():
    policy = _policy(max_batches=256)

    exhausted = sequential_admission(
        policy=policy,
        batches_used=256,
        reference_gap=0.45,
        greek_batch_standard_deviation=0.9,
        pde_discretization_envelope=0.02,
        economic_bound=BOUND,
    )
    assert exhausted.status is SequentialAdmissionStatus.EXHAUSTED


def test_the_policy_digest_moves_with_every_declared_parameter():
    """The policy is frozen and hashed before the run, like the allocation."""
    baseline = _policy().sha256()

    for overrides in (
        {"family_alpha": 0.01},
        {"tests": 27},
        {"min_batches": 32},
        {"aggregate_floor_batches": 64},
        {"planned_batches": 512},
        {"max_batches": 2048},
    ):
        assert _policy(**overrides).sha256() != baseline, overrides
    assert _policy().sha256() == baseline


def test_scanning_a_clean_stream_stops_at_the_floor_and_reports_the_saving():
    policy = _policy(min_batches=16, aggregate_floor_batches=32, max_batches=1024)
    # A stream whose running mean sits on the PDE value with modest dispersion.
    series = [0.10 + 0.002 * (-1) ** index for index in range(1024)]
    substep = [0.001 * (-1) ** index for index in range(1024)]

    outcome = scan_admission_stream(
        policy=policy,
        pde_value=0.10,
        greek_series=series,
        substep_series=substep,
        pde_discretization_envelope=0.01,
        economic_bound=BOUND,
    )

    assert outcome.status is SequentialAdmissionStatus.ADMIT
    assert outcome.batches_used == policy.first_decidable_batch
    assert outcome.batches_saved_against(1024) == 1024 - policy.first_decidable_batch


def test_scanning_a_stream_that_never_decides_reports_exhausted_not_admit():
    policy = _policy(
        min_batches=16,
        aggregate_floor_batches=16,
        planned_batches=64,
        max_batches=64,
    )
    # Gap parked right on the bound with heavy dispersion: undecidable.
    series = [0.5 + 0.9 * (-1) ** index for index in range(64)]

    outcome = scan_admission_stream(
        policy=policy,
        pde_value=0.0,
        greek_series=series,
        substep_series=None,
        pde_discretization_envelope=0.02,
        economic_bound=BOUND,
    )

    assert outcome.status is SequentialAdmissionStatus.EXHAUSTED
    assert outcome.batches_used == 64
    assert outcome.batches_saved_against(64) == 0


def test_scanning_refuses_a_stream_shorter_than_the_decision_floor():
    policy = _policy(aggregate_floor_batches=128)

    with pytest.raises(ValueError):
        scan_admission_stream(
            policy=policy,
            pde_value=0.0,
            greek_series=[0.1] * 64,
            substep_series=None,
            pde_discretization_envelope=0.0,
            economic_bound=BOUND,
        )
