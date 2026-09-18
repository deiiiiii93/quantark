"""The radius policy of the deterministic intraday reference: what a nested O(h^2) ladder may and may not claim.

A ladder radius is a CALIBRATED NUMERICAL ESTIMATE, never an analytical bound. The policy is one versioned,
structured object that both executes and serializes, so the certificate's contract is the policy that ran
(review 2026-09-18, R1 and R4).
"""
import dataclasses
import math

import pytest

from quantark.modelvalidation.builders.intraday_gaussian import LADDER_POLICY, LadderPolicy, ladder_estimate

H = (1.0, 0.5, 0.25, 0.125, 0.0625)


def _ladder(truth, c2, c4=0.0, junk=(0.0,) * 5):
    return [truth + c2 * h * h + c4 * h ** 4 + j for h, j in zip(H, junk)]


def test_a_clean_second_order_ladder_extrapolates_and_is_calibrated_one_level_down():
    estimate = ladder_estimate(_ladder(7.0, 1e-3, 4e-5))
    assert estimate.rule == "geometric" and estimate.observed_order == pytest.approx(2.0, abs=0.05)
    assert abs(estimate.value - 7.0) <= estimate.radius < 1e-3 * H[-1] ** 2 / 3.0     # sharper than the unextrapolated estimate
    calibration = estimate.calibration
    assert calibration["covered"] and calibration["coarse_rule"] == "geometric"
    assert calibration["move"] <= calibration["coarse_radius"]              # the rule one level down covered this value


def test_a_lucky_small_spread_is_not_credited_below_fourth_order_contraction():
    values = _ladder(7.0, 1e-3, 4e-5)
    values[-1] = (3.0 * ((4.0 * values[3] - values[2]) / 3.0) + values[3]) / 4.0          # makes E_n equal E_{n-1}
    estimate = ladder_estimate(values)
    assert estimate.rule == "geometric"
    assert estimate.radius == pytest.approx(abs(estimate.extrapolants[1] - estimate.extrapolants[0]) / 16.0)


def test_extrapolants_on_a_junk_floor_claim_only_the_whole_correction():
    values = _ladder(2.36, 1e-2, junk=(0.0, 1e-7, 2e-7, -3e-7, 6e-7))
    estimate = ladder_estimate(values)
    assert estimate.rule == "correction" and estimate.radius == pytest.approx(abs(estimate.value - values[-1]))
    assert abs(estimate.value - 2.36) <= estimate.radius


def test_an_unconfirmed_order_is_never_extrapolated_and_pays_a_safety_factor_of_three():
    first_order = [1.0 + 0.1 * h for h in H]
    estimate = ladder_estimate(first_order)
    assert estimate.rule == "unextrapolated" and estimate.value == first_order[-1]
    assert estimate.observed_order == pytest.approx(1.0) and estimate.radius == pytest.approx(3.0 * 0.1 * (H[2] - H[3]))
    assert abs(estimate.value - 1.0) <= estimate.radius


def test_alternating_differences_do_not_confirm_an_order_whatever_their_ratio():
    # the measured desk gamma of a single-event snowball at spot 78: converged to 1e-6, but oscillating
    values = [-0.52067, -0.5206546907875997, -0.5206639790534869, -0.5206604262645026, -0.5206609985395039]
    estimate = ladder_estimate(values)
    assert estimate.rule == "unextrapolated" and estimate.value == values[-1]
    assert estimate.radius == pytest.approx(3.0 * abs(values[3] - values[2]))


def test_a_ladder_that_moves_more_at_the_finest_level_claims_nothing():
    estimate = ladder_estimate([1.0, 1.0005, 1.001, 1.0011, 1.0031])
    assert estimate.rule == "unbounded" and math.isinf(estimate.radius)


# --- review R1: equal values do not establish exactness -----------------------------------------------------------
def test_r1_unexplained_stagnation_is_unresolved_never_exact():
    estimate = ladder_estimate([-26.03306581850112] * 5)             # the reviewer's repeated 201-point local grid
    assert estimate.rule == "stagnant" and math.isinf(estimate.radius) and estimate.basis is None


def test_r1_an_identified_exact_case_needs_its_basis_named():
    estimate = ladder_estimate([11.999958904179959] * 5, exact_basis="terminated")
    assert estimate.rule == "exact" and estimate.radius == 0.0 and estimate.basis == "terminated"
    assert estimate.calibration is None                                # an analytical case needs no ladder calibration
    with pytest.raises(ValueError, match="does not stagnate"):
        ladder_estimate(_ladder(7.0, 1e-3), exact_basis="terminated")  # a basis may not be claimed for a moving ladder


def test_r1_a_rule_that_did_not_cover_the_next_level_is_uncalibrated():
    # four clean levels claim a radius of 1.5e-5; the fifth shows the extrapolant still moving by 1.3e-4
    values = _ladder(1.0, 0.1, 1e-3)
    values[-1] += 1e-4
    estimate = ladder_estimate(values)
    assert estimate.calibration["coarse_rule"] == "geometric" and estimate.calibration["coarse_radius"] == pytest.approx(1.46e-5, rel=0.02)
    assert estimate.rule == "uncalibrated" and math.isinf(estimate.radius)
    assert not estimate.calibration["covered"] and estimate.calibration["move"] > estimate.calibration["coarse_radius"]


def test_r1_four_levels_cannot_calibrate_the_rule():
    with pytest.raises(ValueError, match="at least 5"):
        ladder_estimate([1.0, 1.1, 1.12, 1.125])


# --- review R4: one structured, versioned policy executes and serializes ------------------------------------------
def test_r4_the_descriptor_names_every_branch_and_every_parameter():
    described = LADDER_POLICY.describe()
    assert described["version"] == LADDER_POLICY.version and described["radius_kind"] == "calibrated_numerical_estimate"
    assert set(described["branches"]) == {"exact", "stagnant", "geometric", "correction", "unextrapolated", "unbounded", "uncalibrated"}
    assert described["parameters"] == {
        "discretization_order": 2, "refinement_ratio": 2, "min_levels": 5, "order_window": [1.5, 2.5],
        "contraction": 0.5, "max_contraction": 16.0, "unconfirmed_safety": 3.0, "stagnation_floor": 1e-13,
        "roundoff_factor": 16.0}
    assert "unextrapolated" in described["branches"] and "3.0" in described["branches"]["unextrapolated"]


def test_r4_a_changed_parameter_changes_both_the_execution_and_the_descriptor():
    first_order = [1.0 + 0.1 * h for h in H]
    cautious = dataclasses.replace(LADDER_POLICY, unconfirmed_safety=5.0)
    assert ladder_estimate(first_order, policy=cautious).radius == pytest.approx(5.0 * 0.1 * (H[2] - H[3]))
    assert cautious.describe() != LADDER_POLICY.describe()
    assert "5.0" in cautious.describe()["branches"]["unextrapolated"]
    assert isinstance(LADDER_POLICY, LadderPolicy)
