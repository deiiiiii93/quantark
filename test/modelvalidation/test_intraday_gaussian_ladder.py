"""The radius rule of the deterministic intraday reference: what a nested O(h^2) ladder may and may not claim."""
import math

import pytest

from quantark.modelvalidation.builders.intraday_gaussian import ladder_estimate


def _ladder(truth, c2, c4=0.0, junk=(0.0, 0.0, 0.0, 0.0)):
    return [truth + c2 * h * h + c4 * h ** 4 + j for h, j in zip((1.0, 0.5, 0.25, 0.125), junk)]


def test_a_clean_second_order_ladder_extrapolates_and_the_geometric_radius_covers_the_truth():
    estimate = ladder_estimate(_ladder(7.0, 1e-3, 4e-5))
    assert estimate.rule == "geometric" and estimate.observed_order == pytest.approx(2.0, abs=0.05)
    assert abs(estimate.value - 7.0) <= estimate.radius
    assert estimate.radius < 1e-3 * 0.125 ** 2 / 3.0                 # far sharper than the unextrapolated estimate


def test_a_lucky_small_spread_is_not_credited_below_fourth_order_contraction():
    values = _ladder(7.0, 1e-3, 4e-5)
    values[-1] = (3.0 * ((4.0 * values[2] - values[1]) / 3.0) + values[2]) / 4.0          # makes E_n equal E_{n-1}
    estimate = ladder_estimate(values)
    assert estimate.rule == "geometric" and estimate.radius == pytest.approx(abs(estimate.extrapolants[1] - estimate.extrapolants[0]) / 16.0)


def test_extrapolants_on_a_junk_floor_claim_only_the_whole_correction():
    estimate = ladder_estimate(_ladder(2.36, 1e-2, junk=(0.0, 2e-7, -3e-7, 6e-7)))
    assert estimate.rule == "correction"
    assert estimate.radius == pytest.approx(abs(estimate.value - _ladder(2.36, 1e-2, junk=(0.0, 2e-7, -3e-7, 6e-7))[-1]))
    assert abs(estimate.value - 2.36) <= estimate.radius


def test_an_unconfirmed_order_is_never_extrapolated_and_pays_a_safety_factor_of_three():
    first_order = [1.0 + 0.1 * h for h in (1.0, 0.5, 0.25, 0.125)]
    estimate = ladder_estimate(first_order)
    assert estimate.rule == "unextrapolated" and estimate.value == first_order[-1]
    assert estimate.observed_order == pytest.approx(1.0) and estimate.radius == pytest.approx(3.0 * 0.025)
    assert abs(estimate.value - 1.0) <= estimate.radius


def test_alternating_differences_do_not_confirm_an_order_whatever_their_ratio():
    # the measured desk gamma of a single-event snowball at spot 78: converged to 1e-6, but oscillating
    values = [-0.5206546907875997, -0.5206639790534869, -0.5206604262645026, -0.5206609985395039]
    estimate = ladder_estimate(values)
    assert estimate.rule == "unextrapolated" and estimate.value == values[-1]
    assert estimate.radius == pytest.approx(3.0 * abs(values[2] - values[1]))


def test_a_ladder_that_moves_more_at_the_finest_level_claims_nothing():
    estimate = ladder_estimate([1.0, 1.001, 1.0011, 1.0031])
    assert estimate.rule == "unbounded" and math.isinf(estimate.radius)


def test_an_exact_solve_has_a_floating_point_radius_and_no_order():
    estimate = ladder_estimate([11.999958904179959] * 4)
    assert estimate.rule == "exact" and estimate.radius == 0.0 and estimate.observed_order is None
    assert ladder_estimate([0.0, 0.0, 0.0, 0.0]).radius == 0.0


def test_fewer_than_four_levels_cannot_show_whether_the_extrapolants_contract():
    with pytest.raises(ValueError, match="at least 4"):
        ladder_estimate([1.0, 1.1, 1.12])
