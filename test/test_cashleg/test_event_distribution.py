import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from cashleg.event_distribution import EventDistribution, EventType, PricingResult
from util.exceptions import NumericalError


def test_trivial_distribution_for_vanilla_product():
    dist = EventDistribution.trivial(maturity=1.0)
    assert dist.event_times.tolist() == [0.0, 1.0]
    assert dist.probabilities[EventType.MATURITY_NO_KO] == 1.0
    assert dist.survival_probability.tolist() == [1.0, 1.0]


def test_survival_interpolation():
    dist = EventDistribution(
        event_times=np.array([0.25, 0.5, 0.75, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.1, 0.2, 0.1, 0.0]),
            EventType.MATURITY_NO_KO: 0.6,
        },
        survival_probability=np.array([1.0, 0.9, 0.7, 0.6, 0.6]),
    )
    assert dist.survival_at(0.125) == pytest.approx(0.95, abs=1e-9)
    assert dist.survival_at(0.25) == pytest.approx(0.9, abs=1e-9)
    assert dist.survival_at(0.0) == pytest.approx(1.0, abs=1e-9)


def test_invariant_probability_sum():
    with pytest.raises(NumericalError, match="probability"):
        EventDistribution(
            event_times=np.array([1.0]),
            event_dates=None,
            probabilities={
                EventType.KO: np.array([0.3]),
                EventType.MATURITY_NO_KO: 0.2,
            },
            survival_probability=np.array([1.0, 0.7]),
        )


def test_invariant_survival_monotone():
    with pytest.raises(NumericalError, match="monotone"):
        EventDistribution(
            event_times=np.array([0.5, 1.0]),
            event_dates=None,
            probabilities={EventType.MATURITY_NO_KO: 1.0},
            survival_probability=np.array([1.0, 0.5, 0.7]),
        )


def test_coupon_and_ki_are_excluded_from_termination_sum_invariant():
    dist = EventDistribution(
        event_times=np.array([0.5, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.3, 0.0]),
            EventType.KI: np.array([0.2, 0.1]),
            EventType.COUPON: np.array([0.8, 0.7]),
            EventType.MATURITY_NO_KO: 0.7,
        },
        survival_probability=np.array([1.0, 0.7, 0.7]),
    )
    assert dist.probabilities[EventType.COUPON].sum() > 1.0


def test_pricing_result_wraps_npv_and_distribution():
    dist = EventDistribution.trivial(1.0)
    result = PricingResult(npv=12.5, event_distribution=dist)
    assert result.npv == 12.5
    assert result.event_distribution is dist
