"""Tests for credit hazard-rate curves (reduced-form survival modelling)."""
import math

import pytest

from quantark.param.credit import FlatHazardCurve, HazardCurve
from quantark.util.exceptions import ValidationError


def test_flat_hazard_curve_is_a_hazard_curve():
    assert isinstance(FlatHazardCurve(hazard_rate=0.02), HazardCurve)


def test_flat_hazard_constant_rate():
    curve = FlatHazardCurve(hazard_rate=0.03)
    assert curve.get_hazard_rate(0.5) == pytest.approx(0.03)
    assert curve.get_hazard_rate(10.0) == pytest.approx(0.03)


def test_flat_hazard_survival_probability():
    curve = FlatHazardCurve(hazard_rate=0.04)
    # S(t) = exp(-lambda * t)
    assert curve.get_survival_probability(2.0) == pytest.approx(math.exp(-0.04 * 2.0))
    assert curve.get_survival_probability(0.0) == pytest.approx(1.0)


def test_flat_hazard_default_density():
    curve = FlatHazardCurve(hazard_rate=0.05)
    # q(t) = lambda * S(t)
    t = 3.0
    assert curve.get_default_density(t) == pytest.approx(
        0.05 * math.exp(-0.05 * t)
    )


def test_flat_hazard_default_probability_complements_survival():
    curve = FlatHazardCurve(hazard_rate=0.02)
    t = 5.0
    assert curve.get_default_probability(t) == pytest.approx(
        1.0 - curve.get_survival_probability(t)
    )


def test_negative_hazard_rejected():
    with pytest.raises(ValidationError):
        FlatHazardCurve(hazard_rate=-0.01)


def test_negative_time_rejected():
    curve = FlatHazardCurve(hazard_rate=0.02)
    with pytest.raises(ValidationError):
        curve.get_survival_probability(-1.0)
