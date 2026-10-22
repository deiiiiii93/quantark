"""Tests for the credit pricing environment."""
import math
from datetime import datetime

import pytest

from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.priceenv import CreditPricingEnvironment
from quantark.util.exceptions import MarketDataError


def _env(rate=0.03, hazard=0.02):
    return CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=rate),
        hazard_curve=FlatHazardCurve(hazard_rate=hazard),
    )


def test_env_exposes_discount_and_survival():
    env = _env(rate=0.03, hazard=0.02)
    assert env.get_discount_factor(2.0) == pytest.approx(math.exp(-0.03 * 2.0))
    assert env.get_survival_probability(2.0) == pytest.approx(math.exp(-0.02 * 2.0))
    assert env.get_hazard_rate(2.0) == pytest.approx(0.02)
    assert env.get_default_density(2.0) == pytest.approx(
        0.02 * math.exp(-0.02 * 2.0)
    )


def test_missing_curves_rejected():
    with pytest.raises(MarketDataError):
        CreditPricingEnvironment(
            valuation_date=datetime(2026, 6, 13),
            discount_curve=None,
            hazard_curve=FlatHazardCurve(hazard_rate=0.02),
        )
    with pytest.raises(MarketDataError):
        CreditPricingEnvironment(
            valuation_date=datetime(2026, 6, 13),
            discount_curve=FlatRateCurve(rate=0.03),
            hazard_curve=None,
        )


def test_hazard_shift_returns_new_env_and_preserves_original():
    env = _env(hazard=0.02)
    bumped = env.with_hazard_shift(0.0001)  # +1bp
    assert bumped is not env
    assert bumped.get_hazard_rate(1.0) == pytest.approx(0.0201)
    # original untouched (immutability)
    assert env.get_hazard_rate(1.0) == pytest.approx(0.02)


def test_rate_shift_returns_new_env_and_preserves_original():
    env = _env(rate=0.03)
    bumped = env.with_rate_shift(0.0001)  # +1bp parallel
    assert bumped.get_discount_factor(1.0) == pytest.approx(math.exp(-0.0301 * 1.0))
    assert env.get_discount_factor(1.0) == pytest.approx(math.exp(-0.03 * 1.0))
