"""Signed dividend/carry yields (negative implied carry from futures)."""
import math

import pytest

from quantark.param.div.dividend_yield import (
    ContinuousDividendYield,
    TermStructureDividendYield,
)
from quantark.util.exceptions import ValidationError


def test_continuous_accepts_negative_within_bound():
    assert ContinuousDividendYield(-0.05).get_yield(1.0) == -0.05


def test_continuous_rejects_beyond_symmetric_bound():
    with pytest.raises(ValidationError):
        ContinuousDividendYield(-0.25)
    with pytest.raises(ValidationError):
        ContinuousDividendYield(0.25)


def test_term_structure_accepts_negative_nodes():
    ts = TermStructureDividendYield(times=[0.1, 0.5], yields=[-0.02, 0.03])
    assert ts.get_yield(0.1) == pytest.approx(-0.02)
    assert ts.get_yield(0.05) == pytest.approx(-0.02)  # flat extrapolation


def test_term_structure_rejects_magnitude_over_one():
    with pytest.raises(ValidationError):
        TermStructureDividendYield(times=[0.1, 0.5], yields=[-1.5, 0.03])


def test_term_structure_rejects_non_finite():
    with pytest.raises(ValidationError):
        TermStructureDividendYield(times=[0.1, 0.5], yields=[math.nan, 0.03])
