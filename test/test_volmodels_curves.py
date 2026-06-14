import numpy as np
import pytest
from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError
from quantark.volmodels.curves import forward_rates_on_grid, forward_carry_on_grid


def test_flat_curve_forward_rates_constant():
    fwd = forward_rates_on_grid(FlatRateCurve(rate=0.05), np.array([0.0, 0.25, 0.5, 1.0]))
    assert fwd.shape == (3,)
    assert np.allclose(fwd, 0.05, atol=1e-12)


def test_forward_carry_from_zero_yields():
    def zero_yield(t):
        return 0.02 if t <= 0.5 else 0.03
    carry = forward_carry_on_grid(zero_yield, np.array([0.0, 0.5, 1.0]))
    assert carry.shape == (2,)
    assert carry[1] == pytest.approx((0.03 * 1.0 - 0.02 * 0.5) / 0.5, abs=1e-12)


def test_grid_must_be_increasing():
    with pytest.raises(ValidationError):
        forward_rates_on_grid(FlatRateCurve(rate=0.05), np.array([0.0, 0.5, 0.25]))
