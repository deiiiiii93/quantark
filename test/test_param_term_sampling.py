"""Tests for quantark.param.term_sampling (moved from volmodels.curves)."""
import numpy as np
import pytest

from quantark.param.term_sampling import (
    forward_carry_on_grid,
    forward_rates_on_grid,
)
from quantark.param.rrf.rate_curve import FlatRateCurve, LinearRateCurve
from quantark.util.exceptions import ValidationError


def test_reexport_identity_with_volmodels_curves():
    from quantark.volmodels import curves as old

    assert old.forward_rates_on_grid is forward_rates_on_grid
    assert old.forward_carry_on_grid is forward_carry_on_grid


def test_forward_rates_flat_curve_is_flat():
    grid = np.array([0.0, 0.5, 1.0, 2.0])
    out = forward_rates_on_grid(FlatRateCurve(0.03), grid)
    assert out == pytest.approx([0.03, 0.03, 0.03])


def test_forward_rates_linear_curve_hand_computed():
    curve = LinearRateCurve([(1.0, 0.03), (2.0, 0.04)])
    out = forward_rates_on_grid(curve, np.array([0.0, 1.0, 2.0]))
    # zero(1)=3%; fwd over [1,2] = (0.04*2 - 0.03*1)/1 = 5%
    assert out == pytest.approx([0.03, 0.05], abs=1e-12)


def test_forward_carry_flat_yield_is_flat():
    grid = np.array([0.0, 0.5, 1.0])
    out = forward_carry_on_grid(lambda t: 0.02, grid)
    assert out == pytest.approx([0.02, 0.02])


def test_forward_carry_term_structure_hand_computed():
    def q(t):  # q(0.5)=1%, q(1.0)=2%, linear between
        return float(np.interp(t, [0.5, 1.0], [0.01, 0.02]))

    out = forward_carry_on_grid(q, np.array([0.0, 0.5, 1.0]))
    # fwd[0] = q(0.5)*0.5/0.5 = 1%; fwd[1] = (0.02*1 - 0.01*0.5)/0.5 = 3%
    assert out == pytest.approx([0.01, 0.03], abs=1e-12)


def test_grid_validation_rejects_bad_grids():
    curve = FlatRateCurve(0.03)
    with pytest.raises(ValidationError):
        forward_rates_on_grid(curve, np.array([0.5]))
    with pytest.raises(ValidationError):
        forward_rates_on_grid(curve, np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValidationError):
        forward_rates_on_grid(curve, np.array([-0.1, 1.0]))
