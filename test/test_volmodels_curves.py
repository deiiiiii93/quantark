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


class _TermCurve:
    """Minimal term-structure curve exposing get_forward_rate for the helper."""
    def get_forward_rate(self, t0, t1):
        return 0.02 + 0.01 * t1


def test_non_flat_curve_forward_rates():
    fwd = forward_rates_on_grid(_TermCurve(), np.array([0.0, 0.5, 1.0]))
    assert fwd[0] == pytest.approx(0.02 + 0.01 * 0.5, abs=1e-12)
    assert fwd[1] == pytest.approx(0.02 + 0.01 * 1.0, abs=1e-12)


def test_grid_rejects_nonfinite():
    with pytest.raises(ValidationError):
        forward_rates_on_grid(_TermCurve(), np.array([0.0, np.nan, 1.0]))


def test_forward_rates_df_based_non_flat():
    # LinearRateCurve with 2%@0.5y, 4%@1.0y -> DF-based forward over [0.5,1.0]:
    # f = -(ln DF(1.0) - ln DF(0.5))/0.5 = -((-0.04) - (-0.01))/0.5 = 0.06
    from quantark.param.rrf import LinearRateCurve
    curve = LinearRateCurve([(0.5, 0.02), (1.0, 0.04)])
    fwd = forward_rates_on_grid(curve, np.array([0.5, 1.0]))
    assert fwd[0] == pytest.approx(0.06, abs=1e-12)


def test_forward_rates_rejects_nonfinite_curve_output():
    class _BadCurve:
        def get_forward_rate(self, t0, t1):
            return float("nan")
    from quantark.util.exceptions import NumericalError
    with pytest.raises(NumericalError):
        forward_rates_on_grid(_BadCurve(), np.array([0.0, 0.5, 1.0]))
