"""simple_forward_rate: the one floating-projection helper (spec 2026-11-05, T4 / AC18)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantark.param import CurveInterpolation, FlatRateCurve, ZeroCurve
from quantark.param import simple_forward_rate as from_param
from quantark.param.rrf import simple_forward_rate
from quantark.util.exceptions import ValidationError

R = 0.05
FLAT = FlatRateCurve(R)
PILLARS = [(0.25, 0.030), (0.5, 0.031), (1.0, 0.033), (2.0, 0.0345), (5.0, 0.036), (10.0, 0.035)]


def test_exported_from_param_and_rrf():
    assert from_param is simple_forward_rate


@pytest.mark.parametrize("t1,t2,dcf", [(0.0, 0.25, 0.25), (0.5, 0.75, 91 / 360), (1.0, 2.0, 1.0), (0.0, 0.25, None)])
def test_flat_curve_closed_form(t1, t2, dcf):
    expected_dcf = dcf if dcf is not None else t2 - t1
    expected = (math.exp(R * (t2 - t1)) - 1.0) / expected_dcf
    assert simple_forward_rate(FLAT, t1, t2, dcf) == pytest.approx(expected, rel=1e-15)


def test_zero_curve_discount_factor_ratio():
    zc = ZeroCurve(PILLARS, interpolation=CurveInterpolation.LOG_CUBIC_DF)
    for t1, t2, dcf in [(0.0, 0.25, 0.25), (0.3, 0.55, 0.2528), (2.0, 2.25, 0.25), (9.0, 10.5, 1.5)]:
        expected = (zc.get_discount_factor(t1) / zc.get_discount_factor(t2) - 1.0) / dcf
        assert simple_forward_rate(zc, t1, t2, dcf) == pytest.approx(expected, rel=1e-15)
    # a continuously-compounded forward is NOT what this returns (that was the old shortcut)
    assert simple_forward_rate(zc, 1.0, 1.25, 0.25) != pytest.approx(zc.get_forward_rate(1.0, 1.25), abs=1e-6)


def test_running_period_pro_rata_spot_branch():
    t_s, t_e, dcf = -0.1, 0.15, 0.25
    df_e = FLAT.get_discount_factor(t_e)
    expected = (1.0 / df_e - 1.0) / (dcf * t_e / (t_e - t_s))
    assert simple_forward_rate(FLAT, t_s, t_e, dcf) == pytest.approx(expected, rel=1e-15)
    # dcf defaults to t_end - t_start, so the branch reduces to the spot rate to t_e
    assert simple_forward_rate(FLAT, t_s, t_e) == pytest.approx((1.0 / df_e - 1.0) / t_e, rel=1e-15)
    # continuity with the standard branch as t_start -> 0
    assert simple_forward_rate(FLAT, -1e-12, t_e, dcf) == pytest.approx(
        simple_forward_rate(FLAT, 0.0, t_e, dcf), rel=1e-9
    )


def test_returns_float():
    out = simple_forward_rate(FLAT, 0.0, 0.25, 0.25)
    assert isinstance(out, float) and not isinstance(out, np.ndarray)


def test_validation():
    with pytest.raises(ValidationError):
        simple_forward_rate(FLAT, 0.5, 0.5, 0.25)
    with pytest.raises(ValidationError):
        simple_forward_rate(FLAT, 0.5, 0.25, 0.25)
    with pytest.raises(ValidationError):
        simple_forward_rate(FLAT, -0.5, 0.0, 0.5)
    with pytest.raises(ValidationError):
        simple_forward_rate(FLAT, -0.5, -0.25, 0.25)
    with pytest.raises(ValidationError):
        simple_forward_rate(FLAT, 0.0, 0.25, 0.0)
    with pytest.raises(ValidationError):
        simple_forward_rate(FLAT, 0.0, 0.25, -0.25)
    with pytest.raises(ValidationError):
        simple_forward_rate(FLAT, 0.0, float("nan"), 0.25)
