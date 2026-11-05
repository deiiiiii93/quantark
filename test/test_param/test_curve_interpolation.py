"""Shared term interpolation kernel (spec 2026-11-05 curve construction, T1).

Covers AC2 (scheme semantics), AC3 (short/long end on the interpolator) and
AC5 (arrays) for ``TermInterpolator``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from quantark.param.curve.interpolation import (
    CurveInterpolation,
    LongEndRule,
    TermInterpolator,
    long_end_rule,
    mc_g,
    mc_node_forwards,
    mc_region,
    mc_G,
)
from quantark.param.extrapolation import CarryExtrapolation, RateExtrapolation
from quantark.param.rrf.rate_curve import (
    CubicSplineRateCurve,
    LinearRateCurve,
    LogLinearRateCurve,
)
from quantark.util.exceptions import ValidationError

# gently varying, non-monotone zeros (10Y below 5Y); all discrete forwards distinct and
# positive (0.030, 0.032, 0.035, 0.036, 0.037, 0.034), so no positivity clipping fires and
# no Hagan–West knot is degenerate (g0 = 0 or g1 = 0)
PILLARS = [(0.25, 0.030), (0.5, 0.031), (1.0, 0.033), (2.0, 0.0345), (5.0, 0.036), (10.0, 0.035)]
# 2Y at 0.034 makes fd_3 == fd_4 == 0.035: f_3 == fd_3, so interval 3 has g1 = 0 != g0
DEGENERATE_PILLARS = [(0.25, 0.030), (0.5, 0.031), (1.0, 0.033), (2.0, 0.034), (5.0, 0.036), (10.0, 0.035)]
TIMES = np.array([p[0] for p in PILLARS])
LEVELS = np.array([p[1] for p in PILLARS])
SCHEMES = list(CurveInterpolation)
RULES = list(LongEndRule)


def make(scheme, rule=LongEndRule.HOLD_FORWARD, pillars=PILLARS):
    t = [p[0] for p in pillars]
    y = [p[1] for p in pillars]
    return TermInterpolator(t, y, scheme, rule)


# --------------------------------------------------------------------------- enums


def test_long_end_rule_mapping():
    assert long_end_rule(RateExtrapolation.FLAT_ZERO_RATE) is LongEndRule.HOLD_LEVEL
    assert long_end_rule(RateExtrapolation.FLAT_FORWARD_RATE) is LongEndRule.HOLD_FORWARD
    assert long_end_rule(CarryExtrapolation.FLAT_FORWARD_CARRY) is LongEndRule.HOLD_FORWARD
    assert long_end_rule(CarryExtrapolation.ZERO_FORWARD_CARRY) is LongEndRule.HOLD_CUMULATIVE
    with pytest.raises(ValidationError):
        long_end_rule("flat")  # type: ignore[arg-type]


def test_scheme_enum_values():
    assert {s.value for s in CurveInterpolation} == {
        "linear_zero", "log_linear_df", "cubic_zero", "log_cubic_df", "monotone_convex",
    }


# --------------------------------------------------------------------------- pillars (AC1 on the kernel)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("rule", RULES)
def test_pillars_reproduced(scheme, rule):
    interp = make(scheme, rule)
    for t, y in PILLARS:
        assert interp.level(t) == pytest.approx(y, abs=1e-14)
        assert interp.cumulative(t) == pytest.approx(y * t, rel=1e-15)


# --------------------------------------------------------------------------- scheme semantics (AC2)


def test_linear_zero_matches_linear_rate_curve():
    interp = make(CurveInterpolation.LINEAR_ZERO, LongEndRule.HOLD_LEVEL)
    ref = LinearRateCurve(PILLARS)
    for t in np.linspace(0.0, 10.0, 200):
        assert interp.level(t) == pytest.approx(ref.get_rate(t), abs=1e-15)


def test_log_linear_df_matches_log_linear_rate_curve():
    interp = make(CurveInterpolation.LOG_LINEAR_DF, LongEndRule.HOLD_FORWARD)
    ref = LogLinearRateCurve(PILLARS)
    for t in np.linspace(0.0, 10.0, 200):
        assert np.exp(-interp.cumulative(t)) == pytest.approx(ref.get_discount_factor(t), abs=1e-15)


def test_cubic_zero_matches_cubic_spline_rate_curve_inside():
    interp = make(CurveInterpolation.CUBIC_ZERO, LongEndRule.HOLD_LEVEL)
    ref = CubicSplineRateCurve(PILLARS)
    for t in np.linspace(0.25, 10.0, 200):
        assert interp.level(t) == pytest.approx(ref.get_rate(t), abs=1e-12)


def test_log_cubic_df_is_natural_spline_through_origin():
    interp = make(CurveInterpolation.LOG_CUBIC_DF, LongEndRule.HOLD_FORWARD)
    knots = np.concatenate([[0.0], TIMES])
    values = np.concatenate([[0.0], LEVELS * TIMES])
    cs = CubicSpline(knots, values, bc_type="natural")
    assert interp.cumulative(0.0) == 0.0
    for t in np.linspace(0.0, 10.0, 201)[:-1]:  # inside [0, T_n)
        assert interp.cumulative(t) == pytest.approx(float(cs(t)), abs=1e-15)
        assert interp.forward(t) == pytest.approx(float(cs(t, 1)), abs=1e-13)


def test_monotone_convex_reproduces_discrete_forwards():
    interp = make(CurveInterpolation.MONOTONE_CONVEX)
    tk = np.concatenate([[0.0], TIMES])
    yk = np.concatenate([[0.0], LEVELS * TIMES])
    fd = np.diff(yk) / np.diff(tk)
    for i in range(1, len(tk)):
        avg = (interp.cumulative(tk[i]) - interp.cumulative(tk[i - 1])) / (tk[i] - tk[i - 1])
        assert avg == pytest.approx(fd[i - 1], abs=1e-14)


def test_monotone_convex_forward_continuous_at_interior_knots():
    interp = make(CurveInterpolation.MONOTONE_CONVEX)
    eps = 1e-10
    for t in TIMES[:-1]:
        left = interp.forward(t - eps)
        right = interp.forward(t + eps)
        assert left == pytest.approx(right, abs=1e-9)
        # the knot value is the node forward from both sides
        assert interp.forward(t) == pytest.approx(right, abs=1e-9)


def test_monotone_convex_degenerate_knot_documented():
    """g1 = 0 != g0 on an interval (equal adjacent discrete forwards): the published
    formulas give g ≡ 0 on (0, 1], so the forward jumps at the interval's left knot
    while the cumulative stays continuous and every pillar is reproduced."""
    interp = make(CurveInterpolation.MONOTONE_CONVEX, pillars=DEGENERATE_PILLARS)
    tk = np.concatenate([[0.0], [p[0] for p in DEGENERATE_PILLARS]])
    yk = np.concatenate([[0.0], [p[0] * p[1] for p in DEGENERATE_PILLARS]])
    fd = np.diff(yk) / np.diff(tk)
    assert fd[2] == pytest.approx(fd[3], abs=1e-15)  # the coincidence
    left = interp.forward(0.5 - 1e-10)
    right = interp.forward(0.5 + 1e-10)
    assert left == pytest.approx(0.033, abs=1e-9)  # node forward f_2
    assert right == pytest.approx(fd[2], abs=1e-9)  # jumps to the discrete forward
    assert interp.cumulative(0.5 - 1e-10) == pytest.approx(interp.cumulative(0.5 + 1e-10), abs=1e-11)
    for t, y in DEGENERATE_PILLARS:
        assert interp.level(t) == pytest.approx(y, abs=1e-14)


def test_monotone_convex_forward_positive_when_discrete_forwards_positive():
    interp = make(CurveInterpolation.MONOTONE_CONVEX)
    grid = np.linspace(0.0, 10.0, 4001)
    assert np.all(interp.forward(grid) >= 0.0)


@pytest.mark.parametrize("level", [0.03, -0.01])
@pytest.mark.parametrize("rule", RULES)
def test_monotone_convex_flat_input_is_flat_output(level, rule):
    pillars = [(t, level) for t in TIMES]
    interp = make(CurveInterpolation.MONOTONE_CONVEX, rule, pillars)
    grid = np.linspace(0.0, 10.0, 401)
    assert np.max(np.abs(interp.forward(grid[:-1]) - level)) <= 1e-15
    assert np.max(np.abs(interp.level(grid) - level)) <= 1e-15


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("rule", RULES)
def test_flat_input_is_flat_output_every_scheme(scheme, rule):
    pillars = [(t, 0.025) for t in TIMES]
    interp = make(scheme, rule, pillars)
    grid = np.linspace(0.0, 10.0, 401)
    assert np.max(np.abs(interp.level(grid) - 0.025)) <= 1e-15
    assert np.max(np.abs(interp.forward(grid[:-1]) - 0.025)) <= 1e-14


# --------------------------------------------------------------------------- Hagan–West regions


REGION_CASES = {
    1: [(0.02, -0.02), (0.02, -0.03), (-0.02, 0.025), (0.01, -0.005), (0.01, -0.02)],
    2: [(-0.02, 0.05), (0.02, -0.05), (-0.01, 0.0201), (0.03, -0.1)],
    3: [(0.02, -0.005), (-0.02, 0.005), (0.04, -0.0199), (-0.04, 0.0001)],
    4: [(0.02, 0.01), (-0.02, -0.01), (0.03, 0.03), (-0.01, -0.04), (0.0, 0.02), (0.02, 0.0)],
}


@pytest.mark.parametrize("region,cases", REGION_CASES.items())
def test_mc_region_classification(region, cases):
    for g0, g1 in cases:
        assert mc_region(g0, g1)[0] == region, (g0, g1)
    assert mc_region(0.0, 0.0)[0] == 0


@pytest.mark.parametrize("region,cases", REGION_CASES.items())
def test_mc_G_integrates_g(region, cases):
    for g0, g1 in cases:
        if g0 == 0.0 or g1 == 0.0:
            continue  # degenerate limits, covered below
        assert mc_G(1.0, g0, g1) == pytest.approx(0.0, abs=1e-15)
        _, eta, _ = mc_region(g0, g1)
        xs = [0.2, 0.5, 0.8] + ([eta - 1e-3, eta + 1e-3] if 0.0 < eta < 1.0 else [])
        for x in xs:
            if not 0.0 < x < 1.0:
                continue
            numeric, _ = quad(lambda u: mc_g(u, g0, g1), 0.0, x, points=[eta] if 0.0 < eta < x else None,
                              epsabs=1e-14, epsrel=1e-12, limit=200)
            assert mc_G(x, g0, g1) == pytest.approx(numeric, abs=1e-12)


@pytest.mark.parametrize("region,cases", REGION_CASES.items())
def test_mc_g_continuous_at_eta_and_endpoints(region, cases):
    for g0, g1 in cases:
        if g0 == 0.0 or g1 == 0.0:
            continue
        _, eta, _ = mc_region(g0, g1)
        if 0.0 < eta < 1.0:
            assert mc_g(eta - 1e-9, g0, g1) == pytest.approx(mc_g(eta + 1e-9, g0, g1), abs=1e-7)
        assert mc_g(0.0, g0, g1) == pytest.approx(g0, abs=1e-15)
        assert mc_g(1.0, g0, g1) == pytest.approx(g1, abs=1e-15)


def test_mc_degenerate_limits_defined():
    # g0 = 0, g1 != 0 and g0 != 0, g1 = 0 fall to region (iv) with eta in {0, 1}
    assert np.isfinite(mc_g(1.0, 0.0, 0.02)) and np.isfinite(mc_G(1.0, 0.0, 0.02))
    assert mc_G(1.0, 0.0, 0.02) == pytest.approx(0.0, abs=1e-15)
    assert np.isfinite(mc_g(0.0, 0.02, 0.0)) and mc_G(1.0, 0.02, 0.0) == pytest.approx(0.0, abs=1e-15)
    arr = mc_g(np.linspace(0.0, 1.0, 11), 0.0, 0.02)
    assert np.all(np.isfinite(arr))


def test_mc_node_forwards_positivity_rule():
    # all discrete forwards positive: node forwards clipped into [0, 2*min]
    tk = np.array([0.0, 1.0, 2.0, 3.0])
    fd = np.array([0.01, 0.10, 0.01])
    f = mc_node_forwards(tk, fd)
    assert f[1] <= 2 * min(fd[0], fd[1]) + 1e-16 and f[1] >= 0.0
    assert f[2] <= 2 * min(fd[1], fd[2]) + 1e-16 and f[2] >= 0.0
    # a negative discrete forward disables the clip at its nodes (no empty interval)
    fd_neg = np.array([0.01, -0.02, 0.01])
    f_neg = mc_node_forwards(tk, fd_neg)
    assert np.all(np.isfinite(f_neg))
    w = (tk[1] - tk[0]) / (tk[2] - tk[0])
    assert f_neg[1] == pytest.approx(fd_neg[0] + w * (fd_neg[1] - fd_neg[0]))
    # a single interval: both node forwards equal the discrete forward
    assert np.all(mc_node_forwards(np.array([0.0, 2.0]), np.array([0.03])) == 0.03)


# --------------------------------------------------------------------------- forward vs cumulative (AC2)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_forward_is_derivative_of_cumulative(scheme):
    interp = make(scheme)
    h = 1e-6
    mids = [0.1, 0.375, 0.75, 1.5, 3.5, 7.5, 12.0]  # interior + beyond T_n
    for t in mids:
        fd = (interp.cumulative(t + h) - interp.cumulative(t - h)) / (2 * h)
        assert interp.forward(t) == pytest.approx(fd, rel=1e-7)


# --------------------------------------------------------------------------- short end and long end (AC3)


@pytest.mark.parametrize("scheme", [CurveInterpolation.LINEAR_ZERO, CurveInterpolation.CUBIC_ZERO])
def test_short_end_flat_level(scheme):
    interp = make(scheme)
    for t in [0.0, 0.05, 0.2, 0.25]:
        assert interp.level(t) == pytest.approx(LEVELS[0], abs=1e-15)


def test_short_end_log_linear_flat_forward():
    interp = make(CurveInterpolation.LOG_LINEAR_DF)
    for t in [0.0, 0.05, 0.2]:
        assert interp.forward(t) == pytest.approx(LEVELS[0], abs=1e-15)
    assert interp.level(0.0) == LEVELS[0]


@pytest.mark.parametrize("scheme", [CurveInterpolation.LOG_CUBIC_DF, CurveInterpolation.MONOTONE_CONVEX])
def test_short_end_average_forward(scheme):
    interp = make(scheme)
    assert interp.cumulative(TIMES[0]) / TIMES[0] == pytest.approx(LEVELS[0], abs=1e-15)
    assert interp.level(0.0) == LEVELS[0]  # convention


@pytest.mark.parametrize("scheme", SCHEMES)
def test_long_end_rules(scheme):
    tn, yn = PILLARS[-1]
    y_prev_t = PILLARS[-2][1] * PILLARS[-2][0]
    fd_last = (yn * tn - y_prev_t) / (tn - PILLARS[-2][0])
    hold_level = make(scheme, LongEndRule.HOLD_LEVEL)
    hold_fwd = make(scheme, LongEndRule.HOLD_FORWARD)
    hold_cum = make(scheme, LongEndRule.HOLD_CUMULATIVE)
    for T in [10.0 + 1e-12, 12.0, 20.0, 40.0]:
        assert hold_level.level(T) == pytest.approx(yn, abs=1e-15)
        assert hold_level.forward(T) == pytest.approx(yn, abs=1e-15)
        assert hold_fwd.cumulative(T) == pytest.approx(yn * tn + fd_last * (T - tn), rel=1e-14)
        assert hold_fwd.forward(T) == pytest.approx(fd_last, abs=1e-15)
        assert hold_cum.cumulative(T) == pytest.approx(yn * tn, rel=1e-15)
        assert hold_cum.forward(T) == 0.0
    # continuity at T_n
    for interp in (hold_level, hold_fwd, hold_cum):
        assert interp.cumulative(tn) == pytest.approx(yn * tn, rel=1e-15)
    # the right derivative AT T_n is the long-end forward
    assert hold_fwd.forward(tn) == pytest.approx(fd_last, abs=1e-15)
    assert hold_cum.forward(tn) == 0.0


# --------------------------------------------------------------------------- arrays (AC5) and validation


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("rule", RULES)
def test_arrays_equal_scalar_calls_exactly(scheme, rule):
    interp = make(scheme, rule)
    grid = np.array([0.0, 0.1, 0.25, 0.3, 1.0, 1.7, 5.0, 9.99, 10.0, 11.0, 30.0])
    for fn in (interp.level, interp.cumulative, interp.forward):
        vec = fn(grid)
        assert isinstance(vec, np.ndarray) and vec.shape == grid.shape
        for t, v in zip(grid, vec):
            assert fn(float(t)) == v
    assert isinstance(interp.level(1.0), float)
    # 2-D input keeps its shape
    assert interp.cumulative(grid.reshape(1, -1)).shape == (1, grid.size)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_single_pillar_is_flat(scheme):
    interp = TermInterpolator([2.0], [0.03], scheme, LongEndRule.HOLD_FORWARD)
    for t in [0.0, 0.5, 2.0, 3.0]:
        assert interp.level(t) == pytest.approx(0.03, abs=1e-15)
        assert interp.forward(t) == pytest.approx(0.03, abs=1e-15)


def test_validation():
    with pytest.raises(ValidationError):
        TermInterpolator([], [], CurveInterpolation.LINEAR_ZERO, LongEndRule.HOLD_LEVEL)
    with pytest.raises(ValidationError):
        TermInterpolator([0.0, 1.0], [0.01, 0.02], CurveInterpolation.LINEAR_ZERO, LongEndRule.HOLD_LEVEL)
    with pytest.raises(ValidationError):
        TermInterpolator([1.0, 1.0], [0.01, 0.02], CurveInterpolation.LINEAR_ZERO, LongEndRule.HOLD_LEVEL)
    with pytest.raises(ValidationError):
        TermInterpolator([2.0, 1.0], [0.01, 0.02], CurveInterpolation.LINEAR_ZERO, LongEndRule.HOLD_LEVEL)
    with pytest.raises(ValidationError):
        TermInterpolator([1.0, 2.0], [0.01, float("nan")], CurveInterpolation.LINEAR_ZERO, LongEndRule.HOLD_LEVEL)
    with pytest.raises(ValidationError):
        TermInterpolator([1.0], [0.01, 0.02], CurveInterpolation.LINEAR_ZERO, LongEndRule.HOLD_LEVEL)
    interp = make(CurveInterpolation.LINEAR_ZERO)
    with pytest.raises(ValidationError):
        interp.level(-0.1)
    with pytest.raises(ValidationError):
        interp.cumulative(np.array([0.5, -1.0]))
    with pytest.raises(ValidationError):
        TermInterpolator([1.0], [0.01], "linear", LongEndRule.HOLD_LEVEL)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TermInterpolator([1.0], [0.01], CurveInterpolation.LINEAR_ZERO, "hold")  # type: ignore[arg-type]
