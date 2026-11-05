"""ZeroCurve: configurable-interpolation zero curve (spec 2026-11-05, T2).

Covers AC1, AC3, AC4, AC5 (curve) and the trading-clock half of AC13.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.param import CurveInterpolation, ZeroCurve
from quantark.param.curve import ZeroCurve as ZeroCurveFromCurve
from quantark.param.extrapolation import (
    CarryExtrapolation,
    RateExtrapolation,
    extrapolated_zero_rate,
)
from quantark.param.node_roles import NodeRole
from quantark.param.rrf.key_rate import key_rate_bumped_zero_curve
from quantark.param.rrf.rate_curve import (
    LinearRateCurve,
    ParallelShiftRateCurve,
    RateCurve,
)
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError

PILLARS = [(0.25, 0.030), (0.5, 0.031), (1.0, 0.033), (2.0, 0.0345), (5.0, 0.036), (10.0, 0.035)]
SCHEMES = list(CurveInterpolation)
LINEAR_OPERATOR_SCHEMES = [s for s in SCHEMES if s is not CurveInterpolation.MONOTONE_CONVEX]
EXTRAPS = list(RateExtrapolation)


def test_exports():
    assert ZeroCurve is ZeroCurveFromCurve
    assert issubclass(ZeroCurve, RateCurve)


# --------------------------------------------------------------------------- AC1


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("extrap", EXTRAPS)
def test_pillars_reproduced(scheme, extrap):
    zc = ZeroCurve(PILLARS, interpolation=scheme, extrapolation=extrap)
    for t, z in PILLARS:
        assert zc.get_rate(t) == pytest.approx(z, abs=1e-14)
        assert zc.get_discount_factor(t) == pytest.approx(np.exp(-z * t), rel=1e-15)
    assert zc.tenors == [p[0] for p in PILLARS]
    assert zc.rates == [p[1] for p in PILLARS]
    assert zc.pillars == PILLARS
    assert zc.interpolation is scheme and zc.extrapolation is extrap


def test_defaults_and_metadata():
    zc = ZeroCurve(PILLARS)
    assert zc.interpolation is CurveInterpolation.LOG_LINEAR_DF
    assert zc.extrapolation is RateExtrapolation.FLAT_FORWARD_RATE
    assert zc.node_roles is None and zc.last_observable_tenor is None and zc.valuation_date is None
    roles = (NodeRole.CALIBRATED,) * len(PILLARS)
    vd = datetime(2026, 11, 5)
    zc2 = ZeroCurve(PILLARS, node_roles=roles, last_observable_tenor=10.0, valuation_date=vd)
    assert zc2.node_roles == roles and zc2.last_observable_tenor == 10.0 and zc2.valuation_date == vd
    assert "log_linear_df" in repr(zc) and "flat_forward_rate" in repr(zc)


def test_unsorted_pillars_are_sorted_and_copied():
    shuffled = [PILLARS[3], PILLARS[0], PILLARS[5], PILLARS[1], PILLARS[4], PILLARS[2]]
    zc = ZeroCurve(shuffled)
    assert zc.pillars == PILLARS
    zc.pillars.append((20.0, 0.01))  # a copy: the curve is immutable
    assert len(zc.pillars) == len(PILLARS)


# --------------------------------------------------------------------------- AC3


@pytest.mark.parametrize("scheme", SCHEMES)
def test_long_end_flat_zero_rate(scheme):
    zc = ZeroCurve(PILLARS, interpolation=scheme, extrapolation=RateExtrapolation.FLAT_ZERO_RATE)
    for T in [10.0 + 1e-12, 12.0, 25.0]:
        assert zc.get_rate(T) == pytest.approx(0.035, abs=1e-15)
        assert zc.get_rate(T) == pytest.approx(
            extrapolated_zero_rate(zc, T, RateExtrapolation.FLAT_ZERO_RATE), abs=1e-14
        )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_long_end_flat_forward_rate_matches_helper(scheme):
    zc = ZeroCurve(PILLARS, interpolation=scheme, extrapolation=RateExtrapolation.FLAT_FORWARD_RATE)
    for T in [10.0 + 1e-12, 12.0, 25.0]:
        assert zc.get_rate(T) == pytest.approx(
            extrapolated_zero_rate(zc, T, RateExtrapolation.FLAT_FORWARD_RATE), abs=1e-14
        )
    # continuity at T_n and the last discrete forward beyond it
    assert zc.get_rate(10.0) == pytest.approx(0.035, abs=1e-14)
    fd_last = (0.035 * 10.0 - 0.036 * 5.0) / 5.0
    assert zc.get_forward_rate(12.0, 15.0) == pytest.approx(fd_last, abs=1e-13)


@pytest.mark.parametrize("scheme", [CurveInterpolation.LINEAR_ZERO, CurveInterpolation.CUBIC_ZERO])
def test_short_end_flat_level(scheme):
    zc = ZeroCurve(PILLARS, interpolation=scheme)
    for t in [0.0, 0.1, 0.25]:
        assert zc.get_rate(t) == pytest.approx(0.030, abs=1e-15)


def test_short_end_log_linear_flat_forward():
    zc = ZeroCurve(PILLARS, interpolation=CurveInterpolation.LOG_LINEAR_DF)
    assert zc.get_forward_rate(0.0, 0.1) == pytest.approx(0.030, abs=1e-14)
    assert zc.get_forward_rate(0.1, 0.25) == pytest.approx(0.030, abs=1e-14)


@pytest.mark.parametrize("scheme", [CurveInterpolation.LOG_CUBIC_DF, CurveInterpolation.MONOTONE_CONVEX])
def test_short_end_average_forward_and_rate_at_zero_convention(scheme):
    zc = ZeroCurve(PILLARS, interpolation=scheme)
    assert zc.get_rate(0.25) == pytest.approx(0.030, abs=1e-15)
    assert zc.get_rate(0.0) == 0.030  # convention: first pillar level


@pytest.mark.parametrize("scheme", SCHEMES)
def test_time_zero_and_negative(scheme):
    zc = ZeroCurve(PILLARS, interpolation=scheme)
    assert zc.get_discount_factor(0.0) == 1.0
    with pytest.raises(ValidationError):
        zc.get_rate(-1e-9)
    with pytest.raises(ValidationError):
        zc.get_discount_factor(-0.5)


def test_validation():
    with pytest.raises(ValidationError):
        ZeroCurve([])
    with pytest.raises(ValidationError):
        ZeroCurve([(0.0, 0.02), (1.0, 0.03)])  # DF(0) = 1 is implicit
    with pytest.raises(ValidationError):
        ZeroCurve([(1.0, 0.02), (1.0, 0.03)])
    with pytest.raises(ValidationError):
        ZeroCurve([(1.0, float("nan"))])
    with pytest.raises(ValidationError):
        ZeroCurve([(-1.0, 0.02)])
    with pytest.raises(ValidationError):
        ZeroCurve(PILLARS, interpolation="log_linear_df")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ZeroCurve(PILLARS, extrapolation=CarryExtrapolation.FLAT_FORWARD_CARRY)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- AC4


@pytest.mark.parametrize("scheme", LINEAR_OPERATOR_SCHEMES)
@pytest.mark.parametrize("extrap", EXTRAPS)
def test_parallel_shift_exact_pillar_shift(scheme, extrap):
    vd = datetime(2026, 11, 5)
    roles = (NodeRole.CALIBRATED,) * len(PILLARS)
    base = ZeroCurve(PILLARS, interpolation=scheme, extrapolation=extrap, node_roles=roles,
                     last_observable_tenor=10.0, valuation_date=vd)
    s = 0.0025
    up = base.parallel_shifted(s)
    assert type(up) is ZeroCurve
    assert up.interpolation is scheme and up.extrapolation is extrap
    assert up.node_roles == roles and up.last_observable_tenor == 10.0 and up.valuation_date == vd
    assert up.rates == pytest.approx([z + s for z in base.rates], abs=1e-18)
    for T in np.linspace(0.0, 20.0, 101):
        assert up.get_discount_factor(T) == pytest.approx(
            base.get_discount_factor(T) * np.exp(-s * T), abs=1e-14
        )
    down = base.parallel_shifted(-s)
    for T in [0.3, 3.0, 15.0]:
        assert down.get_rate(T) == pytest.approx(base.get_rate(T) - s, abs=1e-14)


def test_parallel_shift_monotone_convex_wraps():
    base = ZeroCurve(PILLARS, interpolation=CurveInterpolation.MONOTONE_CONVEX)
    up = base.parallel_shifted(0.001)
    assert isinstance(up, ParallelShiftRateCurve)
    for T in [0.1, 0.7, 3.3, 12.0]:
        assert up.get_discount_factor(T) == pytest.approx(
            base.get_discount_factor(T) * np.exp(-0.001 * T), rel=1e-15
        )


def test_with_pillars_keeps_configuration():
    vd = datetime(2026, 11, 5)
    base = ZeroCurve(PILLARS, interpolation=CurveInterpolation.CUBIC_ZERO,
                     extrapolation=RateExtrapolation.FLAT_ZERO_RATE, last_observable_tenor=7.0,
                     valuation_date=vd)
    new = base.with_pillars([(0.5, 0.02), (3.0, 0.025)])
    assert new.pillars == [(0.5, 0.02), (3.0, 0.025)]
    assert new.interpolation is CurveInterpolation.CUBIC_ZERO
    assert new.extrapolation is RateExtrapolation.FLAT_ZERO_RATE
    assert new.last_observable_tenor == 7.0 and new.valuation_date == vd
    assert base.pillars == PILLARS  # untouched


@pytest.mark.parametrize("scheme", SCHEMES)
def test_key_rate_bump_returns_configured_zero_curve(scheme):
    vd = datetime(2026, 11, 5)
    base = ZeroCurve(PILLARS, interpolation=scheme, extrapolation=RateExtrapolation.FLAT_ZERO_RATE,
                     valuation_date=vd)
    bumped = key_rate_bumped_zero_curve(base, 2.0, 1e-4)
    assert type(bumped) is ZeroCurve
    assert bumped.interpolation is scheme and bumped.extrapolation is RateExtrapolation.FLAT_ZERO_RATE
    assert bumped.valuation_date == vd
    expected = [(t, z + 1e-4 if t == 2.0 else z) for t, z in PILLARS]
    assert bumped.pillars == pytest.approx(expected)
    assert bumped.get_rate(2.0) == pytest.approx(0.0345 + 1e-4, abs=1e-14)
    assert bumped.get_rate(0.25) == pytest.approx(0.030, abs=1e-14)  # other pillars untouched
    with pytest.raises(ValidationError):
        key_rate_bumped_zero_curve(base, 2.5, 1e-4)


def test_key_rate_bump_old_classes_unchanged():
    old = LinearRateCurve(PILLARS)
    bumped = key_rate_bumped_zero_curve(old, 1.0, 1e-4)
    assert type(bumped) is LinearRateCurve
    assert bumped.pillars[2] == (1.0, 0.033 + 1e-4)
    assert bumped.get_rate(0.75) == pytest.approx(0.032 + 0.5e-4, abs=1e-15)


# --------------------------------------------------------------------------- AC5


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("extrap", EXTRAPS)
def test_arrays_equal_scalar_calls_exactly(scheme, extrap):
    zc = ZeroCurve(PILLARS, interpolation=scheme, extrapolation=extrap)
    grid = np.array([0.0, 0.1, 0.25, 0.6, 1.0, 1.9, 5.0, 9.99, 10.0, 11.0, 30.0])
    rates = zc.get_rate(grid)
    dfs = zc.get_discount_factor(grid)
    assert isinstance(rates, np.ndarray) and isinstance(dfs, np.ndarray)
    for t, r, d in zip(grid, rates, dfs):
        assert zc.get_rate(float(t)) == r
        assert zc.get_discount_factor(float(t)) == d
    assert isinstance(zc.get_rate(1.0), float) and isinstance(zc.get_discount_factor(1.0), float)
    assert zc.get_discount_factor(grid.reshape(1, -1)).shape == (1, grid.size)


# --------------------------------------------------------------------------- AC13 (trading clock)


def test_trading_clock_wrapper_composes():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    anchor = datetime(2026, 11, 5)
    tmap = BusinessTimeMap(TradingClock(cal, 244), anchor, datetime(2027, 11, 5))
    zc = ZeroCurve(PILLARS, interpolation=CurveInterpolation.LOG_CUBIC_DF)
    wrapped = TradingClockRateCurve(zc, tmap)
    for u in [0.1, 0.25, 0.5, 0.9]:
        assert wrapped.get_discount_factor(u) == zc.get_discount_factor(tmap.to_calendar(u))
    shifted = wrapped.parallel_shifted(1e-4)
    assert isinstance(shifted, TradingClockRateCurve) and type(shifted.inner) is ZeroCurve


# --------------------------------------------------------------------------- forwards


def test_forward_rate_log_linear_is_piecewise_constant():
    zc = ZeroCurve(PILLARS, interpolation=CurveInterpolation.LOG_LINEAR_DF)
    fd_3 = (0.033 * 1.0 - 0.031 * 0.5) / 0.5
    assert zc.get_forward_rate(0.5, 0.75) == pytest.approx(fd_3, abs=1e-14)
    assert zc.get_forward_rate(0.6, 1.0) == pytest.approx(fd_3, abs=1e-14)
    assert zc.instantaneous_forward(0.7) == pytest.approx(fd_3, abs=1e-15)
    assert zc.instantaneous_forward(np.array([0.7, 0.8])).shape == (2,)
