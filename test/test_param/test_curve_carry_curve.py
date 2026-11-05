"""CarryCurve and its dividend-yield view (spec 2026-11-05, T3).

Covers the curve-level parts of AC12 and AC5 for carry / yield.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.param import (
    CarryCurve,
    CarryImpliedDividendYield,
    CurveInterpolation,
    FlatRateCurve,
    TermStructureDividendYield,
    ZeroCurve,
)
from quantark.param.div.dividend_yield import DividendYield, ParallelShiftDividendYield
from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.param.extrapolation import CarryExtrapolation, RateExtrapolation
from quantark.param.node_roles import NodeRole
from quantark.util.exceptions import ValidationError

# B(T) = log F/S nodes: negative carry (q > r) at the front, positive later — generic
NODES = [(0.25, -0.0030), (0.5, -0.0050), (1.0, -0.0080), (2.0, -0.0120), (5.0, -0.0150)]
TIMES = np.array([n[0] for n in NODES])
SCHEMES = [s for s in CurveInterpolation if s is not CurveInterpolation.MONOTONE_CONVEX]
GRID = np.linspace(0.0, 8.0, 101)  # 0 .. beyond T_n = 5
RATE_PILLARS = [(0.25, 0.020), (1.0, 0.022), (2.0, 0.025), (5.0, 0.027), (10.0, 0.028)]


# --------------------------------------------------------------------------- construction


def test_exports_and_defaults():
    cc = CarryCurve(NODES)
    assert cc.interpolation is CurveInterpolation.LOG_LINEAR_DF
    assert cc.extrapolation is CarryExtrapolation.FLAT_FORWARD_CARRY
    assert cc.nodes == NODES and cc.tenors == list(TIMES) and cc.carries == [n[1] for n in NODES]
    assert cc.valuation_date is None and cc.node_roles is None
    assert cc.last_observable_tenor == 5.0  # defaults to the last node, like ForwardCarryCurve
    assert "log_linear_df" in repr(cc)


def test_metadata_sorting_and_copy():
    vd = datetime(2026, 11, 5)
    roles = (NodeRole.CALIBRATED,) * len(NODES)
    cc = CarryCurve(list(reversed(NODES)), node_roles=roles, last_observable_tenor=2.0, valuation_date=vd)
    assert cc.nodes == NODES
    assert cc.node_roles == roles and cc.last_observable_tenor == 2.0 and cc.valuation_date == vd
    cc.nodes.append((9.0, 0.0))
    assert len(cc.nodes) == len(NODES)


def test_validation():
    with pytest.raises(ValidationError):
        CarryCurve(NODES, interpolation=CurveInterpolation.MONOTONE_CONVEX)
    with pytest.raises(ValidationError):
        CarryCurve([])
    with pytest.raises(ValidationError):
        CarryCurve([(0.0, 0.0), (1.0, -0.01)])
    with pytest.raises(ValidationError):
        CarryCurve([(1.0, -0.01), (1.0, -0.02)])
    with pytest.raises(ValidationError):
        CarryCurve([(1.0, float("inf"))])
    with pytest.raises(ValidationError):
        CarryCurve(NODES, extrapolation=RateExtrapolation.FLAT_FORWARD_RATE)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CarryCurve(NODES, interpolation="linear_zero")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CarryCurve(NODES).carry(-0.1)


# --------------------------------------------------------------------------- AC12: ForwardCarryCurve equivalence


def test_default_carry_curve_equals_forward_carry_curve():
    cc = CarryCurve(NODES)
    ref = ForwardCarryCurve(NODES)
    for T in GRID:
        assert cc.carry(T) == pytest.approx(ref.carry(T), abs=1e-15)
        assert cc.forward(100.0, T) == pytest.approx(ref.forward(100.0, T), rel=1e-15)
    for t0, t1 in [(0.0, 0.25), (0.1, 0.4), (0.5, 0.5), (1.0, 1.0), (2.0, 7.0), (6.0, 6.0), (6.0, 7.5)]:
        assert cc.interval_carry(t0, t1) == pytest.approx(ref.interval_carry(t0, t1), abs=1e-15)
    with pytest.raises(ValidationError):
        cc.interval_carry(1.0, 0.5)


def test_zero_forward_carry_holds_b_flat():
    cc = CarryCurve(NODES, extrapolation=CarryExtrapolation.ZERO_FORWARD_CARRY)
    ref = ForwardCarryCurve(NODES, extrapolation=CarryExtrapolation.ZERO_FORWARD_CARRY)
    for T in [5.0, 5.5, 7.0, 20.0]:
        assert cc.carry(T) == pytest.approx(-0.0150, abs=1e-15)
        assert cc.carry(T) == pytest.approx(ref.carry(T), abs=1e-15)
    assert cc.forward_carry(6.0) == 0.0
    assert cc.interval_carry(6.0, 6.0) == 0.0  # differs from ForwardCarryCurve's last-segment slope by definition
    assert cc.interval_carry(6.0, 7.0) == 0.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_nodes_reproduced_every_scheme(scheme):
    cc = CarryCurve(NODES, interpolation=scheme)
    for t, b in NODES:
        assert cc.carry(t) == pytest.approx(b, abs=1e-16)
    assert cc.carry(0.0) == 0.0


def test_linear_zero_is_linear_in_yield_under_flat_rate():
    """LINEAR_ZERO interpolates the zero carry yield B/T linearly; with a flat r that is
    exactly TermStructureDividendYield's linear-in-q convention inside [T_1, T_n]."""
    r = 0.03
    rc = FlatRateCurve(r)
    cc = CarryCurve(NODES, interpolation=CurveInterpolation.LINEAR_ZERO)
    q_nodes = [r - b / t for t, b in NODES]
    ref = TermStructureDividendYield(times=list(TIMES), yields=q_nodes)
    view = cc.to_dividend_yield(rc)
    for T in np.linspace(0.25, 5.0, 50):
        assert view.get_yield(T) == pytest.approx(ref.get_yield(T), abs=1e-15)


# --------------------------------------------------------------------------- the dividend-yield view


@pytest.mark.parametrize("rate_curve", [FlatRateCurve(0.03), ZeroCurve(RATE_PILLARS)])
def test_view_is_pointwise_exact_and_matches_legacy_view(rate_curve):
    cc = CarryCurve(NODES)
    view = cc.to_dividend_yield(rate_curve)
    assert isinstance(view, DividendYield) and isinstance(view, CarryImpliedDividendYield)
    legacy = ForwardCarryCurve(NODES).to_dividend_yield(rate_curve)
    for T in GRID[1:]:
        expected = (rate_curve.get_rate(float(T)) * T - cc.carry(T)) / T
        assert view.get_yield(float(T)) == pytest.approx(expected, abs=1e-15)
        assert view.get_yield(float(T)) == pytest.approx(legacy.get_yield(float(T)), abs=1e-15)
    for t, b in NODES:
        assert view.get_yield(t) == pytest.approx(rate_curve.get_rate(t) - b / t, abs=1e-15)
    with pytest.raises(ValidationError):
        view.get_yield(0.0)
    with pytest.raises(ValidationError):
        view.get_yield(-1.0)
    arr = view.get_yield(np.array([0.0, 0.5, 2.0, 7.0]))
    assert arr[0] == 0.0  # array entries at T <= 0 map to 0.0 (grid samplers weight by q*T)
    assert view.node_roles is cc.node_roles and view.last_observable_tenor == cc.last_observable_tenor
    assert view.carry_curve is cc and view.rate_curve is rate_curve


@pytest.mark.parametrize("rate_curve", [FlatRateCurve(0.03), ZeroCurve(RATE_PILLARS)])
@pytest.mark.parametrize("scheme", SCHEMES)
def test_arrays_equal_scalar_calls_exactly(rate_curve, scheme):
    cc = CarryCurve(NODES, interpolation=scheme)
    grid = np.array([0.0, 0.1, 0.25, 0.4, 1.0, 1.7, 5.0, 6.0, 12.0])
    carries = cc.carry(grid)
    fwd = cc.forward_carry(grid)
    assert isinstance(carries, np.ndarray) and carries.shape == grid.shape
    for t, b, f in zip(grid, carries, fwd):
        assert cc.carry(float(t)) == b
        assert cc.forward_carry(float(t)) == f
    view = cc.to_dividend_yield(rate_curve)
    ys = view.get_yield(grid)
    assert isinstance(ys, np.ndarray) and ys.shape == grid.shape
    for t, y in zip(grid[1:], ys[1:]):
        assert view.get_yield(float(t)) == y
    assert isinstance(cc.carry(1.0), float) and isinstance(view.get_yield(1.0), float)
    assert view.get_yield(grid.reshape(3, 3)).shape == (3, 3)


class _ScalarOnlyCurve(FlatRateCurve):
    """A rate curve whose get_rate takes floats only (array dispatch raises TypeError)."""

    def get_rate(self, time_to_maturity):
        if np.ndim(time_to_maturity) != 0:
            raise TypeError("scalar only")
        return 0.03 + 0.001 * float(time_to_maturity)


class _BrokenCurve(FlatRateCurve):
    def get_rate(self, time_to_maturity):
        if np.ndim(time_to_maturity) != 0:
            raise RuntimeError("genuine failure")
        return 0.03


def test_to_dividend_yield_refuses_two_different_anchors():
    from datetime import datetime

    dated = CarryCurve(NODES, valuation_date=datetime(2026, 11, 5))
    rc_same = ZeroCurve(RATE_PILLARS, valuation_date=datetime(2026, 11, 5))
    rc_other = ZeroCurve(RATE_PILLARS, valuation_date=datetime(2026, 11, 6))
    dated.to_dividend_yield(rc_same)            # same anchor: fine
    dated.to_dividend_yield(FlatRateCurve(0.03))  # attribute-less curve: passes
    CarryCurve(NODES).to_dividend_yield(rc_other)  # numeric carry curve: passes
    with pytest.raises(ValidationError, match="one clock per run"):
        dated.to_dividend_yield(rc_other)


def test_view_scalar_only_rate_curve_falls_back_per_element_and_real_errors_propagate():
    # CP2 #5: only a dispatch failure (TypeError/ValueError) triggers the per-element
    # fallback, which is then bit-identical to the scalar calls; anything else propagates
    cc = CarryCurve(NODES)
    grid = np.array([0.5, 1.0, 1.7, 5.0])
    view = cc.to_dividend_yield(_ScalarOnlyCurve(0.03))
    ys = view.get_yield(grid)
    for t, y in zip(grid, ys):
        assert view.get_yield(float(t)) == y
    with pytest.raises(RuntimeError, match="genuine failure"):
        cc.to_dividend_yield(_BrokenCurve(0.03)).get_yield(grid)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_view_parallel_shift_flat_forward_carry_is_node_shift(scheme):
    rc = ZeroCurve(RATE_PILLARS)
    vd = datetime(2026, 11, 5)
    cc = CarryCurve(NODES, interpolation=scheme, valuation_date=vd)
    view = cc.to_dividend_yield(rc)
    s = 0.004
    shifted = view.parallel_shifted(s)
    assert isinstance(shifted, CarryImpliedDividendYield)
    assert shifted.carry_curve.valuation_date == vd and shifted.carry_curve.interpolation is scheme
    assert shifted.carry_curve.nodes == pytest.approx([(t, b - s * t) for t, b in NODES])
    for T in GRID[1:]:
        assert shifted.get_yield(float(T)) == pytest.approx(view.get_yield(float(T)) + s, abs=1e-14)


def test_view_parallel_shift_zero_forward_carry_wraps():
    rc = FlatRateCurve(0.03)
    cc = CarryCurve(NODES, extrapolation=CarryExtrapolation.ZERO_FORWARD_CARRY)
    view = cc.to_dividend_yield(rc)
    shifted = view.parallel_shifted(0.004)
    assert isinstance(shifted, ParallelShiftDividendYield)
    for T in GRID[1:]:
        assert shifted.get_yield(float(T)) == pytest.approx(view.get_yield(float(T)) + 0.004, abs=1e-14)
    # the node shift would NOT be exact here: beyond T_n the flat tail cannot carry q + s
    naive = cc.shifted_yield(0.004).to_dividend_yield(rc)
    assert naive.get_yield(7.0) != pytest.approx(view.get_yield(7.0) + 0.004, abs=1e-6)


def test_with_nodes_and_shifted_yield_keep_configuration():
    vd = datetime(2026, 11, 5)
    cc = CarryCurve(NODES, interpolation=CurveInterpolation.LOG_CUBIC_DF,
                    extrapolation=CarryExtrapolation.ZERO_FORWARD_CARRY, last_observable_tenor=2.0,
                    valuation_date=vd)
    new = cc.with_nodes([(1.0, -0.01), (3.0, -0.02)])
    assert new.nodes == [(1.0, -0.01), (3.0, -0.02)]
    assert new.interpolation is CurveInterpolation.LOG_CUBIC_DF
    assert new.extrapolation is CarryExtrapolation.ZERO_FORWARD_CARRY
    assert new.last_observable_tenor == 2.0 and new.valuation_date == vd
    sh = cc.shifted_yield(0.01)
    assert sh.nodes == pytest.approx([(t, b - 0.01 * t) for t, b in NODES])
    assert sh.interpolation is cc.interpolation and sh.valuation_date == vd
    assert cc.nodes == NODES


# --------------------------------------------------------------------------- FLAT_CARRY_YIELD tail
# spec .ship/2026-11-05-snowball-2y-carry-curve AC1-AC3


def test_flat_carry_yield_maps_to_hold_level():
    from quantark.param.curve.interpolation import LongEndRule, long_end_rule

    assert long_end_rule(CarryExtrapolation.FLAT_CARRY_YIELD) is LongEndRule.HOLD_LEVEL


@pytest.mark.parametrize("scheme", SCHEMES)
def test_flat_carry_yield_holds_last_zero_carry_yield(scheme):
    cc = CarryCurve(NODES, interpolation=scheme, extrapolation=CarryExtrapolation.FLAT_CARRY_YIELD)
    t_n, b_n = NODES[-1]
    for T in (t_n, 1.5 * t_n, 3.0 * t_n):
        assert float(cc.carry(T)) / T == pytest.approx(b_n / t_n, rel=1e-14, abs=1e-16)
    eps = 1e-9
    assert float(cc.carry(t_n + eps)) == pytest.approx(float(cc.carry(t_n)), abs=1e-10)
    # interior (incl. the short end) identical to the default tail, bitwise
    ref = CarryCurve(NODES, interpolation=scheme)
    interior = GRID[GRID <= t_n]
    np.testing.assert_array_equal(cc.carry(interior), ref.carry(interior))


@pytest.mark.parametrize("scheme", SCHEMES)
def test_view_parallel_shift_flat_carry_yield_is_exact_node_shift(scheme):
    cc = CarryCurve(NODES, interpolation=scheme, extrapolation=CarryExtrapolation.FLAT_CARRY_YIELD)
    view = cc.to_dividend_yield(FlatRateCurve(0.02))
    shifted = view.parallel_shifted(0.0123)
    assert isinstance(shifted, CarryImpliedDividendYield)
    ts = np.array([0.1, 0.25, 0.7, 2.0, 5.0, 6.5, 12.0])
    np.testing.assert_allclose(shifted.get_yield(ts), view.get_yield(ts) + 0.0123, rtol=0.0, atol=1e-14)


def test_legacy_forward_carry_curve_rejects_flat_carry_yield():
    with pytest.raises(ValidationError):
        ForwardCarryCurve(NODES, extrapolation=CarryExtrapolation.FLAT_CARRY_YIELD)
