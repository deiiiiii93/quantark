"""ForwardCarryCurve tests (spec WP3.1)."""
import numpy as np
import pytest

from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.util.exceptions import ValidationError

NODES = [(0.25, -0.02), (0.5, -0.045), (1.0, -0.10)]  # B(T), q >> r style


def test_carry_interpolation_linear_in_B():
    c = ForwardCarryCurve(NODES)
    assert c.carry(0.25) == pytest.approx(-0.02)
    assert c.carry(0.375) == pytest.approx((-0.02 + -0.045) / 2.0)
    assert c.carry(0.0) == 0.0


def test_forward():
    c = ForwardCarryCurve(NODES)
    assert c.forward(6000.0, 1.0) == pytest.approx(6000.0 * np.exp(-0.10))


def test_interval_carry_and_degenerate_point():
    c = ForwardCarryCurve(NODES)
    assert c.interval_carry(0.25, 0.5) == pytest.approx((-0.045 + 0.02) / 0.25)
    # t1 == t0 -> right-segment slope dB/dt+ (spec convention)
    assert c.interval_carry(0.25, 0.25) == pytest.approx((-0.045 + 0.02) / 0.25)


def test_to_dividend_yield_node_exact():
    # invariant: q(T)*T = r(T)*T - B(T) at every node
    from quantark.param import FlatRateCurve
    r = 0.0356
    c = ForwardCarryCurve(NODES)
    dy = c.to_dividend_yield(FlatRateCurve(rate=r))
    for T, B in NODES:
        assert dy.get_yield(T) * T == pytest.approx(r * T - B, abs=1e-12)


def test_from_forward_nodes_round_trip():
    s0 = 6000.0
    fwd_nodes = [(0.5, 5800.0), (1.0, 5500.0)]
    c = ForwardCarryCurve.from_forward_nodes(s0, fwd_nodes)
    for T, F in fwd_nodes:
        assert c.forward(s0, T) == pytest.approx(F)


def test_from_index_futures_consistent_with_implied_yields():
    from quantark.asset.equity.market import (
        IndexFuturesCurve, IndexFuturesQuote,
    )
    from quantark.param import FlatRateCurve

    rate = FlatRateCurve(rate=0.0356)
    curve = IndexFuturesCurve(
        underlying="000852.SH",
        spot=6000.0,
        quotes=[
            IndexFuturesQuote(contract="IM2303", maturity=0.25,
                              price=5850.0, multiplier=200.0),
            IndexFuturesQuote(contract="IM2306", maturity=0.5,
                              price=5700.0, multiplier=200.0),
        ],
    )
    carry = ForwardCarryCurve.from_index_futures(curve, rate)
    # B(T) = ln(F/S0) exactly (implied-yield algebra round-trips)
    assert carry.carry(0.25) == pytest.approx(np.log(5850.0 / 6000.0))
    assert carry.carry(0.5) == pytest.approx(np.log(5700.0 / 6000.0))


def test_rejects_unsorted_or_nonpositive_tenors():
    with pytest.raises(ValidationError):
        ForwardCarryCurve([(0.5, -0.01), (0.25, -0.02)])
    with pytest.raises(ValidationError):
        ForwardCarryCurve([(0.0, 0.0), (0.5, -0.01)])
