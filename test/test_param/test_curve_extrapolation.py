"""Long-end extrapolation schemes (spec WP3.5)."""
import pytest

from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.param.extrapolation import (
    CarryExtrapolation,
    RateExtrapolation,
    VolExtrapolation,
    extrapolated_total_variance,
    extrapolated_zero_rate,
    extrapolation_scheme_risk,
)

NODES = [(0.5, -0.05), (1.0, -0.12)]


def test_carry_schemes_continuous_and_distinct():
    flat_fwd = ForwardCarryCurve(NODES)  # default FLAT_FORWARD_CARRY
    zero_fwd = ForwardCarryCurve(
        NODES, extrapolation=CarryExtrapolation.ZERO_FORWARD_CARRY
    )
    # continuity at the last observable node
    assert flat_fwd.carry(1.0) == zero_fwd.carry(1.0) == pytest.approx(-0.12)
    # ZERO_FORWARD_CARRY: B flat beyond
    assert zero_fwd.carry(2.0) == pytest.approx(-0.12)
    # FLAT_FORWARD_CARRY: last segment slope (-0.12+0.05)/0.5 = -0.14 continues
    assert flat_fwd.carry(2.0) == pytest.approx(-0.12 + (-0.14) * 1.0)


def test_rate_schemes():
    from quantark.param.rrf.rate_curve import LinearRateCurve

    c = LinearRateCurve([(0.5, 0.03), (1.0, 0.04)])
    assert extrapolated_zero_rate(
        c, 2.0, RateExtrapolation.FLAT_ZERO_RATE
    ) == pytest.approx(0.04)
    # last forward over [0.5,1.0] = (0.04*1 - 0.03*0.5)/0.5 = 0.05
    # r(2)*2 = 0.04*1 + 0.05*1 -> r(2) = 0.045
    assert extrapolated_zero_rate(
        c, 2.0, RateExtrapolation.FLAT_FORWARD_RATE
    ) == pytest.approx(0.045)
    # inside the pillar range both schemes delegate to the curve
    assert extrapolated_zero_rate(
        c, 0.75, RateExtrapolation.FLAT_FORWARD_RATE
    ) == pytest.approx(c.get_rate(0.75))


def test_vol_schemes():
    class _StubSurface:
        def get_vol(self, strike, t):
            return {0.5: 0.18, 1.0: 0.20}[round(t, 6)]

    v_flat_iv = extrapolated_total_variance(
        _StubSurface(), 6000.0, 2.0, VolExtrapolation.FLAT_TOTAL_IMPLIED_VOL,
        last_observable_tenor=1.0,
    )
    assert v_flat_iv == pytest.approx(0.20 ** 2 * 2.0)

    # forward var over [0.5, 1.0]: (0.04*1 - 0.0324*0.5)/0.5 = 0.0476
    v_flat_fwd = extrapolated_total_variance(
        _StubSurface(), 6000.0, 2.0, VolExtrapolation.FLAT_FORWARD_VOL,
        last_observable_tenor=1.0, penultimate_tenor=0.5,
    )
    assert v_flat_fwd == pytest.approx(0.04 + 0.0476 * 1.0)


def test_scheme_risk_line_item():
    out = extrapolation_scheme_risk(
        price_fn=lambda env: {"a": 100.0, "b": 97.5}[env],
        default_env="a", alt_env="b",
    )
    assert out == {
        "pv_default": 100.0, "pv_alternative": 97.5, "scheme_risk": -2.5,
    }
