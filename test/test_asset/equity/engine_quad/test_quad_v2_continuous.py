"""Continuous-barrier controls independent of the product recursion."""
import numpy as np
import pytest
from scipy.special import ndtr

from quantark.asset.equity.param import QuadV2Params
from quantark.asset.equity.engine.quad.v2.contract import affine
from quantark.asset.equity.engine.quad.v2.continuous import continuous_readout
from test_quad_v2_engine import snowball, environment
from quantark.asset.equity.engine.quad import SnowballQuadEngineV2


@pytest.mark.parametrize("mean", [-0.05, 0.0, 0.08])
@pytest.mark.parametrize("reverse", [False, True])
def test_survival_probability_and_complete_derivatives(mean, reverse):
    barrier = np.log(120.0 if reverse else 75.0)
    x = np.log([90.0, 100.0, 110.0])
    variance = 0.04
    sd = np.sqrt(variance)

    def survival(points):
        d = (barrier - points) if reverse else (points - barrier)
        m = -mean if reverse else mean
        return ndtr((d + m) / sd) - np.exp(-2 * m * d / variance) * ndtr((m - d) / sd)

    got = continuous_readout(
        affine([1.0]),
        affine([0.0]),
        x,
        mean,
        variance,
        0.97,
        barrier,
        reverse,
        QuadV2Params(),
    )[:, 0]
    np.testing.assert_allclose(got[0], 0.97 * survival(x), atol=2e-14)
    h = 2e-5
    expected1 = 0.97 * (survival(x + h) - survival(x - h)) / (2 * h)
    expected2 = 0.97 * (survival(x + h) - 2 * survival(x) + survival(x - h)) / h**2
    np.testing.assert_allclose(got[1], expected1, atol=2e-8)
    np.testing.assert_allclose(got[2], expected2, atol=3e-6)


def test_extreme_drift_does_not_overflow_reflection_multiplier():
    got = continuous_readout(
        affine([1.0]),
        affine([0.0]),
        np.log([100.0]),
        -0.37,
        0.005**2,
        1.0,
        np.log(75.0),
        False,
        QuadV2Params(),
    )
    assert np.isfinite(got).all()
    assert abs(got[0, 0, 0]) < 1e-12


@pytest.mark.parametrize("reverse", [False, True])
def test_continuous_boundary_has_same_undefined_greeks_in_scalar_and_batch(reverse):
    barrier = np.log(100.0)

    def evaluate(points):
        return continuous_readout(
            affine([1.0]),
            affine([0.0]),
            np.asarray(points),
            0.01,
            0.04,
            1.0,
            barrier,
            reverse,
            QuadV2Params(),
        )

    scalar = evaluate([barrier])[:, 0, 0]
    batch = evaluate([barrier, np.log(90.0), np.log(110.0)])[:, 0, 0]
    assert scalar[0] == 0.0
    assert np.isnan(scalar[1:]).all()
    np.testing.assert_allclose(scalar, batch, equal_nan=True)


def test_continuous_price_refines_and_greeks_match_bumps():
    p = snowball(times=(0.25, 0.5, 0.75, 1.0), continuous=True)
    e = environment()
    standard = SnowballQuadEngineV2().calculate_point_greeks(p, e)
    fine = SnowballQuadEngineV2(QuadV2Params(order=10, cells_per_sd=3.0))
    reference = fine.calculate_point_greeks(p, e)
    assert standard["price"] == pytest.approx(reference["price"], abs=1e-5)
    assert standard["delta"] == pytest.approx(reference["delta"], abs=1e-5)
    context = fine._prepared
    s, b = 100.0, 1e-4
    prices = context.evaluate([s * (1 - b), s, s * (1 + b)])["price"]
    assert (prices[2] - prices[0]) / (2 * s * b) == pytest.approx(
        reference["delta"], abs=2e-6
    )
    assert (prices[2] - 2 * prices[1] + prices[0]) / (s * b) ** 2 == pytest.approx(
        reference["gamma"], abs=2e-6
    )
