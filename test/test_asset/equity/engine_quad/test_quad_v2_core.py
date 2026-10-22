"""Independent analytic and algebraic controls for QUAD V2."""
import numpy as np
import pytest
from scipy.stats import norm
from quantark.util.exceptions import ValidationError

from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.engine.quad.v2.basis import Mesh
from quantark.asset.equity.engine.quad.v2.contract import Piece, Function, affine
from quantark.asset.equity.engine.quad.v2.gaussian import readout
from quantark.asset.equity.engine.quad.v2.operator import Operator, ByteCache
from quantark.asset.equity.engine.quad.v2.direct import apply_direct
from quantark.asset.equity.engine.quad.v2.fft import apply_fft


@pytest.mark.parametrize("vol", [0.005, 0.2, 0.6])
@pytest.mark.parametrize("maturity", [0.0001, 0.5, 2.0])
def test_exact_terminal_put_and_greeks(vol, maturity):
    r, q = 0.03, 0.01
    spots = np.array([75.0, 100.0, 125.0])
    function = Function(
        (
            Piece(-np.inf, np.log(100.0), np.array([100.0]), np.array([-1.0])),
            Piece(np.log(100.0), np.inf, np.zeros(1), np.zeros(1)),
        )
    )
    v = vol**2 * maturity
    result = readout(
        function,
        np.log(spots),
        (r - q) * maturity - v / 2,
        v,
        np.exp(-r * maturity),
        QuadV2Params(),
    )
    d1 = (np.log(spots / 100) + (r - q) * maturity + v / 2) / np.sqrt(v)
    price = 100 * np.exp(-r * maturity) * norm.cdf(-d1 + np.sqrt(v)) - spots * np.exp(
        -q * maturity
    ) * norm.cdf(-d1)
    delta = -np.exp(-q * maturity) * norm.cdf(-d1)
    gamma = np.exp(-q * maturity) * norm.pdf(d1) / (spots * np.sqrt(v))
    np.testing.assert_allclose(result[0, 0], price, atol=1e-11)
    np.testing.assert_allclose(result[1, 0] / spots, delta, atol=1e-11)
    np.testing.assert_allclose(
        (result[2, 0] - result[1, 0]) / spots**2, gamma, atol=2e-10
    )


@pytest.mark.parametrize("cells,band", [(32, 3), (120, 9), (200, 35)])
def test_direct_fft_equivalence(cells, band):
    rng = np.random.default_rng(2718)
    values = rng.normal(size=(3, cells, 6))
    kernel = rng.normal(size=(2 * band + 1, 6, 6))
    direct = apply_direct(kernel, values, 200_000)
    fft = apply_fft(
        kernel, values, 20_000_000, ByteCache(100_000), ("test", cells, band)
    )
    np.testing.assert_allclose(fft, direct, atol=2e-12, rtol=2e-12)


def test_zero_variance_affine():
    result = readout(
        affine([7.0], [2.0]), np.log([80.0, 100.0]), 0.2, 0.0, 0.97, QuadV2Params()
    )
    expected = 0.97 * 2 * np.array([80.0, 100.0]) * np.exp(0.2)
    np.testing.assert_allclose(result[0, 0], expected + 0.97 * 7)
    np.testing.assert_allclose(result[1, 0], expected)
    np.testing.assert_allclose(result[2, 0], expected)


def test_discounted_cash_grid_moment():
    mesh = Mesh(1.0, 0.015, 180, 8)
    from quantark.asset.equity.engine.quad.v2.contract import grid_function

    values = np.full((1, mesh.cells, mesh.order), 5.0)
    op = Operator(mesh, QuadV2Params())
    result = op.integrate(grid_function(mesh, values), 0.003, 0.01, 0.98)
    np.testing.assert_allclose(result, 4.9, atol=5e-13)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": 2},
        {"backend": "bad"},
        {"tail_sd": 2},
        {"cells_per_sd": 0},
        {"max_nodes": True},
        {"readout_order": 10000},
    ],
)
def test_params_reject_invalid_inputs(kwargs):
    with pytest.raises(ValidationError):
        QuadV2Params(**kwargs)
