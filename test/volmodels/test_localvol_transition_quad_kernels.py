"""Local-vol kernels A (CTMC + exact exponential) and B (Gauss-Richardson): identities, refusals, PDE agreement."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import norm

from example.case_studies.localvol_transition_quad import adapters
from example.case_studies.localvol_transition_quad.contract import (snowball_contract, up_out_contract,
                                                                    vanilla_contract)
from example.case_studies.localvol_transition_quad.engine import (IDENTITY_FRACTION, OUTPUTS, PV_BUDGET,
                                                                  LocalVolQuad, call_from_put)
from example.case_studies.localvol_transition_quad.grid import GridConfig, UnresolvedGrid, build_grid
from example.case_studies.localvol_transition_quad.kernels import CTMCKernel, GaussRichardsonKernel
from example.case_studies.localvol_transition_quad.market import Market, constant_surface, load_fixture

S0, R, Q, SIGMA = 100.0, 0.03, 0.01, 0.25
MARKET = Market(S0, R, Q, constant_surface(SIGMA))
FLOORS = {"pv": PV_BUDGET, "delta": 1e-3, "gamma": 1e-2, "vega": 1e-4, "theta": 1e-5, "rho": 1e-5, "carry": 1e-5}


def budget(key, ref):
    return PV_BUDGET if key == "pv" else max(0.01 * abs(ref), FLOORS[key])


def bs(option, k, t):
    f = S0 * math.exp((R - Q) * t)
    d1 = (math.log(f / k) + 0.5 * SIGMA ** 2 * t) / (SIGMA * math.sqrt(t))
    d2 = d1 - SIGMA * math.sqrt(t)
    sign = 1 if option == "call" else -1
    return math.exp(-R * t) * sign * (f * norm.cdf(sign * d1) - k * norm.cdf(sign * d2))


@pytest.mark.parametrize("backend", ["contour", "chebyshev"])
@pytest.mark.parametrize("dt", [1 / 252, 1 / 12])
def test_exponential_backends_match_dense_expm(backend, dt):
    m = load_fixture()
    c = snowball_contract(m.spot, 1.0, 12)
    grid = build_grid(c, 0.3, GridConfig(cells=150, half_width_std=6.0))
    kernel = CTMCKernel(backend=backend)
    up, down = kernel.rates(grid, m, 0.01)
    q = np.diag(-(up + down)) + np.diag(up[:-1], 1) + np.diag(down[1:], -1)
    v = np.vstack((np.ones(grid.size), np.exp(grid.centres) / m.spot))
    np.testing.assert_allclose(kernel.expm_apply(v, up, down, dt), v @ expm(dt * q).T, rtol=0, atol=1e-12)


def test_ctmc_rates_match_forward_and_variance():
    m = load_fixture()
    c = snowball_contract(m.spot, 1.0, 12)
    grid = build_grid(c, 0.3, GridConfig(cells=300))
    up, down = CTMCKernel().rates(grid, m, 0.3)
    x = grid.centres
    hp, hm = np.diff(x)[1:], np.diff(x)[:-1]
    sigma2 = m.surface.local_vol(np.exp(x[1:-1]), 0.3) ** 2
    np.testing.assert_allclose(up[1:-1] * np.expm1(hp) + down[1:-1] * np.expm1(-hm), m.r - m.q, atol=1e-10)
    np.testing.assert_allclose(up[1:-1] * hp ** 2 + down[1:-1] * hm ** 2, sigma2, rtol=1e-12)


@pytest.mark.parametrize("kernel", [CTMCKernel, lambda: GaussRichardsonKernel(1)], ids=["ctmc", "gauss"])
@pytest.mark.parametrize("option", ["put", "call"])
def test_ac1_vanilla_black_scholes(kernel, option):
    c = vanilla_contract(S0, 1.0, option=option)
    pv = LocalVolQuad(kernel(), GridConfig(cells=400, half_width_std=6.0, subdivide=3)).price(c, MARKET)
    assert pv["values"]["pv"] == pytest.approx(bs(option, S0, 1.0) / S0, abs=1e-5)


@pytest.mark.parametrize("kernel", [CTMCKernel, lambda: GaussRichardsonKernel(1)], ids=["ctmc", "gauss"])
def test_ac1_parity_route_matches_direct_call(kernel):
    engine = LocalVolQuad(kernel(), GridConfig(cells=400, half_width_std=6.0, subdivide=3))
    put = engine.price(vanilla_contract(S0, 1.0, option="put"), MARKET)["values"]["pv"]
    call = engine.price(vanilla_contract(S0, 1.0, option="call"), MARKET)["values"]["pv"]
    via_parity = call_from_put(put, vanilla_contract(S0, 1.0), MARKET)
    assert via_parity == pytest.approx(bs("call", S0, 1.0) / S0, abs=1e-5)
    assert via_parity == pytest.approx(call, abs=1e-5)


@pytest.mark.parametrize("kernel", [CTMCKernel, lambda: GaussRichardsonKernel(1)], ids=["ctmc", "gauss"])
def test_ac1_forward_identity_over_the_life(kernel):
    """Daily propagation of S/S0 over one year returns the discounted forward e^{-qT} at s0."""
    c = vanilla_contract(S0, 1.0)
    grid = build_grid(c, SIGMA, GridConfig(cells=400, half_width_std=6.0, subdivide=3))
    k = kernel()
    v = np.exp(grid.centres - grid.centres[grid.spot_index])[None, :]
    times = [d / 252 for d in range(253)]
    for left, right in zip(times[-2::-1], times[::-1]):
        v = k.propagate(v, grid, MARKET, left, right)
    assert v[0, grid.spot_index] == pytest.approx(math.exp(-Q * 1.0), rel=1e-6)


@pytest.mark.parametrize("kernel,subdivide", [(lambda: CTMCKernel(time_pieces=4), 1),
                                              (lambda: GaussRichardsonKernel(1), 3)], ids=["ctmc", "gauss"])
def test_backward_propagation_applies_the_latest_pillar_piece_first(kernel, subdivide):
    """One interval spanning surface pillars must equal the chain of its daily sub-intervals.

    Regression: pillar-cut pieces were applied in increasing time, which is invisible on
    daily intervals and on constant surfaces (commuting generators) but wrong over a long
    interval on a term-structured surface.
    """
    m = load_fixture()
    c = vanilla_contract(m.spot, 0.25, option="put")
    grid = build_grid(c, 0.4, GridConfig(cells=200, half_width_std=6.0, subdivide=subdivide))
    v = np.maximum(c.strike - np.exp(grid.centres), 0.0)[None, :] / m.spot
    whole = kernel().propagate(v, grid, m, 0.0, 0.25)
    chained, k = v, kernel()
    times = np.linspace(0.0, 0.25, 64)
    for left, right in zip(times[-2::-1], times[::-1]):
        chained = k.propagate(chained, grid, m, float(left), float(right))
    i = grid.spot_index
    assert whole[0, i] == pytest.approx(chained[0, i], abs=2e-5)


@pytest.mark.slow
def test_ac3_gauss_validates_on_real_surface_when_resolved():
    m = load_fixture()
    c = snowball_contract(m.spot, 0.25, 3)
    kernel = GaussRichardsonKernel(1)
    out = LocalVolQuad(kernel, GridConfig(cells=400, half_width_std=6.0, subdivide=9), validate=True).price(c, m)
    tol = out["diagnostics"]["tolerance"]
    assert kernel.diagnostics
    assert all(d["mass"] <= tol and d["forward"] <= tol for d in kernel.diagnostics)
    assert out["diagnostics"]["forward"] <= tol and out["diagnostics"]["mass"] <= tol
    assert kernel.validate is False             # validate is scoped to the call


def test_ac3_timed_mode_checks_identities_every_interval():
    """D16: timed mode carries [1, S/S0] through every interval; a broken forward is refused."""

    class LeakyKernel(CTMCKernel):
        def propagate(self, values, grid, market, t0, t1):
            out = super().propagate(values, grid, market, t0, t1)
            out[-1] *= 1.0 + 1e-6                 # a forward-identity defect on the S probe
            return out

    c = snowball_contract(S0, 0.25, 3)
    out = LocalVolQuad(CTMCKernel(), GridConfig(cells=300)).price(c, MARKET)
    assert out["diagnostics"]["intervals"] == len(c.ki_times)
    assert out["diagnostics"]["forward"] <= out["diagnostics"]["tolerance"]
    with pytest.raises(UnresolvedGrid, match="identity residual"):
        LocalVolQuad(LeakyKernel(), GridConfig(cells=300)).price(c, MARKET)


@pytest.mark.slow
def test_ac3_validation_mode_on_real_surface():
    m = load_fixture()
    c = snowball_contract(m.spot, 0.25, 3)
    kernel = CTMCKernel()
    out = LocalVolQuad(kernel, GridConfig(cells=400, half_width_std=6.0), validate=True).price(c, m)
    tol = out["diagnostics"]["tolerance"]
    assert tol == pytest.approx(IDENTITY_FRACTION * PV_BUDGET / len(c.ki_times))
    assert kernel.diagnostics and all(d["mass"] <= tol and d["forward"] <= tol and d["semigroup"] <= tol
                                      for d in kernel.diagnostics)
    assert len(kernel.diagnostics) == kernel.operators


def test_ac3_ctmc_refuses_negative_rates():
    m = Market(S0, 0.0, 0.5, constant_surface(0.01))
    c = vanilla_contract(S0, 1.0)
    with pytest.raises(UnresolvedGrid, match="negative Markov-chain rate"):
        LocalVolQuad(CTMCKernel(), GridConfig(cells=16, half_width_std=400.0)).price(c, m)


def test_ac3_gauss_refuses_coarse_grid_and_narrow_band():
    c = vanilla_contract(S0, 1.0)
    with pytest.raises(UnresolvedGrid, match="quantization"):
        LocalVolQuad(GaussRichardsonKernel(1), GridConfig(cells=16, half_width_std=6.0)).price(
            snowball_contract(S0, 1.0, 12), MARKET)
    with pytest.raises(UnresolvedGrid):      # a 2-sd band is refused (mass or moment matching)
        LocalVolQuad(GaussRichardsonKernel(1, n_sd=2.0), GridConfig(cells=400), validate=True).price(c, MARKET)


@pytest.mark.slow
@pytest.mark.parametrize("contract", [vanilla_contract(S0, 1.0), up_out_contract(S0, 1.0, 12),
                                      snowball_contract(S0, 1.0, 12), snowball_contract(S0, 2.0, 24)],
                         ids=["put", "up_out", "snowball_1y", "snowball_2y"])
def test_ac8_constant_vol_full_risk_matches_production_pde(contract):
    from quantark.asset.equity.param import PDEParams
    ours = LocalVolQuad(CTMCKernel(), GridConfig(cells=400, half_width_std=6.0, subdivide=3)).full_risk(
        contract, MARKET)["values"]
    pde = adapters.pde_full_risk(contract, MARKET, PDEParams(accuracy="high"))["values"]
    for key in OUTPUTS:
        assert abs(ours[key] - pde[key]) <= budget(key, pde[key]), (key, ours[key], pde[key])


def test_full_risk_is_deterministic():
    c = snowball_contract(S0, 0.25, 3)
    engine = LocalVolQuad(CTMCKernel(), GridConfig(cells=200))
    a, b = engine.full_risk(c, MARKET)["values"], engine.full_risk(c, MARKET)["values"]
    assert a == b
