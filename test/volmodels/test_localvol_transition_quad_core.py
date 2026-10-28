"""Grid, events and recursion of the local-vol QUAD, validated with the exact constant-vol kernel."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from example.case_studies.localvol_transition_quad import adapters
from example.case_studies.localvol_transition_quad.contract import (Contract, snowball_contract,
                                                                    up_out_contract, vanilla_contract)
from example.case_studies.localvol_transition_quad.engine import LocalVolQuad
from example.case_studies.localvol_transition_quad.grid import GridConfig, UnresolvedGrid, build_grid
from example.case_studies.localvol_transition_quad.kernels import BSMExactKernel, gaussian_rows
from example.case_studies.localvol_transition_quad.market import Market, constant_surface

S0, R, Q, SIGMA = 100.0, 0.03, 0.01, 0.25
MARKET = Market(S0, R, Q, constant_surface(SIGMA))


def bs(option, k, t):
    f = S0 * math.exp((R - Q) * t)
    d1 = (math.log(f / k) + 0.5 * SIGMA ** 2 * t) / (SIGMA * math.sqrt(t))
    d2 = d1 - SIGMA * math.sqrt(t)
    df = math.exp(-R * t)
    if option == "call":
        return df * (f * norm.cdf(d1) - k * norm.cdf(d2))
    return df * (k * norm.cdf(-d2) - f * norm.cdf(-d1))


def quad(cells=800, width=6.0):
    return LocalVolQuad(BSMExactKernel(), GridConfig(cells=cells, half_width_std=width))


@pytest.mark.parametrize("option", ["put", "call"])
@pytest.mark.parametrize("strike_m", [0.9, 1.0, 1.1])
def test_ac1_vanilla_matches_black_scholes(option, strike_m):
    c = vanilla_contract(S0, 1.0, strike_m=strike_m, option=option)
    pv = quad().price(c, MARKET)["values"]["pv"]
    assert pv == pytest.approx(bs(option, strike_m * S0, 1.0) / S0, abs=1e-5)


def test_ac1_forward_identity_and_mass():
    c = vanilla_contract(S0, 1.0)
    grid = build_grid(c, SIGMA, GridConfig(cells=800, half_width_std=6.0))
    p = BSMExactKernel().matrix(grid, MARKET, 0.0, 1 / 252)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    interior = slice(100, grid.size - 100)
    s = np.exp(grid.centres)
    fwd = (p @ s)[interior] / (s[interior] * math.exp((R - Q) / 252))
    np.testing.assert_allclose(fwd, 1.0, rtol=0, atol=1e-8)


def test_ac1_delta_gamma_match_black_scholes():
    c = vanilla_contract(S0, 1.0, option="put")
    v = quad(1600).price(c, MARKET)["values"]
    t, k = 1.0, S0
    d1 = (math.log(S0 / k) + (R - Q + 0.5 * SIGMA ** 2) * t) / (SIGMA * math.sqrt(t))
    delta = -math.exp(-Q * t) * norm.cdf(-d1)
    gamma = math.exp(-Q * t) * norm.pdf(d1) / (S0 * SIGMA * math.sqrt(t))
    assert v["delta"] == pytest.approx(delta, abs=1e-4)
    assert v["gamma"] == pytest.approx(gamma * S0, abs=1e-3)


@pytest.mark.parametrize("seed", range(20))
def test_ac6_levels_on_edges_and_spot_on_centre(seed):
    rng = np.random.default_rng(seed)
    s0 = float(rng.uniform(50, 5000))
    ki_m, ko_m = rng.uniform(0.6, 0.97), rng.uniform(1.005, 1.3)
    c = Contract("snowball", float(rng.uniform(0.25, 2.0)), s0 * rng.uniform(ki_m + 0.01, 1.0), s0,
                 ko=ko_m * s0, ki=ki_m * s0, ko_times=(0.25,), ki_times=(0.25,))
    c = Contract("snowball", c.maturity, c.strike, s0, ko=c.ko, ki=c.ki, ko_times=(c.maturity,),
                 ki_times=(c.maturity,))
    grid = build_grid(c, float(rng.uniform(0.1, 0.5)), GridConfig(cells=int(rng.integers(64, 600))))
    for level in (c.ko, c.ki):
        assert np.min(np.abs(grid.edges - math.log(level))) < 1e-12
    assert abs(grid.centres[grid.spot_index] - math.log(s0)) < 1e-12
    k = 3
    w = grid.widths[grid.spot_index - k:grid.spot_index + k + 1]
    np.testing.assert_allclose(w, w[0], rtol=1e-10)
    widths = grid.widths
    assert np.min(widths) >= np.median(widths) / 3, "sliver cell (it would set the operator stiffness)"
    block = (grid.edges[grid.spot_index - k], grid.edges[grid.spot_index + k + 1])
    if not block[0] < math.log(c.strike) < block[1]:
        assert np.min(np.abs(grid.edges - math.log(c.strike))) < 1e-12
    assert np.all(np.diff(grid.edges) > 0)


def test_quantization_refusal():
    c = vanilla_contract(S0, 1.0)
    grid = build_grid(c, SIGMA, GridConfig(cells=16, half_width_std=6.0))
    with pytest.raises(UnresolvedGrid):
        gaussian_rows(grid, grid.centres, np.full(grid.size, 1e-6))


def test_ac2_up_out_matches_production_discrete_quad():
    from quantark.asset.equity.engine.quad.discrete_quad_engine import DiscreteQuadEngine
    c = up_out_contract(S0, 1.0, 12)
    ours = quad(1600).price(c, MARKET)["values"]["pv"]
    theirs = DiscreteQuadEngine().price(adapters.production_up_out(c), adapters.environment(MARKET)) / S0
    assert ours == pytest.approx(theirs, abs=1e-4)


def _order(values, ratio):
    a, b, c = values
    return math.log(abs(a - b) / abs(b - c)) / math.log(ratio)


@pytest.mark.slow
@pytest.mark.parametrize("contract", [snowball_contract(S0, 1.0, 12), up_out_contract(S0, 1.0, 12)],
                         ids=["snowball_daily_ki", "up_out_monthly"])
def test_ac13_second_order_under_refinement(contract):
    """Nested odd refinement (every cell split in 3) keeps barrier edges and the s0 centre fixed."""
    pvs = [LocalVolQuad(BSMExactKernel(), GridConfig(cells=400, half_width_std=6.0, subdivide=m))
           .price(contract, MARKET)["values"]["pv"] for m in (1, 3, 9)]
    order = _order(pvs, 3)
    assert 1.6 <= order <= 2.6, (pvs, order)


def test_subdivision_keeps_levels_and_spot():
    c = snowball_contract(S0, 1.0, 12)
    base = build_grid(c, SIGMA, GridConfig(cells=300))
    fine = build_grid(c, SIGMA, GridConfig(cells=300, subdivide=3))
    assert fine.size == 3 * base.size
    assert set(np.round(base.edges, 12)) <= set(np.round(fine.edges, 12))
    assert fine.centres[fine.spot_index] == pytest.approx(math.log(S0), abs=1e-12)
