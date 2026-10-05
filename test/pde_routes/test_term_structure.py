"""Term structure against independent references (spec 3C decision 27): the local-vol vanilla on a constant surface
against Black-Scholes with integrated carry; the DCN against the term recursion at standard / high."""
import math

import pytest

from pde_routes.cases import dcn_case, kinked, surface_const, vanilla, D_RATES, D_YIELDS
from pde_routes.dcn_reference import dcn_exact_term
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
from quantark.asset.equity.param import PDEParams


def bs_integrated_carry(call, spot, strike, T, vol, env):
    df = float(env.get_discount_factor(T))
    fwd = spot * math.exp((env.get_rate(T) - env.get_div_yield(T)) * T)
    sd = vol * math.sqrt(T)
    d1 = (math.log(fwd / strike) + 0.5 * sd * sd) / sd
    n = lambda z: 0.5 * math.erfc(-z / math.sqrt(2.0))          # noqa: E731
    return df * (fwd * n(d1) - strike * n(d1 - sd)) if call else df * (strike * n(sd - d1) - fwd * n(-d1))


@pytest.mark.parametrize("call,strike,T", [(True, 100.0, 1.0), (False, 100.0, 1.0), (True, 120.0, 2.0),
                                           (False, 80.0, 0.5), (True, 100.0, 1.5)])
def test_local_vol_vanilla_on_kinked_curves(call, strike, T):
    env = kinked(100.0)
    got = LocalVolPDESolver(local_vol_surface=surface_const()).price(vanilla(call, strike, T), env)
    assert abs(got - bs_integrated_carry(call, 100.0, strike, T, 0.22, env)) <= 1e-3   # measured 2.6e-4


@pytest.mark.parametrize("accuracy,bound_bp", [("standard", 1.0), ("high", 0.3)])
def test_dcn_on_kinked_curves(accuracy, bound_bp):
    for name in ("A", "B", "SA", "SB"):
        product, env0 = dcn_case(name)
        env = kinked(env0.spot, vol=float(env0.get_vol(env0.spot, 1.0)), rates=D_RATES, yields=D_YIELDS,
                     valuation_date=env0.valuation_date)
        exact = dcn_exact_term(product, env)["pv"][0]
        got = DCNPDEEngine(PDEParams(accuracy=accuracy)).price(product, env)
        assert 1e4 * abs(got - exact) / product.notional <= bound_bp, name          # measured 0.22 / 0.11 bp
