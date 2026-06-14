import numpy as np
import pytest
from datetime import datetime
from quantark.param import GridVolSurface, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.mc.local_vol_mc_engine import LocalVolMCEngine
from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.util.enum import OptionType
from quantark.util.exceptions import PricingError


def _env(vol=0.2, r=0.03, q=0.01):
    strikes = list(100.0 * np.exp(np.linspace(-0.5, 0.5, 7)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 7), vol))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=100.0), vol_surface=surf,
        div_yield=ContinuousDividendYield(q),
    )


def _call(strike=100.0, T=1.0):
    return EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL, maturity=T)


def test_equity_lv_pde_matches_bs_flat_surface():
    env = _env(0.2, 0.03, 0.01)
    bs = BlackScholesEngine().price(_call(), env)
    pde = LocalVolPDESolver(params=PDEParams(grid_size=400, time_steps=200)).price(_call(), env)
    assert pde == pytest.approx(bs, abs=3e-2)


def test_equity_lv_mc_matches_bs_flat_surface():
    env = _env(0.2, 0.03, 0.01)
    bs = BlackScholesEngine().price(_call(), env)
    mc = LocalVolMCEngine(params=MCParams(num_paths=200_000, time_steps=252, seed=11,
                                          use_antithetic=True)).price(_call(), env)
    assert mc == pytest.approx(bs, abs=0.15)


def test_equity_lv_requires_grid_vol_surface():
    env = _env()
    env.vol_surface = FlatVolSurface(0.2)
    with pytest.raises(PricingError):
        LocalVolPDESolver().price(_call(), env)


def test_equity_lv_greeks_delta_gamma_theta_rho_no_vega():
    env = _env()
    g = LocalVolPDESolver(params=PDEParams(grid_size=300, time_steps=150)).calculate_greeks(_call(), env)
    assert set(["price", "delta", "gamma", "theta", "rho"]).issubset(g.keys())
    assert "vega" not in g
    assert g["delta"] > 0  # call delta positive


def test_equity_lv_pde_does_not_invoke_mc(monkeypatch):
    # Prove the PDE path never calls the MC kernel.
    import quantark.volmodels.localvol.mc_kernel as mck

    def _boom(*a, **k):
        raise AssertionError("PDE must not call the MC kernel")

    monkeypatch.setattr(mck, "price_european_lv_mc", _boom)
    env = _env()
    LocalVolPDESolver(params=PDEParams(grid_size=200, time_steps=100)).price(_call(), env)


def test_equity_lv_prebuilt_surface_used(monkeypatch):
    env = _env()
    import quantark.asset.equity.engine.pde.local_vol_pde_solver as mod

    def _boom(*a, **k):
        raise AssertionError("should not rebuild Dupire when prebuilt surface supplied")

    # build once via the real builder, then ensure construction-supplied surface skips rebuild
    lv = LocalVolPDESolver()._build_surface(env)
    monkeypatch.setattr(mod, "build_dupire_local_vol", _boom)
    price = LocalVolPDESolver(params=PDEParams(grid_size=200, time_steps=100),
                              local_vol_surface=lv).price(_call(), env)
    assert price > 0
