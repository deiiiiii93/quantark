import numpy as np
import pytest
from datetime import datetime
from quantark.param import GridVolSurface, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.engine.mc.local_vol_mc_engine import FxLocalVolMCEngine
from quantark.asset.fx.engine.pde.local_vol_pde_solver import FxLocalVolPDESolver
from quantark.util.enum import OptionType
from quantark.util.exceptions import PricingError


def _fx_env(vol=0.1, rd=0.03, rf=0.01):
    strikes = list(1.20 * np.exp(np.linspace(-0.4, 0.4, 7)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 7), vol))
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rd), foreign_curve=FlatRateCurve(rf), vol_surface=surf,
    )


def _opt(strike=1.20, T=1.0, notional=1_000_000.0, participation=1.0):
    return FxVanillaOption(strike=strike, option_type=OptionType.CALL, maturity=T,
                           notional_foreign=notional, participation_rate=participation)


def test_fx_lv_pde_matches_gk_flat_surface():
    env = _fx_env(0.1, 0.03, 0.01)
    opt = _opt()
    gk = GarmanKohlhagenEngine().price(opt, env)
    pde = FxLocalVolPDESolver(grid_size=500, time_steps=200).price(opt, env)
    assert pde == pytest.approx(gk, rel=2e-3)


def test_fx_lv_mc_matches_gk_flat_surface():
    env = _fx_env(0.1, 0.03, 0.01)
    opt = _opt()
    gk = GarmanKohlhagenEngine().price(opt, env)
    mc = FxLocalVolMCEngine(num_paths=300_000, time_steps=200, seed=5).price(opt, env)
    # per-unit MC error ~ notional * 1e-4; scale tolerance to notional
    assert mc == pytest.approx(gk, rel=5e-3)


def test_fx_lv_sizing_scales_with_notional_and_participation():
    env = _fx_env()
    base = FxLocalVolPDESolver(grid_size=400, time_steps=150).price(_opt(notional=1e6), env)
    dbl = FxLocalVolPDESolver(grid_size=400, time_steps=150).price(_opt(notional=2e6), env)
    part = FxLocalVolPDESolver(grid_size=400, time_steps=150).price(
        _opt(notional=1e6, participation=2.0), env)
    assert dbl == pytest.approx(2 * base, rel=1e-9)
    assert part == pytest.approx(2 * base, rel=1e-9)


def test_fx_lv_rejects_market_forward_override():
    env = _fx_env()
    env.market_forward = 1.25
    with pytest.raises(PricingError):
        FxLocalVolPDESolver().price(_opt(), env)


def test_fx_lv_requires_grid_vol_surface():
    env = _fx_env()
    env.vol_surface = FlatVolSurface(0.1)
    with pytest.raises(PricingError):
        FxLocalVolPDESolver().price(_opt(), env)


def test_fx_lv_greeks_exclude_vega():
    env = _fx_env()
    g = FxLocalVolPDESolver(grid_size=300, time_steps=120).calculate_greeks(_opt(), env)
    assert set(["price", "delta", "gamma", "rho_dom", "rho_for"]).issubset(g.keys())
    assert "vega" not in g


def test_fx_lv_greeks_match_gk_magnitude():
    env = _fx_env(0.1, 0.03, 0.01)
    opt = _opt()
    gk = GarmanKohlhagenEngine().calculate_greeks(opt, env)
    g = FxLocalVolPDESolver(grid_size=600, time_steps=250).calculate_greeks(opt, env)
    assert "theta" in g
    assert g["rho_dom"] == pytest.approx(gk["rho_dom"], rel=2e-2)
    assert g["rho_for"] == pytest.approx(gk["rho_for"], rel=2e-2)
    assert g["delta"] == pytest.approx(gk["delta"], rel=3e-2)  # FD-on-PDE delta carries grid noise


def test_fx_lv_rejects_expiry_delivery_mismatch():
    env = _fx_env()
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          delivery=1.05, notional_foreign=1e6)
    with pytest.raises(PricingError):
        FxLocalVolPDESolver().price(opt, env)


def test_fx_lv_theta_with_explicit_equal_delivery():
    # Explicit delivery == maturity must survive the theta maturity-shrink (both reduced).
    env = _fx_env()
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          delivery=1.0, notional_foreign=1e6)
    g = FxLocalVolPDESolver(grid_size=300, time_steps=120).calculate_greeks(opt, env)
    assert "theta" in g and g["theta"] < 0
