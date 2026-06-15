import numpy as np
import pytest
from datetime import datetime
from quantark.param import GridVolSurface, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment, FxPricingEnvironment
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.mc import HestonSLVMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.engine.mc import FxHestonSLVMCEngine
from quantark.util.enum import OptionType
from quantark.util.exceptions import PricingError
from quantark.volmodels.heston import HestonParams


P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.5)


def _grid_env(vol=0.2, r=0.0, q=0.0):
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 9), vol))
    return PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=100.0), vol_surface=surf,
                              div_yield=ContinuousDividendYield(q))


def test_equity_slv_reprices_flat_vanilla():
    env = _grid_env(0.2, 0.0, 0.0)
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    bs = BlackScholesEngine().price(opt, env)
    slv = HestonSLVMCEngine(P, eta=1.0, params=MCParams(num_paths=120_000, time_steps=100,
                                                        seed=7)).price(opt, env)
    assert slv == pytest.approx(bs, abs=0.35)  # MC error (rigorous check is in the kernel test)


def test_equity_slv_requires_grid_vol_surface():
    env = _grid_env()
    env.vol_surface = FlatVolSurface(0.2)
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    with pytest.raises(PricingError):
        HestonSLVMCEngine(P).price(opt, env)


def test_equity_slv_greeks_no_vega():
    env = _grid_env()
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    g = HestonSLVMCEngine(P, params=MCParams(num_paths=30_000, time_steps=40, seed=1)).calculate_greeks(opt, env)
    assert set(["price", "delta", "gamma", "theta", "rho"]).issubset(g.keys())
    assert "vega" not in g


def test_equity_slv_engine_accepts_precomputed_leverage_surface():
    env = _grid_env()
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    lv = HestonSLVMCEngine(P)._build_surface(env)
    M = 50
    dt = np.full(M, 1.0 / M)
    from quantark.volmodels.slv import calibrate_leverage_surface

    leverage = calibrate_leverage_surface(
        100.0, P, lv, dt, np.zeros(M), np.zeros(M),
        num_paths=20_000, num_bins=20, seed=5,
    )
    engine = HestonSLVMCEngine(
        P,
        params=MCParams(num_paths=20_000, time_steps=M, seed=7),
        local_vol_surface=lv,
        leverage_surface=leverage,
    )
    assert engine.price(opt, env) > 0.0


def _fx_grid_env(vol=0.1, rd=0.0, rf=0.0):
    strikes = list(1.20 * np.exp(np.linspace(-0.5, 0.5, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 9), vol))
    return FxPricingEnvironment(valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=1.20),
                                domestic_curve=FlatRateCurve(rd), foreign_curve=FlatRateCurve(rf),
                                vol_surface=surf)


def test_fx_slv_reprices_flat_vanilla():
    env = _fx_grid_env(0.1, 0.0, 0.0)
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          notional_foreign=1_000_000.0)
    gk = GarmanKohlhagenEngine().price(opt, env)
    slv = FxHestonSLVMCEngine(P, eta=1.0, num_paths=80_000, time_steps=100, seed=9).price(opt, env)
    assert slv == pytest.approx(gk, rel=3e-2)
