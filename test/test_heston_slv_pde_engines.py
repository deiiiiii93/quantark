import numpy as np
import pytest
from datetime import datetime
from quantark.param import GridVolSurface, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment, FxPricingEnvironment
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.pde import HestonSLVPDESolver
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.engine.pde import FxHestonSLVPDESolver
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface
from quantark.util.enum.engine_enums import LeverageCalibrationMethod

P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.2, rho=-0.5)


def _lev(spot, rate_curve, div, rd_carry=0.0):
    strikes = list(spot * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    lv = build_dupire_local_vol(GridVolSurface(strikes, mats, np.full((6, 9), 0.2)),
                                spot=spot, rate_curve=rate_curve, div_yield=div)
    M = 60
    dt = np.full(M, 1.0 / M)
    rf = np.full(M, rd_carry)
    cf = np.zeros(M)
    return calibrate_leverage_surface(spot, P, lv, dt, rf, cf, eta=1.0,
                                      method=LeverageCalibrationMethod.MC_BINNING,
                                      num_paths=40_000, num_bins=20, seed=5)


def test_equity_slv_pde_reprices_flat_vanilla():
    env = PricingEnvironment(rate_curve=FlatRateCurve(0.0), valuation_date=datetime(2026, 1, 1),
                             spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(0.2),
                             div_yield=ContinuousDividendYield(0.0))
    lev = _lev(100.0, FlatRateCurve(0.0), lambda t: 0.0)
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    bs = BlackScholesEngine().price(opt, env)
    slv = HestonSLVPDESolver(P, lev, eta=1.0, n_x=200, n_v=80, n_t=80).price(opt, env)
    assert slv == pytest.approx(bs, abs=0.4)


def test_equity_slv_pde_greeks_no_vega():
    env = PricingEnvironment(rate_curve=FlatRateCurve(0.0), valuation_date=datetime(2026, 1, 1),
                             spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(0.2),
                             div_yield=ContinuousDividendYield(0.0))
    lev = _lev(100.0, FlatRateCurve(0.0), lambda t: 0.0)
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    g = HestonSLVPDESolver(P, lev, eta=1.0, n_x=160, n_v=70, n_t=60).calculate_greeks(opt, env)
    assert set(["price", "delta", "gamma", "theta", "rho"]).issubset(g.keys())
    assert "vega" not in g
    assert g["gamma"] > 0.0


def test_slv_pde_requires_leverage_surface():
    with pytest.raises(ValidationError):
        HestonSLVPDESolver(P, leverage_surface=None)  # type: ignore


def test_fx_slv_pde_reprices_flat_vanilla():
    env = FxPricingEnvironment(valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=1.20),
                               domestic_curve=FlatRateCurve(0.0), foreign_curve=FlatRateCurve(0.0),
                               vol_surface=FlatVolSurface(0.2))
    lev = _lev(1.20, FlatRateCurve(0.0), lambda t: 0.0)
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1e6)
    gk = GarmanKohlhagenEngine().price(opt, env)
    slv = FxHestonSLVPDESolver(P, lev, eta=1.0, n_x=200, n_v=80, n_t=80).price(opt, env)
    assert slv == pytest.approx(gk, rel=0.04)
