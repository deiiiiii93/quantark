"""Demo: Local Volatility, Heston, and Heston-SLV pricing for equity and FX.

Run: python example/vol_models_demo.py

Shows the new quantark.volmodels engines:
  - Local Volatility (Dupire) via Monte Carlo and PDE
  - Heston via semi-analytical (Gatheral), Monte Carlo (QE), and PDE (Craig-Sneyd)
  - Heston Stochastic-Local-Volatility via MC (on-the-fly leverage) and backward PDE
    (consuming an MC-calibrated leverage surface)
for both the equity (PricingEnvironment) and FX (FxPricingEnvironment) asset classes.
"""

from datetime import datetime

import numpy as np

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment, FxPricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType, HestonAnalyticalMethod, HestonMCScheme

from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine, HestonAnalyticalEngine
from quantark.asset.equity.engine.mc import LocalVolMCEngine, HestonMCEngine, HestonSLVMCEngine
from quantark.asset.equity.engine.pde import (
    LocalVolPDESolver, HestonPDESolver, HestonSLVPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.riskmeasures import VolModelRiskCalculator

from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine, FxHestonAnalyticalEngine
from quantark.asset.fx.riskmeasures import FxVolModelRiskCalculator

from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.risk import ModelRiskRequest, SlvLeverageRiskMode, SurfaceBump
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig
from quantark.util.enum.engine_enums import LeverageCalibrationMethod

HESTON = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)


def equity_demo():
    print("\n=== EQUITY (S0=100, K=100, T=1, r=3%, q=1%) ===")
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 9), 0.20))  # flat 20% market IV
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=100.0), vol_surface=surf,
        div_yield=ContinuousDividendYield(0.01),
    )
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)

    print(f"  Black-Scholes (flat 20%) : {BlackScholesEngine().price(opt, env):.4f}")
    print(f"  Local Vol  PDE           : {LocalVolPDESolver(PDEParams(grid_size=300, time_steps=150)).price(opt, env):.4f}")
    print(f"  Local Vol  MC            : {LocalVolMCEngine(MCParams(num_paths=100_000, time_steps=100, seed=1, use_antithetic=True)).price(opt, env):.4f}")
    he = HestonAnalyticalEngine(HESTON, method=EngineType.ANALYTICAL(HestonAnalyticalMethod.GATHERAL))
    print(f"  Heston analytical        : {he.price(opt, env):.4f}")
    print(f"  Heston MC (QE)           : {HestonMCEngine(HESTON, scheme=HestonMCScheme.QUADEXP, params=MCParams(num_paths=100_000, time_steps=50, seed=1, use_antithetic=True)).price(opt, env):.4f}")
    print(f"  Heston PDE (Craig-Sneyd) : {HestonPDESolver(HESTON, n_x=200, n_v=80, n_t=80).price(opt, env):.4f}")
    print(f"  SLV  MC                  : {HestonSLVMCEngine(HESTON, eta=1.0, params=MCParams(num_paths=100_000, time_steps=100, seed=1)).price(opt, env):.4f}")

    # SLV PDE consumes a precomputed leverage surface. Two calibrators:
    #   - MC binning (van der Stoep): leverage from a simulated (S, v) cloud.
    #   - Forward Fokker-Planck (default): deterministic density evolution, no MC noise.
    lv = build_dupire_local_vol(surf, spot=100.0, rate_curve=env.rate_curve, div_yield=env.get_div_yield)
    n = 60
    t_grid = np.linspace(0.0, 1.0, n + 1)
    r_fwd = np.array([env.rate_curve.get_forward_rate(t_grid[i], t_grid[i + 1]) for i in range(n)])
    from quantark.volmodels.curves import forward_carry_on_grid
    carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
    lev_mc = calibrate_leverage_surface(100.0, HESTON, lv, np.diff(t_grid), r_fwd, carry_fwd,
                                        eta=1.0, method=LeverageCalibrationMethod.MC_BINNING,
                                        num_paths=40_000, num_bins=20, seed=1)
    print(f"  SLV  PDE (MC leverage)   : {HestonSLVPDESolver(HESTON, lev_mc, eta=1.0, n_x=180, n_v=70, n_t=70).price(opt, env):.4f}")
    lev_fp = calibrate_leverage_surface(100.0, HESTON, lv, np.diff(t_grid), r_fwd, carry_fwd,
                                        eta=1.0, fp_config=FpCalibrationConfig(n_x=161, n_z=81))
    print(f"  SLV  PDE (FFP leverage)  : {HestonSLVPDESolver(HESTON, lev_fp, eta=1.0, n_x=180, n_v=70, n_t=70).price(opt, env):.4f}")
    slv_pde = HestonSLVPDESolver(HESTON, lev_fp, eta=1.0, n_x=180, n_v=70, n_t=70)

    # Scalar vega remains absent from calculate_greeks(). Structured risk is explicit.
    risk = VolModelRiskCalculator()
    heston_risk = risk.calculate_model_risk(
        opt, env, he, ModelRiskRequest(parameter_names=("v0", "rho"), surface_bumps=()),
    )
    print("  Heston model risk         : " + ", ".join(
        f"{point.name}={point.derivative:.4f}" for point in heston_risk.successful_points
    ))
    local_market = risk.calculate_market_vega(
        opt, env, LocalVolPDESolver(PDEParams(grid_size=180, time_steps=90)),
    )
    print(f"  Dupire market-IV vega     : {local_market.points[0].derivative:.4f}")
    leverage_risk = risk.calculate_model_risk(
        opt,
        env,
        slv_pde,
        ModelRiskRequest(
            parameter_names=(),
            surface_bumps=(SurfaceBump.parallel(),),
            slv_leverage_mode=SlvLeverageRiskMode.FROZEN,
        ),
    )
    print(f"  SLV leverage parallel risk: {leverage_risk.points[0].derivative:.4f}")


def fx_demo():
    print("\n=== FX EUR/USD (spot=1.20, K=1.20, T=1, r_d=3%, r_f=1%) ===")
    strikes = list(1.20 * np.exp(np.linspace(-0.5, 0.5, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 9), 0.10))
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(0.03), foreign_curve=FlatRateCurve(0.01), vol_surface=surf,
    )
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          notional_foreign=1_000_000.0)
    print(f"  Garman-Kohlhagen (10%)   : {GarmanKohlhagenEngine().price(opt, env):,.2f}")
    heston = FxHestonAnalyticalEngine(HESTON)
    print(f"  FX Heston analytical     : {heston.price(opt, env):,.2f}")
    risk = FxVolModelRiskCalculator().calculate_model_risk(
        opt, env, heston, ModelRiskRequest(parameter_names=("v0", "rho"), surface_bumps=()),
    )
    print("  FX Heston model risk     : " + ", ".join(
        f"{point.name}={point.derivative:,.2f}" for point in risk.successful_points
    ))


if __name__ == "__main__":
    equity_demo()
    fx_demo()
    print("\nDone.")
