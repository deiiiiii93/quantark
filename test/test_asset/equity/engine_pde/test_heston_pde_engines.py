import pytest
from datetime import datetime
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment, FxPricingEnvironment
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import HestonAnalyticalEngine
from quantark.asset.equity.engine.pde import HestonPDESolver
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.asset.fx.engine.analytical import FxHestonAnalyticalEngine
from quantark.asset.fx.engine.pde import FxHestonPDESolver
from quantark.util.enum import OptionType
from quantark.volmodels.heston import HestonParams


P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)


def _eq_env(r=0.03, q=0.01):
    return PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(0.2),
                              div_yield=ContinuousDividendYield(q))


def test_equity_heston_pde_matches_analytic():
    env = _eq_env()
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    analytic = HestonAnalyticalEngine(P).price(opt, env)
    pde = HestonPDESolver(P, n_x=220, n_v=90, n_t=120).price(opt, env)
    assert pde == pytest.approx(analytic, abs=0.3)


def test_equity_heston_pde_greeks_no_vega():
    env = _eq_env()
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    g = HestonPDESolver(P, n_x=160, n_v=70, n_t=60).calculate_greeks(opt, env)
    assert set(["price", "delta", "gamma", "theta", "rho"]).issubset(g.keys())
    assert "vega" not in g
    assert 0.0 < g["delta"] < 1.0 + 1e-6


def _fx_env(rd=0.03, rf=0.01):
    return FxPricingEnvironment(valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=1.20),
                                domestic_curve=FlatRateCurve(rd), foreign_curve=FlatRateCurve(rf),
                                vol_surface=FlatVolSurface(0.1))


def test_fx_heston_pde_matches_analytic():
    env = _fx_env()
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          notional_foreign=1_000_000.0)
    analytic = FxHestonAnalyticalEngine(P).price(opt, env)
    pde = FxHestonPDESolver(P, n_x=220, n_v=90, n_t=120).price(opt, env)
    assert pde == pytest.approx(analytic, rel=3e-3)


def test_equity_heston_pde_gamma_positive_and_reasonable():
    env = _eq_env()
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    g = HestonPDESolver(P, n_x=320, n_v=120, n_t=140).calculate_greeks(opt, env)
    # ATM call gamma must be clearly positive (the bilinear-bump bug produced ~0/garbage).
    assert g["gamma"] > 0.005
    assert g["theta"] < 0


def test_2d_event_damping_keys_follow_resolved_stepper_counts():
    from quantark.asset.equity.engine.pde.core.components import resolve_stepper
    from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import event_damped_step_keys
    from quantark.asset.equity.param import PDEParams

    def st(**kw):
        return resolve_stepper(PDEParams(stepper=CrankNicolson(**kw)), DAMPING_DEFAULTS["2D"])  # the 2D row

    maps = {"ko": {10: None, 20: None}, "ki": {0: None}}
    assert event_damped_step_keys(st(), maps, 100) == {11, 12, 21, 22}  # default 2
    assert event_damped_step_keys(st(event_damping_steps=0), maps, 100) is None
    assert event_damped_step_keys(st(event_damping_steps=1), maps, 100) == {11, 21}
    # the two counts are independent (the old use_rannacher master switch is gone)
    assert event_damped_step_keys(st(terminal_damping_steps=0), maps, 100) == {11, 12, 21, 22}
