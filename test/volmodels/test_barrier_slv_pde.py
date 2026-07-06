import numpy as np
from datetime import datetime

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.volmodels.slv.slv_pde_kernel import price_barrier_slv_pde, price_european_slv_pde
from quantark.volmodels.slv.slv_mc_kernel import price_barrier_slv_mc
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.product.option import BarrierOption
from quantark.util.enum import OptionType, BarrierType, ObservationType

HEST = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)


def _unit_lev(s0=100.):
    ks = np.array(list(s0 * np.exp(np.linspace(-1.2, 1.2, 15))))
    return LeverageSurface(time_grid=np.linspace(0, 1, 6), strike_grid=ks, leverage_grid=np.ones((6, ks.size)))


def test_bsm_flat_limit_matches_analytical():
    flat = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=1e-9, rho=0.0)
    s0, r, q = 100., 0.03, 0.01
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    surf = GridVolSurface(strikes, list(np.linspace(0.1, 2.0, 6)), np.full((6, 9), 0.20))
    env = PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                             spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=ContinuousDividendYield(q))
    prod = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS)
    ana = BarrierAnalyticalEngine().price(prod, env)
    p = price_barrier_slv_pde(s0, 100., True, 1.0, flat, _unit_lev(), r, q, barrier=130., is_up=True,
                              is_out=True, continuous=True, n_x=240, n_v=10, n_t=120)
    assert abs(p - ana) < 0.3


def test_barrier_far_reproduces_european():
    lev = _unit_lev()
    p_eur = price_european_slv_pde(100., 100., True, 1.0, HEST, lev, 0.03, 0.01, n_x=160, n_v=64, n_t=80)
    p_bar = price_barrier_slv_pde(100., 100., True, 1.0, HEST, lev, 0.03, 0.01, barrier=400., is_up=True,
                                  is_out=True, continuous=True, n_x=160, n_v=64, n_t=80)
    assert abs(p_bar - p_eur) < 0.15


def test_discrete_terminal_observation_ignores_rannacher_half_step():
    lev = _unit_lev()
    p_rannacher = price_barrier_slv_pde(
        100., 100., True, 1.0, HEST, lev, 0.03, 0.01, barrier=125.,
        is_up=True, is_out=True, continuous=False, observe_taus=[0.0],
        n_x=160, n_v=64, n_t=80, rannacher=True,
    )
    p_plain = price_barrier_slv_pde(
        100., 100., True, 1.0, HEST, lev, 0.03, 0.01, barrier=125.,
        is_up=True, is_out=True, continuous=False, observe_taus=[0.0],
        n_x=160, n_v=64, n_t=80, rannacher=False,
    )
    assert abs(p_rannacher - p_plain) < 0.02


def test_discrete_pinned_barrier_uses_surviving_side_limit():
    lev = _unit_lev()
    p = price_barrier_slv_pde(
        100., 100., True, 1.0, HEST, lev, 0.03, 0.01, barrier=125.,
        is_up=True, is_out=True, continuous=False,
        observe_taus=[1.0, 0.75, 0.5, 0.25, 0.0],
        n_x=160, n_v=64, n_t=80,
    )
    # Regression: zeroing the grid node pinned exactly at the discrete barrier
    # over-knocked the coarse PDE and produced about 4.29 for this case.
    assert p > 4.35


def test_mc_pde_agreement():
    lev = _unit_lev()
    dt = np.full(120, 1.0 / 120); rf = np.full(120, 0.03); cf = np.full(120, 0.01); df = np.exp(-0.03)
    p_pde = price_barrier_slv_pde(100., 100., True, 1.0, HEST, lev, 0.03, 0.01, barrier=125., is_up=True,
                                  is_out=True, continuous=True, n_x=200, n_v=80, n_t=120)
    p_mc, se = price_barrier_slv_mc(100., 100., True, HEST, lev, dt, rf, cf, df, barrier=125., is_up=True,
                                    is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                    num_paths=150_000, seed=19, return_stderr=True)
    # Continuous PDE truncates the domain at the barrier; MC uses Brownian-bridge survival over
    # simulated SLV paths. They should agree within MC/statistical and path-discretization error.
    assert abs(p_pde - p_mc) < max(0.5, 4 * se)
