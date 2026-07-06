import numpy as np
from datetime import datetime

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.volmodels.slv.slv_mc_kernel import price_european_slv_mc, price_barrier_slv_mc
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.product.option import BarrierOption
from quantark.util.enum import OptionType, BarrierType, ObservationType


def _flat_lv(vol=0.20, s0=100., r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    surf = GridVolSurface(strikes, list(np.linspace(0.1, 2.0, 6)), np.full((6, 9), vol))
    env = PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                             spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=ContinuousDividendYield(q))
    lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve, div_yield=env.get_div_yield)
    return lv, env, surf


def _unit_leverage(s0=100.):
    ks = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    tg = np.linspace(0.0, 1.0, 6)
    return LeverageSurface(time_grid=tg, strike_grid=ks, leverage_grid=np.ones((6, ks.size)))


def _grid(T=1.0, n=100, r=0.03, q=0.01):
    return np.full(n, T / n), np.full(n, r), np.full(n, q), np.exp(-r * T)


HEST = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)


def test_barrier_far_reproduces_european():
    lv, _, _ = _flat_lv()
    lev = _unit_leverage()
    dt, rf, cf, df = _grid()
    p_eur, se = price_european_slv_mc(100., 100., True, HEST, lv, dt, rf, cf, df, eta=1.0,
                                      num_paths=60_000, seed=1, leverage_surface=lev, return_stderr=True)
    p_bar, se_b = price_barrier_slv_mc(100., 100., True, HEST, lev, dt, rf, cf, df, barrier=1e7, is_up=True,
                                       is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                       num_paths=60_000, seed=2, return_stderr=True)
    assert abs(p_bar - p_eur) < 4 * np.hypot(se, se_b)


def test_bsm_flat_limit_matches_analytical():
    flat = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=1e-6, rho=0.0)
    _, env, _ = _flat_lv()
    lev = _unit_leverage()
    prod = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS)
    ana = BarrierAnalyticalEngine().price(prod, env)
    dt, rf, cf, df = _grid()
    p, se = price_barrier_slv_mc(100., 100., True, flat, lev, dt, rf, cf, df, barrier=130., is_up=True,
                                 is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                 num_paths=120_000, seed=5, return_stderr=True)
    assert abs(p - ana) < max(0.2, 3 * se)


def test_in_out_parity_zero_rebate():
    lv, _, _ = _flat_lv()
    lev = _unit_leverage()
    dt, rf, cf, df = _grid()
    common = dict(barrier=115., pay_at_hit=False, continuous=True, num_paths=60_000, seed=3, rebate=0.)
    ko = price_barrier_slv_mc(100., 100., True, HEST, lev, dt, rf, cf, df, is_up=True, is_out=True, **common)
    ki = price_barrier_slv_mc(100., 100., True, HEST, lev, dt, rf, cf, df, is_up=True, is_out=False, **common)
    van = price_european_slv_mc(100., 100., True, HEST, lv, dt, rf, cf, df, eta=1.0,
                                num_paths=60_000, seed=3, leverage_surface=lev)
    assert abs((ko + ki) - van) < 0.15
