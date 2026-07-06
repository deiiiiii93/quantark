import numpy as np
import pytest
from datetime import datetime

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc, price_barrier_heston_mc
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.product.option import BarrierOption
from quantark.util.enum import OptionType, BarrierType, ObservationType
from quantark.util.enum.engine_enums import HestonMCScheme


def _grid(T=1.0, n=100, r=0.03, q=0.01):
    return np.full(n, T / n), np.full(n, r), np.full(n, q), np.exp(-r * T)


HEST = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)


def test_barrier_far_reproduces_european_same_scheme():
    dt, rf, cf, df = _grid()
    p_eur, se_e = price_european_heston_mc(100., 100., True, HEST, dt, rf, cf, df, scheme=HestonMCScheme.EULERLOG,
                                           num_paths=60_000, seed=1, return_stderr=True)
    p_bar, se_b = price_barrier_heston_mc(100., 100., True, HEST, dt, rf, cf, df, barrier=1e7, is_up=True,
                                          is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                          num_paths=60_000, seed=2, return_stderr=True)
    # independent RNG streams -> agree within a few combined standard errors
    assert abs(p_bar - p_eur) < 4 * np.hypot(se_e, se_b)


def test_bsm_flat_limit_matches_analytical():
    # sigma -> 0, rho 0, v0=theta -> GBM at 20%
    flat = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=1e-6, rho=0.0)
    s0, r, q = 100., 0.03, 0.01
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    surf = GridVolSurface(strikes, list(np.linspace(0.1, 2.0, 6)), np.full((6, 9), 0.20))
    env = PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                             spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=ContinuousDividendYield(q))
    prod = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS)
    ana = BarrierAnalyticalEngine().price(prod, env)
    dt, rf, cf, df = _grid(r=r, q=q)
    p, se = price_barrier_heston_mc(s0, 100., True, flat, dt, rf, cf, df, barrier=130., is_up=True,
                                    is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                    num_paths=120_000, seed=5, return_stderr=True)
    assert abs(p - ana) < max(0.2, 3 * se)


def test_in_out_parity_zero_rebate():
    dt, rf, cf, df = _grid()
    common = dict(barrier=115., pay_at_hit=False, continuous=True, num_paths=60_000, seed=3, rebate=0.)
    ko = price_barrier_heston_mc(100., 100., True, HEST, dt, rf, cf, df, is_up=True, is_out=True, **common)
    ki = price_barrier_heston_mc(100., 100., True, HEST, dt, rf, cf, df, is_up=True, is_out=False, **common)
    van = price_european_heston_mc(100., 100., True, HEST, dt, rf, cf, df, scheme=HestonMCScheme.EULERLOG,
                                   num_paths=60_000, seed=3)
    assert abs((ko + ki) - van) < 0.15
