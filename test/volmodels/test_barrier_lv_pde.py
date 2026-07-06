import numpy as np
from datetime import datetime

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.localvol.pde_kernel import price_barrier_lv_pde, price_european_lv_pde
from quantark.volmodels.localvol.mc_kernel import price_barrier_lv_mc
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.product.option import BarrierOption
from quantark.util.enum import OptionType, BarrierType, ObservationType


def _lv(vol=0.20, s0=100., r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.7, 0.7, 11)))
    surf = GridVolSurface(strikes, list(np.linspace(0.05, 2.0, 7)), np.full((7, 11), vol))
    env = PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                             spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=ContinuousDividendYield(q))
    return build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve, div_yield=env.get_div_yield), env


def _grid(T=1.0, n=150, r=0.03, q=0.01):
    return T, np.full(n, T / n), np.full(n, r), np.full(n, q)


def test_up_out_call_matches_analytical():
    lv, env = _lv()
    T, dt, rf, cf = _grid()
    prod = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS)
    ana = BarrierAnalyticalEngine().price(prod, env)
    p = price_barrier_lv_pde(100., 100., True, T, lv, dt, rf, cf, barrier=130., is_up=True, is_out=True,
                             continuous=True, n_s=500)
    assert abs(p - ana) < 0.25


def test_barrier_far_reproduces_european():
    lv, _ = _lv()
    T, dt, rf, cf = _grid()
    p_eur = price_european_lv_pde(100., 100., True, T, lv, dt, rf, cf, n_s=400)
    p_bar = price_barrier_lv_pde(100., 100., True, T, lv, dt, rf, cf, barrier=400., is_up=True,
                                 is_out=True, continuous=True, n_s=400, s_max=600.)
    assert abs(p_bar - p_eur) < 0.1


def test_in_out_parity_zero_rebate():
    lv, _ = _lv()
    T, dt, rf, cf = _grid()
    ko = price_barrier_lv_pde(100., 100., True, T, lv, dt, rf, cf, barrier=120., is_up=True, is_out=True,
                              continuous=True, n_s=400)
    ki = price_barrier_lv_pde(100., 100., True, T, lv, dt, rf, cf, barrier=120., is_up=True, is_out=False,
                              continuous=True, n_s=400)
    van = price_european_lv_pde(100., 100., True, T, lv, dt, rf, cf, n_s=400)
    assert abs((ko + ki) - van) < 0.05


def test_mc_pde_agreement():
    lv, _ = _lv()
    T, dt, rf, cf = _grid()
    df = np.exp(-0.03 * T)
    p_pde = price_barrier_lv_pde(100., 100., True, T, lv, dt, rf, cf, barrier=125., is_up=True,
                                 is_out=True, continuous=True, n_s=500)
    p_mc, se = price_barrier_lv_mc(100., 100., True, lv, dt, rf, cf, df, barrier=125., is_up=True,
                                   is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                   num_paths=150_000, seed=11, return_stderr=True)
    assert abs(p_pde - p_mc) < max(0.25, 4 * se)
