import numpy as np
from datetime import datetime

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
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


def test_discrete_up_barrier_default_grid_places_barrier_midcell():
    lv = LocalVolSurface(np.array([1.0, 20_000.0]), np.array([0.0, 1.0]),
                         np.full((2, 2), 0.288))
    s0 = 8489.3
    strike = s0
    barrier = 1.10 * s0
    T = 0.452
    n_t = 260
    t_grid = np.linspace(0.0, T, n_t + 1)
    dt = np.diff(t_grid)
    rf = np.full(n_t, 0.0117)
    cf = np.full(n_t, 0.1345)
    obs = sorted({round(d, 6) for d in np.arange(1, int(T * 52) + 1) / 52.0 if d < T}
                 | {float(T)})
    observe_steps = [int(np.argmin(np.abs(t_grid - d))) for d in obs]

    default_500 = price_barrier_lv_pde(
        s0, strike, True, T, lv, dt, rf, cf, barrier=barrier, is_up=True,
        is_out=True, continuous=False, observe_steps=observe_steps, n_s=500,
    )
    refined = price_barrier_lv_pde(
        s0, strike, True, T, lv, dt, rf, cf, barrier=barrier, is_up=True,
        is_out=True, continuous=False, observe_steps=observe_steps, n_s=2000,
    )
    # Simulate the pre-fix production grid by explicitly freezing the unaligned far boundary.
    unaligned_smax = max(4.0 * max(s0, strike), 1.5 * barrier)
    unaligned_500 = price_barrier_lv_pde(
        s0, strike, True, T, lv, dt, rf, cf, barrier=barrier, is_up=True,
        is_out=True, continuous=False, observe_steps=observe_steps, n_s=500,
        s_max=unaligned_smax,
    )

    assert abs(default_500 - refined) < 0.15
    assert abs(unaligned_500 - refined) > 0.75
