import numpy as np
import pytest

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc, price_barrier_lv_mc
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.product.option import BarrierOption
from quantark.util.enum import OptionType, BarrierType, ObservationType
from datetime import datetime


def _flat_lv(vol=0.20, s0=100.0, r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 9), vol))
    env = PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                             spot_quote=SpotQuote(spot=s0), vol_surface=surf,
                             div_yield=ContinuousDividendYield(q))
    lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve, div_yield=env.get_div_yield)
    return lv, env


def _grid(T=1.0, n=100, r=0.03, q=0.01):
    dt = np.full(n, T / n)
    return dt, np.full(n, r), np.full(n, q), np.exp(-r * T)


def test_barrier_far_reproduces_european():
    lv, _ = _flat_lv()
    dt, rf, cf, df = _grid()
    p_eur = price_european_lv_mc(100., 100., True, lv, dt, rf, cf, df, num_paths=60_000, seed=1)
    p_bar = price_barrier_lv_mc(100., 100., True, lv, dt, rf, cf, df, barrier=1e7, is_up=True,
                                is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                num_paths=60_000, seed=1)
    assert abs(p_bar - p_eur) < 0.03


def test_up_out_call_matches_analytical():
    lv, env = _flat_lv()
    dt, rf, cf, df = _grid()
    prod = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS)
    ana = BarrierAnalyticalEngine().price(prod, env)
    p, se = price_barrier_lv_mc(100., 100., True, lv, dt, rf, cf, df, barrier=130., is_up=True,
                                is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                num_paths=120_000, seed=7, return_stderr=True)
    assert abs(p - ana) < max(0.15, 3 * se)


def test_in_out_parity_zero_rebate():
    lv, _ = _flat_lv()
    dt, rf, cf, df = _grid()
    common = dict(barrier=115., pay_at_hit=False, continuous=True, num_paths=80_000, seed=3, rebate=0.)
    ko = price_barrier_lv_mc(100., 100., True, lv, dt, rf, cf, df, is_up=True, is_out=True, **common)
    ki = price_barrier_lv_mc(100., 100., True, lv, dt, rf, cf, df, is_up=True, is_out=False, **common)
    van = price_european_lv_mc(100., 100., True, lv, dt, rf, cf, df, num_paths=80_000, seed=3)
    assert abs((ko + ki) - van) < 0.05
