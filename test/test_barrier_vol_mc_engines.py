import numpy as np
import pytest
from datetime import datetime

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.engine.mc import (
    LocalVolBarrierMCEngine, HestonBarrierMCEngine, HestonSLVBarrierMCEngine,
)
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.util.enum import OptionType, BarrierType, ObservationType
from quantark.util.exceptions import PricingError, ValidationError


def _env(vol=0.20, s0=100., r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    surf = GridVolSurface(strikes, list(np.linspace(0.1, 2.0, 6)), np.full((6, 9), vol))
    return PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=ContinuousDividendYield(q))


def _uo_call():
    return BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS)


def _unit_lev(s0=100.):
    ks = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(time_grid=np.linspace(0, 1, 6), strike_grid=ks, leverage_grid=np.ones((6, ks.size)))


def test_lv_barrier_engine_matches_analytical():
    px = LocalVolBarrierMCEngine(MCParams(num_paths=120_000, time_steps=100, seed=7)).price(_uo_call(), _env())
    ana = BarrierAnalyticalEngine().price(_uo_call(), _env())
    assert abs(px - ana) < 0.4


def test_participation_rate_scales_price():
    prod2 = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                          barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS,
                          participation_rate=2.0)
    eng = LocalVolBarrierMCEngine(MCParams(num_paths=40_000, time_steps=80, seed=1))
    assert eng.price(prod2, _env()) == pytest.approx(2.0 * eng.price(_uo_call(), _env()), rel=1e-12)


def test_rejects_non_barrier_and_schedule():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    eng = LocalVolBarrierMCEngine(MCParams(num_paths=1000))
    with pytest.raises(PricingError):
        eng.price(EuropeanVanillaOption(strike=100., option_type=OptionType.CALL, maturity=1.0), _env())


def test_heston_and_slv_barrier_engines_run():
    hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
    ph = HestonBarrierMCEngine(hp, MCParams(num_paths=40_000, time_steps=80, seed=2)).price(_uo_call(), _env())
    ps = HestonSLVBarrierMCEngine(hp, _unit_lev(), MCParams(num_paths=40_000, time_steps=80, seed=2)).price(_uo_call(), _env())
    assert ph > 0 and ps > 0
