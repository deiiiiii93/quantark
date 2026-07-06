"""Regression tests for the Codex code-review fixes: participation scales only the option
leg (not the rebate), and ObservationType.EXPIRY is supported. Flat-vol so the LV engines
collapse to Black-Scholes and can be checked against the analytical barrier engine."""
import numpy as np
from datetime import datetime

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.mc import LocalVolBarrierMCEngine
from quantark.asset.equity.engine.pde import LocalVolBarrierPDESolver
from quantark.util.enum import OptionType, BarrierType, ObservationType


def _env(vol=0.20, s0=100., r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.7, 0.7, 11)))
    surf = GridVolSurface(strikes, list(np.linspace(0.05, 2.0, 7)), np.full((7, 11), vol))
    return PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=ContinuousDividendYield(q))


def test_participation_scales_option_leg_only():
    # up-out call, participation 2.0, rebate 5.0 (paid at maturity). The rebate leg must NOT
    # be doubled: compare against the analytical engine which encodes the correct contract.
    prod = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=125.,
                         barrier_type=BarrierType.UP_OUT, rebate=5.0, participation_rate=2.0,
                         observation_type=ObservationType.CONTINUOUS)
    ana = BarrierAnalyticalEngine().price(prod, _env())
    pde = LocalVolBarrierPDESolver(PDEParams(grid_size=600, time_steps=200)).price(prod, _env())
    mc = LocalVolBarrierMCEngine(MCParams(num_paths=200_000, time_steps=150, seed=9)).price(prod, _env())
    assert abs(pde - ana) < 0.6
    assert abs(mc - ana) < 0.8
    # sanity: doubling the whole price (wrong) would be far off — participation-on-rebate bug guard
    naive_wrong = 2.0 * BarrierAnalyticalEngine().price(
        BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=125.,
                      barrier_type=BarrierType.UP_OUT, rebate=5.0, participation_rate=1.0,
                      observation_type=ObservationType.CONTINUOUS), _env())
    assert abs(pde - naive_wrong) > 1.0  # the correct price is NOT the naive double


def test_expiry_monitoring_supported():
    # EXPIRY: barrier checked only at maturity -> call knocked out iff S_T >= B.
    prod = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=120.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.EXPIRY)
    ana = BarrierAnalyticalEngine().price(prod, _env())
    pde = LocalVolBarrierPDESolver(PDEParams(grid_size=600, time_steps=150)).price(prod, _env())
    mc = LocalVolBarrierMCEngine(MCParams(num_paths=200_000, time_steps=100, seed=4)).price(prod, _env())
    assert abs(pde - ana) < 0.5
    assert abs(mc - ana) < 0.6
