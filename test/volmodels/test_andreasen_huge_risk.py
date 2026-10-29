"""Opt-in Andreasen–Huge construction in the structured vol-model risk calculators (spec D10, AC8–AC9)."""
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from quantark.asset.equity.engine.pde import LocalVolPDESolver
from quantark.asset.equity.engine.pde.grid.config import GridConfig as PDEGrid
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.vol_model_risk import VolModelRiskCalculator
from quantark.asset.fx.riskmeasures.vol_model_risk import FxVolModelRiskCalculator
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol import (
    AndreasenHugeConfig,
    LocalVolConstruction,
    LocalVolSurface,
    build_andreasen_huge_local_vol,
)
from quantark.volmodels.risk import MarketVegaRequest, ModelRiskRequest, SurfaceBump, bump_local_vol_surface

FIXTURE = Path(__file__).resolve().parents[1] / "golden/modelvalidation/data/iv_surface_20240208.json"
R = 0.02
AH = LocalVolConstruction.ANDREASEN_HUGE


@lru_cache(maxsize=1)
def _fixture():
    art = IvSurfaceArtifact.from_file(FIXTURE)
    q = float(art.implied_q_pillars(R)[1][-1])
    return art.grid_vol_surface(), float(art.s0), q


def _env():
    grid, spot, q = _fixture()
    return PricingEnvironment(rate_curve=FlatRateCurve(R), valuation_date=datetime(2024, 2, 8),
                              spot_quote=SpotQuote(spot), vol_surface=grid, div_yield=ContinuousDividendYield(q))


def _engine(surface=None, points=1600, spd=8):
    params = PDEParams(accuracy="high", grid=PDEGrid(points=points, steps_per_day=spd,
                                                      max_points=20000, max_steps=400000))
    return LocalVolPDESolver(params=params, local_vol_surface=surface)


def _quoted_put(expiry_index=3, strike_index=23):
    grid, _, _ = _fixture()
    t = float(grid.maturities[expiry_index])
    k = float(grid.strikes[strike_index])
    iv = float(np.asarray(grid.iv_grid)[expiry_index, strike_index])
    return EuropeanVanillaOption(strike=k, option_type=OptionType.PUT, maturity=t), t, k, iv


def _bs_put(s, k, t, r, q, v):
    sd = v * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q) * t + 0.5 * sd * sd) / sd
    return k * math.exp(-r * t) * ndtr(-(d1 - sd)) - s * math.exp(-q * t) * ndtr(-d1)


def test_construction_switch_is_validated():
    with pytest.raises(ValidationError):
        VolModelRiskCalculator(local_vol_construction="andreasen_huge")
    with pytest.raises(ValidationError):
        VolModelRiskCalculator(andreasen_huge_config=AndreasenHugeConfig())       # config without the switch
    VolModelRiskCalculator(local_vol_construction=AH, andreasen_huge_config=AndreasenHugeConfig())


@pytest.mark.slow
def test_market_vega_on_andreasen_huge_is_the_quoted_black_scholes_vega():
    """AH reprices the base and both bumped quote surfaces, so its market-vega pnl is the BS half-difference."""
    product, t, k, iv = _quoted_put()
    grid, spot, q = _fixture()
    calc = VolModelRiskCalculator(local_vol_construction=AH)
    result = calc.calculate_market_vega(product, _env(), _engine(points=3200, spd=32),
                                        MarketVegaRequest(surface_bumps=(SurfaceBump.parallel(0.01),)))
    assert result.metadata["model"] == "andreasen_huge_local_vol"
    point = result.points[0]
    assert point.status == "ok", point.error
    expect = 0.5 * (_bs_put(spot, k, t, R, q, iv + 0.01) - _bs_put(spot, k, t, R, q, iv - 0.01))
    assert point.pnl == pytest.approx(expect, rel=0.02)


def test_default_calculator_keeps_the_dupire_label():
    strikes = list(100.0 * np.exp(np.linspace(-0.35, 0.35, 5)))
    grid = GridVolSurface(strikes, [0.25, 0.75, 1.5], np.full((3, 5), 0.2))
    env = PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=datetime(2026, 1, 1),
                             spot_quote=SpotQuote(100.0), vol_surface=grid, div_yield=ContinuousDividendYield(0.01))
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    engine = LocalVolPDESolver(params=PDEParams(grid=PDEGrid(points=100, time_steps=50)))
    result = VolModelRiskCalculator().calculate_market_vega(call, env, engine)
    assert result.metadata["model"] == "dupire_local_vol"


def test_model_risk_on_a_prebuilt_andreasen_huge_surface_bumps_it_directly():
    grid, spot, q = _fixture()
    surface = build_andreasen_huge_local_vol(grid, spot, FlatRateCurve(R), lambda t: q)
    product, _, _, _ = _quoted_put(expiry_index=2)
    engine = _engine(surface)
    bump = SurfaceBump.parallel(0.01)
    result = VolModelRiskCalculator().calculate_model_risk(
        product, _env(), engine, ModelRiskRequest(parameter_names=(), surface_bumps=(bump,)))
    point = result.points[0]
    assert point.status == "ok", point.error
    up = engine._price_with_surface(product, _env(), bump_local_vol_surface(surface, bump, 1))
    down = engine._price_with_surface(product, _env(), bump_local_vol_surface(surface, bump, -1))
    assert point.pnl == pytest.approx(0.5 * (up - down), rel=1e-12)


def test_switch_on_without_prebuilt_prices_model_risk_on_andreasen_huge():
    grid, spot, q = _fixture()
    product, _, _, _ = _quoted_put(expiry_index=2)
    engine = _engine()
    calc = VolModelRiskCalculator(local_vol_construction=AH)
    result = calc.calculate_model_risk(product, _env(), engine,
                                       ModelRiskRequest(parameter_names=(), surface_bumps=(SurfaceBump.parallel(0.01),)))
    assert result.metadata["model"] == "andreasen_huge_local_vol"
    ah_surface = build_andreasen_huge_local_vol(grid, spot, FlatRateCurve(R), lambda t: q)
    assert result.base_price == pytest.approx(engine._price_with_surface(product, _env(), ah_surface), rel=1e-12)


def test_fx_calculator_routes_the_switch_to_the_andreasen_huge_builder(monkeypatch):
    import quantark.asset.vol_model_risk as vmr

    calls = []
    real = vmr.build_andreasen_huge_local_vol

    def spy(*args, **kwargs):
        calls.append(kwargs.get("config"))
        return real(*args, **kwargs)

    monkeypatch.setattr(vmr, "build_andreasen_huge_local_vol", spy)
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=1.2),
        domestic_curve=FlatRateCurve(0.02), foreign_curve=FlatRateCurve(0.01),
        vol_surface=GridVolSurface([1.0, 1.1, 1.2, 1.3, 1.4], [0.5, 1.0, 1.5], np.full((3, 5), 0.12)))
    config = AndreasenHugeConfig(max_abs_d=6.0)
    calc = FxVolModelRiskCalculator(local_vol_construction=AH, andreasen_huge_config=config)
    surface = calc._build_local_vol(env, env.vol_surface)
    assert isinstance(surface, LocalVolSurface)
    assert calls == [config]
