"""Andreasen–Huge repricing acceptance on the production local-vol PDE (spec AC2–AC5).

Each quoted European put is priced by ``LocalVolPDESolver`` on the sampled AH surface and inverted to
Black–Scholes implied vol (zero rates at the option's maturity); the gap to the quoted vol must stay
within 0.05 vol points. The PDE grid is the one T0 measured as converged (one more refinement moves
no gap by more than 0.005 vol points).
"""
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr

from quantark.asset.equity.engine.pde.grid.config import GridConfig as PDEGrid
from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.volmodels.localvol import (
    AndreasenHugeConfig,
    build_andreasen_huge_local_vol,
    build_dupire_local_vol,
    calibrate_andreasen_huge,
)

FIXTURE = Path(__file__).resolve().parents[1] / "golden/modelvalidation/data/iv_surface_20240208.json"
VALUATION = datetime(2024, 2, 8)
R = 0.02
BAR_VOL_POINTS = 0.05
PDE_POINTS, PDE_STEPS_PER_DAY = 3200, 32       # T0: converged rung (see docs/research/dupire_andreasen_huge)


@lru_cache(maxsize=1)
def _fixture():
    art = IvSurfaceArtifact.from_file(FIXTURE)
    q = float(art.implied_q_pillars(R)[1][-1])
    return art.grid_vol_surface(), float(art.s0), q


def _params(points=PDE_POINTS, spd=PDE_STEPS_PER_DAY):
    return PDEParams(accuracy="high", grid=PDEGrid(points=points, steps_per_day=spd,
                                                    max_points=20000, max_steps=400000))


def _bs_put(s, k, t, r, q, v):
    sd = v * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q) * t + 0.5 * sd * sd) / sd
    return k * math.exp(-r * t) * ndtr(-(d1 - sd)) - s * math.exp(-q * t) * ndtr(-d1)


def _implied(price, s, k, t, r, q):
    return brentq(lambda v: _bs_put(s, k, t, r, q, v) - price, 1e-4, 5.0, xtol=1e-14, rtol=1e-14, maxiter=200)


def _gaps_vol_points(surface, env, quotes, spot, r_of, q_of):
    """quotes: iterable of (expiry, strike, quoted iv) -> gaps in vol points."""
    params = _params()
    out = []
    for t, k, iv in quotes:
        product = EuropeanVanillaOption(strike=float(k), option_type=OptionType.PUT, maturity=float(t))
        price = LocalVolPDESolver(params=params, local_vol_surface=surface).price(product, env)
        out.append(100.0 * (_implied(price, spot, k, t, r_of(t), q_of(t)) - iv))
    return np.asarray(out)


def _flat_env(spot, q, vol_surface):
    return PricingEnvironment(rate_curve=FlatRateCurve(R), valuation_date=VALUATION, spot_quote=SpotQuote(spot),
                              vol_surface=vol_surface, div_yield=ContinuousDividendYield(div_yield=q))


@lru_cache(maxsize=1)
def _fixture_ah_surface():
    grid, spot, q = _fixture()
    return build_andreasen_huge_local_vol(grid, spot, FlatRateCurve(R), lambda t: q)


@pytest.mark.slow
@pytest.mark.parametrize("expiry_index", range(6))
def test_every_fixture_put_reprices_within_the_bar(expiry_index):
    grid, spot, q = _fixture()
    t = float(grid.maturities[expiry_index])
    iv = np.asarray(grid.iv_grid, dtype=float)[expiry_index]
    quotes = [(t, k, v) for k, v in zip(np.asarray(grid.strikes, dtype=float), iv)]
    gaps = _gaps_vol_points(_fixture_ah_surface(), _flat_env(spot, q, grid), quotes, spot,
                            lambda _: R, lambda _: q)
    assert gaps.size == 33
    assert np.max(np.abs(gaps)) <= BAR_VOL_POINTS, (t, gaps.round(4).tolist())


@pytest.mark.slow
def test_the_dupire_builder_fails_the_same_check():
    """Control: the default builder overprices the 8d quotes by more than 10 vol points on average."""
    grid, spot, q = _fixture()
    dupire = build_dupire_local_vol(grid, spot=spot, rate_curve=FlatRateCurve(R), div_yield=lambda t: q)
    t = float(grid.maturities[0])
    iv = np.asarray(grid.iv_grid, dtype=float)[0]
    strikes = np.asarray(grid.strikes, dtype=float)
    quotes = [(t, strikes[j], iv[j]) for j in range(0, strikes.size, 4)]
    gaps = _gaps_vol_points(dupire, _flat_env(spot, q, grid), quotes, spot, lambda _: R, lambda _: q)
    assert float(np.mean(gaps)) > 10.0, gaps.round(3).tolist()


@pytest.mark.slow
def test_quote_rule_surface_reprices_its_included_quotes():
    grid, spot, q = _fixture()
    flat = GridVolSurface(list(grid.strikes), list(grid.maturities), np.full(np.shape(grid.iv_grid), 0.25))
    config = AndreasenHugeConfig(max_abs_d=4.0)
    ah = calibrate_andreasen_huge(flat, spot, FlatRateCurve(R), lambda t: q, config)
    surface = ah.to_local_vol_surface()
    strikes = np.asarray(flat.strikes, dtype=float)
    quotes = []
    for i, t in enumerate(flat.maturities[:2]):                      # 8d (quotes excluded) and 36d
        t = float(t)
        f = ah.forward(t)
        d = np.abs(np.log(strikes / f)) / (0.25 * math.sqrt(t))
        quotes += [(t, k, 0.25) for k in strikes[d <= 4.0][::3]]
    gaps = _gaps_vol_points(surface, _flat_env(spot, q, flat), quotes, spot, lambda _: R, lambda _: q)
    assert np.max(np.abs(gaps)) <= BAR_VOL_POINTS, gaps.round(4).tolist()


def _term_curves():
    pillars = [0.02, 0.25, 0.5, 1.0]
    rate = LinearRateCurve(list(zip(pillars, [0.01, 0.02, 0.025, 0.03])))
    div = TermStructureDividendYield(times=pillars, yields=[0.09, 0.07, 0.06, 0.05])
    return rate, div


@pytest.mark.slow
def test_term_structure_carry_reprices_with_interpolated_zero_curves():
    grid, spot, _ = _fixture()
    rate, div = _term_curves()
    ah = calibrate_andreasen_huge(grid, spot, rate, div.get_yield)
    surface = ah.to_local_vol_surface()
    env = PricingEnvironment(rate_curve=rate, valuation_date=VALUATION, spot_quote=SpotQuote(spot),
                             vol_surface=grid, div_yield=div)
    strikes = np.asarray(grid.strikes, dtype=float)
    iv = np.asarray(grid.iv_grid, dtype=float)
    quotes = [(float(t), strikes[j], iv[i, j]) for i, t in enumerate(grid.maturities)
              for j in range(0, strikes.size, 4)]
    gaps = _gaps_vol_points(surface, env, quotes, spot, rate.get_rate, div.get_yield)
    assert np.max(np.abs(gaps)) <= BAR_VOL_POINTS, gaps.round(4).tolist()


def test_term_structure_sampled_rows_carry_the_forward_moneyness_map():
    grid, spot, _ = _fixture()
    rate, div = _term_curves()
    ah = calibrate_andreasen_huge(grid, spot, rate, div.get_yield)
    surface = ah.to_local_vol_surface()
    t0, t1 = float(grid.maturities[2]), float(grid.maturities[3])
    k = int(np.argmin(np.abs(surface.time_grid - 0.5 * (t0 + t1))))
    t = float(surface.time_grid[k])
    f_env = spot * math.exp((rate.get_rate(t) - div.get_yield(t)) * t)
    mapped = ah._calibration_local_vol(surface.strike_grid * ah.forward(t) / f_env, t)
    assert np.allclose(surface.lv_grid[k], mapped, rtol=1e-14, atol=0.0)
    assert not np.allclose(surface.lv_grid[k], ah._calibration_local_vol(surface.strike_grid, t), rtol=1e-6)
