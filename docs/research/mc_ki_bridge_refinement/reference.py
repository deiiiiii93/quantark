"""Frozen references for the KI-bridge refinement accuracy tests (AC6, AC7).

1. Heston and SLV bridge-free references (AC6). The A1 analogue (2w, spot 7% above KI, KO at maturity)
   is priced with DISCRETE knock-in at n = 240 / 960 / 1920 equally spaced dates -- no Brownian bridge
   anywhere -- and extrapolated to continuous monitoring linearly in sqrt(T/n) (Broadie-Glasserman-Kou:
   the discrete-to-continuous gap is c * sigma * sqrt(dt) to leading order). The intercept and its
   standard error (weighted least squares with the MC standard errors) are written to JSON.
2. LV case-B step-cap measurement (AC7): 1m, KI and KO at maturity only, on the AH and Dupire fixture
   surfaces, default vs max_sde_step_days = 1 and 0.1, against the LV PDE at 3200x64, with exactly the
   test protocol (seeds, paths). The AC7 bars are set from these numbers.

    env PYTHONPATH=<tree>:<tree>/test python reference.py OUT.json
"""
import json
import math
import sys
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.mc import HestonSLVSnowballMCEngine, HestonSnowballMCEngine, LocalVolSnowballMCEngine
from quantark.asset.equity.engine.pde import LocalVolSnowballPDESolver
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol, calibrate_andreasen_huge
from quantark.volmodels.slv.leverage import LeverageSurface

T_A1 = 14 / 365
T_B = 30 / 365
HESTON = HestonParams(v0=0.06, kappa=2.0, theta=0.06, sigma=0.5, rho=-0.7)
FIXTURE = "test/golden/modelvalidation/data/iv_surface_20240208.json"
R_FIXTURE = 0.02
SEEDS_A = (11, 23, 37, 53, 71, 89, 101, 131)
BRIDGE_FREE_DATES = (240, 960, 1920)
BRIDGE_FREE_SEEDS = tuple(range(16))
BRIDGE_FREE_PATHS = 1 << 16


def heston_env():
    return PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(math.sqrt(HESTON.v0)),
                              div_yield=ContinuousDividendYield(0.0))


def leverage():
    strikes = np.array(list(100.0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    smile = np.linspace(1.15, 0.90, strikes.size)
    return LeverageSurface(time_grid=np.linspace(0.0, 1.0, 4), strike_grid=strikes,
                           leverage_grid=np.vstack([smile, 0.75 * smile + 0.25, 0.5 * smile + 0.5,
                                                    np.ones_like(smile)]))


def slv_dummy_local_vol():
    """The SLV engine builds a local-vol surface even when its leverage is supplied; unused here."""
    s0 = 100.0
    grid = GridVolSurface(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))), [0.25, 0.5, 1.0], np.full((3, 11), 0.25))
    return build_dupire_local_vol(grid, spot=s0, rate_curve=FlatRateCurve(0.02), div_yield=lambda t: 0.0)


def snowball(spot, T, i_over_s, ki):
    initial = spot / i_over_s
    return SnowballOption(
        initial_price=initial, strike=initial, maturity=T, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=1.03 * initial, ko_rate=0.15,
                                     ko_observation_type=ObservationType.DISCRETE, ko_observation_dates=[T],
                                     ki_barrier=0.80 * initial, **ki))


def a1(spot, n_dates=None):
    if n_dates is None:
        return snowball(spot, T_A1, 0.86, dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True))
    dates = list(np.linspace(0.0, T_A1, n_dates + 1)[1:])
    return snowball(spot, T_A1, 0.86, dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=dates))


def case_b(spot):
    return snowball(spot, T_B, 0.90, dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=[T_B]))


def heston_models():
    lev, lv = leverage(), slv_dummy_local_vol()
    return {
        "heston": lambda p, **kw: HestonSnowballMCEngine(HESTON, p, **kw),
        "slv": lambda p, **kw: HestonSLVSnowballMCEngine(HESTON, params=p, leverage_surface=lev,
                                                         local_vol_surface=lv, **kw),
    }


def mean_price(make, product, env, seeds, paths, **kw):
    pvs, ses = [], []
    for seed in seeds:
        engine = make(MCParams(num_paths=paths, seed=seed, use_antithetic=True), **kw)
        pvs.append(engine.price(product, env))
        ses.append(engine.get_last_std_error())
    return float(np.mean(pvs)), math.sqrt(sum(s * s for s in ses)) / len(ses)


def bridge_free(make, env):
    xs, ys, ses = [], [], []
    for n in BRIDGE_FREE_DATES:
        pv, se = mean_price(make, a1(100.0, n), env, BRIDGE_FREE_SEEDS, BRIDGE_FREE_PATHS)
        xs.append(math.sqrt(T_A1 / n)); ys.append(pv); ses.append(se)
        print("bridge-free", n, pv, se, flush=True)
    x, y, w = np.array(xs), np.array(ys), 1.0 / np.array(ses) ** 2
    design = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(design.T @ (w[:, None] * design))
    intercept, slope = cov @ design.T @ (w * y)
    return dict(price=float(intercept), se=float(math.sqrt(cov[0, 0])), slope=float(slope),
                points=[dict(dates=n, sqrt_dt=a, price=b, se=c) for n, a, b, c in zip(BRIDGE_FREE_DATES, xs, ys, ses)])


def fixture():
    art = IvSurfaceArtifact.from_file(FIXTURE)
    q = float(art.implied_q_pillars(R_FIXTURE)[1][-1])
    grid, spot = art.grid_vol_surface(), float(art.s0)
    env = PricingEnvironment(rate_curve=FlatRateCurve(R_FIXTURE), valuation_date=datetime(2024, 2, 8),
                             spot_quote=SpotQuote(spot), vol_surface=grid,
                             div_yield=ContinuousDividendYield(div_yield=q))
    surfaces = {
        "AH": calibrate_andreasen_huge(grid, spot, FlatRateCurve(R_FIXTURE), lambda t: q).to_local_vol_surface(),
        "Dupire": build_dupire_local_vol(grid, spot=spot, rate_curve=FlatRateCurve(R_FIXTURE),
                                         div_yield=lambda t: q),
    }
    return env, spot, surfaces


def pde_params():
    return PDEParams(accuracy="high", grid=GridConfig(points=3200, steps_per_day=64, max_points=20000,
                                                      max_steps=800000))


def main(out):
    result = {"protocol": dict(bridge_free_dates=BRIDGE_FREE_DATES, bridge_free_seeds=list(BRIDGE_FREE_SEEDS),
                               bridge_free_paths=BRIDGE_FREE_PATHS, case_b_seeds=list(SEEDS_A),
                               case_b_paths=1 << 17, heston=HESTON.__dict__ if hasattr(HESTON, "__dict__") else str(HESTON))}
    env = heston_env()
    for name, make in heston_models().items():
        result[f"{name}_a1_bridge_free"] = bridge_free(make, env)
        print(name, result[f"{name}_a1_bridge_free"], flush=True)
    fenv, spot, surfaces = fixture()
    product = case_b(spot)
    for name, lv in surfaces.items():
        ref = LocalVolSnowballPDESolver(pde_params(), local_vol_surface=lv).price(product, fenv)
        row = dict(pde=ref)
        for label, kw in (("default", {}), ("cap_1", dict(max_sde_step_days=1)), ("cap_0.1", dict(max_sde_step_days=0.1))):
            pv, se = mean_price(lambda p, **k: LocalVolSnowballMCEngine(p, local_vol_surface=lv, **k), product, fenv,
                                SEEDS_A, 1 << 17, **kw)
            row[label] = dict(price=pv, se=se, err_bp=1e4 * (pv - ref) / product.initial_price,
                              se_bp=1e4 * se / product.initial_price)
            print(name, label, row[label], flush=True)
        result[f"lv_case_b_{name}"] = row
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
