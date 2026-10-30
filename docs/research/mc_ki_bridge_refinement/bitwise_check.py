"""Default-price byte compare for the continuous-KI bridge refinement (AC1).

Prices six MC families with their default constructors on a small synthetic environment and writes the
floats (repr, so every bit is kept) to JSON. Run it on the base commit and on the branch, on the same
machine; the two JSON files must be identical. The branch run also prices each family with the new
keywords passed explicitly at their defaults, which must equal the plain default.

    env PYTHONPATH=<tree>:<tree>/test python bitwise_check.py <tree> OUT.json

The script refuses to run unless ``quantark`` is imported from ``<tree>``: a .pth in the shared venv
imports the main checkout's package at startup, and a compare of main against main proves nothing.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

import quantark
from quantark.asset.equity.engine.mc import (
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import HestonSLVQESnowballMCEngine, QESnowballMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import HestonMCScheme, MonteCarloMethod
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv.leverage import LeverageSurface

S0 = 100.0
PARAMS = dict(num_paths=4096, seed=7, use_antithetic=True)


def environment():
    strikes = list(S0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = [0.1, 0.25, 0.5, 1.0]
    skew = 0.22 - 0.10 * np.linspace(-0.5, 0.5, 9)
    surface = GridVolSurface(strikes, maturities, np.tile(skew, (len(maturities), 1)))
    return PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=S0), vol_surface=surface,
                              div_yield=ContinuousDividendYield(0.01))


def products():
    dates = [0.25, 0.5, 0.75, 1.0]
    snowball = SnowballOption(
        initial_price=S0, strike=S0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=dates, ki_barrier=80.0,
                                     ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True))
    discrete = SnowballOption(
        initial_price=S0, strike=S0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=dates, ki_barrier=80.0,
                                     ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=dates))
    ko_reset = KnockOutResetSnowballOption(
        initial_price=S0, strike=S0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=dates,
                                     ki_barrier=80.0, ki_continuous=True),
        post_barrier_config=BarrierConfig(ko_barrier=98.0, ko_rate=0.03, ko_observation_dates=dates))
    phoenix = create_standard_phoenix(initial_price=S0, strike=S0, maturity=1.0, ko_barrier=103.0, ko_rate=0.0,
                                      ki_barrier=80.0, coupon_barrier=85.0, coupon_rate=0.01, num_observations=4)
    return snowball, ko_reset, phoenix, discrete


def engines(env, explicit):
    lv = build_dupire_local_vol(env.vol_surface, spot=env.spot, rate_curve=env.rate_curve,
                                div_yield=env.get_div_yield)
    hp = HestonParams(v0=0.06, kappa=2.0, theta=0.06, sigma=0.5, rho=-0.7)
    strikes = np.array(list(S0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    smile = np.linspace(1.15, 0.90, strikes.size)
    lev = LeverageSurface(time_grid=np.linspace(0.0, 1.0, 4), strike_grid=strikes,
                          leverage_grid=np.vstack([smile, 0.75 * smile + 0.25, 0.5 * smile + 0.5,
                                                   np.ones_like(smile)]))
    bridge = dict(ki_bridge_nodes_per_day=1) if explicit else {}
    cap = dict(max_sde_step_days=None) if explicit else {}
    p = lambda **kw: MCParams(**{**PARAMS, **kw})  # noqa: E731
    qmc = dict(use_antithetic=False)
    slv = dict(leverage_surface=lev, local_vol_surface=lv)
    return {
        "bsm_snowball": (lambda: SnowballMCEngine(p(), **bridge), 0),
        "bsm_snowball_discrete": (lambda: SnowballMCEngine(p(), **bridge), 3),
        "bsm_ko_reset": (lambda: SnowballMCEngine(p(), **bridge), 1),
        "bsm_phoenix": (lambda: PhoenixMCEngine(p(), **bridge), 2),
        "lv_snowball": (lambda: LocalVolSnowballMCEngine(p(), local_vol_surface=lv, **bridge, **cap), 0),
        "lv_snowball_discrete": (lambda: LocalVolSnowballMCEngine(p(), local_vol_surface=lv, **bridge, **cap), 3),
        "lv_ko_reset": (lambda: LocalVolSnowballMCEngine(p(), local_vol_surface=lv, **bridge, **cap), 1),
        "lv_snowball_quasi": (lambda: LocalVolSnowballMCEngine(p(**qmc), method=MonteCarloMethod.QUASI,
                                                               local_vol_surface=lv, **bridge), 0),
        "lv_snowball_rqmc": (lambda: LocalVolSnowballMCEngine(p(**qmc), method=MonteCarloMethod.RANDOMIZED_QUASI,
                                                              local_vol_surface=lv, **bridge), 0),
        "lv_phoenix": (lambda: LocalVolPhoenixMCEngine(p(), local_vol_surface=lv, **bridge, **cap), 2),
        "heston_qe_snowball": (lambda: HestonSnowballMCEngine(hp, p(), **bridge, **cap), 0),
        "heston_eulerlog_snowball": (lambda: HestonSnowballMCEngine(hp, p(), scheme=HestonMCScheme.EULERLOG,
                                                                 **bridge, **cap), 0),
        "heston_qem_snowball": (lambda: HestonSnowballMCEngine(hp, p(), scheme=HestonMCScheme.QUADEXP_M,
                                                               **bridge, **cap), 0),
        "qe_standalone_snowball": (lambda: QESnowballMCEngine(hp, p(), **bridge, **cap), 0),
        "heston_phoenix": (lambda: HestonPhoenixMCEngine(hp, p(), **bridge, **cap), 2),
        "slv_snowball": (lambda: HestonSLVSnowballMCEngine(hp, params=p(), **slv, **bridge, **cap), 0),
        "slv_qe_snowball": (lambda: HestonSLVQESnowballMCEngine(hp, params=p(), **slv, **bridge, **cap), 0),
        "slv_phoenix": (lambda: HestonSLVPhoenixMCEngine(hp, params=p(), **slv, **bridge, **cap), 2),
    }


def main(tree, out):
    tree = os.path.realpath(tree)
    imported = os.path.realpath(quantark.__file__)
    if not imported.startswith(tree + os.sep):
        raise SystemExit(f"quantark imported from {imported}, not from the measured tree {tree}")
    env = environment()
    prods = products()
    result = {}
    for explicit in (False, True):
        prices = {}
        for name, (make, i) in engines(env, explicit).items():
            try:
                prices[name] = repr(float(make().price(prods[i], env)))
            except TypeError as exc:      # base: the keywords do not exist yet
                prices[name] = f"unsupported: {exc}"
        result["explicit_defaults" if explicit else "defaults"] = prices
        print(result, flush=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
