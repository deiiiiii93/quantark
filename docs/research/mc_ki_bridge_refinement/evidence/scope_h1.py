"""Scope of the continuous-KI bridge bias and of the H1 fix (finer bridge nodes). No quantark edits.

H1 = the continuous-KI node set built on `factor` x bus_days_in_year (bridge intervals and SDE steps
shrink together), applied through an environment proxy that only the node-set builder reads.

Families (bp of notional vs the family's PDE; cost = wall per seed / base wall per seed):
  LV snowball C   1y continuous KI, monthly KO, AH + Dupire fixture surfaces
  LV phoenix      2w continuous KI, spot 7% above KI, AH + Dupire
  Heston snowball 2w A1 analogue, synthetic skewed Heston (rho -0.7)
  SLV snowball    same, smile leverage
Heston/SLV references: 2D ADI PDE at two rungs, plus bridge-free MC (dense DISCRETE KI at 240/960/1920
dates) extrapolated linearly in sqrt(dt) to continuous monitoring.
"""
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/fuxinyao/quant-ark/test/volmodels")
sys.path.insert(0, str(Path(__file__).parent))
import test_andreasen_huge_repricing as rep  # noqa: E402
from diagnose_gap import pde_params, snowball  # noqa: E402

from quantark.asset.equity.engine.mc import (  # noqa: E402
    HestonSLVSnowballMCEngine, HestonSnowballMCEngine, LocalVolSnowballMCEngine,
)
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import LocalVolPhoenixMCEngine  # noqa: E402
from quantark.asset.equity.engine.pde import (  # noqa: E402
    HestonSLVSnowballPDESolver, HestonSnowballPDESolver, LocalVolSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver  # noqa: E402
from quantark.asset.equity.param import MCParams  # noqa: E402
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import BarrierConfig  # noqa: E402
from quantark.asset.equity.product.option.snowball_option import SnowballOption  # noqa: E402
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402
from quantark.volmodels.heston import HestonParams  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol, calibrate_andreasen_huge  # noqa: E402
from quantark.volmodels.slv.leverage import LeverageSurface  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "scope_h1.json"
RESULTS = json.loads(OUT.read_text()) if OUT.exists() else {}


class _EnvProxy:
    def __init__(self, env, factor):
        self._env, self._factor = env, factor

    def __getattr__(self, name):
        value = getattr(self._env, name)
        return value * self._factor if name == "bus_days_in_year" else value


_PATCHED = {}


def patched(cls):
    if cls not in _PATCHED:
        class H1(cls):
            ki_factor = 1

            def _build_time_grid(self, product, pricing_env, T):
                env = pricing_env if self.ki_factor == 1 else _EnvProxy(pricing_env, self.ki_factor)
                return super()._build_time_grid(product, env, T)
        H1.__name__ = f"H1{cls.__name__}"
        _PATCHED[cls] = H1
    return _PATCHED[cls]


def mc(make, product, env, *, paths, seeds, factor=1):
    pvs, ses, walls = [], [], []
    for seed in seeds:
        eng = make(MCParams(num_paths=paths, seed=seed, use_antithetic=True))
        eng.ki_factor = factor
        t0 = time.perf_counter()
        pvs.append(eng.price(product, env))
        walls.append(time.perf_counter() - t0)
        ses.append(eng.get_last_std_error())
    return float(np.mean(pvs)), math.sqrt(sum(s * s for s in ses)) / len(ses), float(np.mean(walls))


def record(key, value):
    RESULTS[key] = value
    OUT.write_text(json.dumps(RESULTS, indent=1))
    print(key, value, flush=True)


def family(tag, make, product, env, ref, *, paths, seeds, factors, notional):
    base_wall = None
    for f in factors:
        key = f"{tag} | H1 x{f}"
        if key in RESULTS:
            base_wall = base_wall or RESULTS[f"{tag} | H1 x1"]["wall"]
            continue
        pv, se, wall = mc(make, product, env, paths=paths, seeds=seeds, factor=f)
        base_wall = base_wall or wall
        record(key, dict(err_bp=1e4 * (pv - ref) / notional, se_bp=1e4 * se / notional, wall=wall,
                         cost=wall / base_wall))


def discrete_copy(product, n, T):
    bc = product.barrier_config
    return SnowballOption(
        initial_price=product.initial_price, strike=product.strike, maturity=T, contract_multiplier=1.0,
        is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=bc.ko_barrier, ko_rate=bc.ko_rate,
                                     ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(bc.ko_observation_dates), ki_barrier=bc.ki_barrier,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=list(np.linspace(0.0, T, n + 1)[1:])))


def main():
    grid, spot, q = rep._fixture()
    R = rep.R
    env = rep._flat_env(spot, q, grid)
    bday = 1.0 / float(env.bus_days_in_year)
    lvs = {
        "AH": calibrate_andreasen_huge(grid, spot, FlatRateCurve(R), lambda t: q).to_local_vol_surface(),
        "Dupire": build_dupire_local_vol(grid, spot=spot, rate_curve=FlatRateCurve(R), div_yield=lambda t: q),
    }
    T2w = 14 / 365

    # --- LV phoenix, 2w near KI -------------------------------------------------------------------
    initial = spot / 0.86
    phoenix = create_standard_phoenix(initial_price=initial, strike=initial, maturity=T2w, ko_barrier=1.03 * initial,
                                      ko_rate=0.15, ki_barrier=0.80 * initial, coupon_barrier=0.85 * initial,
                                      coupon_rate=0.01, num_observations=2)
    for name, lv in lvs.items():
        ref = LocalVolPhoenixPDESolver(pde_params(3200, 64), local_vol_surface=lv).price(phoenix, env)
        record(f"LV phoenix {name} | PDE rung 6400-3200 bp", 1e4 * (LocalVolPhoenixPDESolver(
            pde_params(6400, 128), local_vol_surface=lv).price(phoenix, env) - ref) / initial)
        family(f"LV phoenix {name}", lambda p, lv=lv: patched(LocalVolPhoenixMCEngine)(p, local_vol_surface=lv),
               phoenix, env, ref, paths=1 << 17, seeds=range(8), factors=(1, 4, 16, 64), notional=initial)

    # --- Heston / SLV snowball, 2w near KI (synthetic) --------------------------------------------
    hp = HestonParams(v0=0.06, kappa=2.0, theta=0.06, sigma=0.5, rho=-0.7)
    s0 = 100.0
    henv = PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(s0), vol_surface=FlatVolSurface(math.sqrt(hp.v0)),
                              div_yield=ContinuousDividendYield(0.0))
    hprod = snowball(s0, T2w, i_over_s=0.86, ki="continuous", ko_times=[T2w], bday=bday)
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    smile = np.linspace(1.15, 0.90, strikes.size)
    lev = LeverageSurface(time_grid=np.linspace(0.0, 1.0, 4), strike_grid=strikes,
                          leverage_grid=np.vstack([smile, 0.75 * smile + 0.25, 0.5 * smile + 0.5,
                                                   np.ones_like(smile)]))
    from quantark.param import GridVolSurface
    flat_lv = build_dupire_local_vol(
        GridVolSurface(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))), [0.25, 0.5, 1.0], np.full((3, 11), 0.25)),
        spot=s0, rate_curve=FlatRateCurve(0.02), div_yield=lambda t: 0.0)  # unused: leverage is supplied
    models = {
        "Heston": (lambda p: patched(HestonSnowballMCEngine)(hp, p),
                   lambda n: HestonSnowballPDESolver(hp, n_x=n, n_v=n // 2, n_t=n)),
        "SLV": (lambda p: patched(HestonSLVSnowballMCEngine)(hp, params=p, leverage_surface=lev,
                                                             local_vol_surface=flat_lv),
                lambda n: HestonSLVSnowballPDESolver(hp, lev, n_x=n, n_v=n // 2, n_t=n)),
    }
    notional = hprod.initial_price
    for name, (make, make_pde) in models.items():
        key = f"{name} snowball | PDE"
        if key not in RESULTS:
            rungs = {}
            for n in (200, 400, 800):
                t0 = time.perf_counter()
                rungs[n] = make_pde(n).price(hprod, henv)
                print(name, "pde", n, rungs[n], time.perf_counter() - t0, flush=True)
            record(key, {str(k): v for k, v in rungs.items()})
        ref = RESULTS[key]["800"]
        family(f"{name} snowball", make, hprod, henv, ref, paths=1 << 17, seeds=range(8),
               factors=(1, 4, 16, 64), notional=notional)
        key = f"{name} snowball | bridge-free extrapolated"
        if key not in RESULTS:
            xs, ys = [], []
            for n in (240, 960, 1920):
                pv, se, _ = mc(make, discrete_copy(hprod, n, T2w), henv, paths=1 << 16, seeds=range(16))
                xs.append(math.sqrt(T2w / n)); ys.append(1e4 * (pv - ref) / notional)
                print(name, "discrete", n, ys[-1], 1e4 * se / notional, flush=True)
            record(key, dict(points=list(zip(xs, ys)), continuous_bp=float(np.polyfit(xs, ys, 1)[1])))

    # --- LV snowball C, 1y (the expensive family last) ------------------------------------------
    c = snowball(spot, 1.0, i_over_s=1.0, ki="continuous", ko_times=[m / 12 for m in range(1, 13)], bday=bday)
    for name, lv in lvs.items():
        ref = LocalVolSnowballPDESolver(pde_params(3200, 64), local_vol_surface=lv).price(c, env)
        family(f"LV snowball C 1y {name}", lambda p, lv=lv: patched(LocalVolSnowballMCEngine)(p, local_vol_surface=lv),
               c, env, ref, paths=1 << 15, seeds=range(16), factors=(1, 4, 16), notional=c.initial_price)


if __name__ == "__main__":
    main()
