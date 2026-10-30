"""Phase-1 evidence, step 2: which continuous-KI treatment is wrong on a skewed LV surface?

A1 (2w, spot 7% above KI). Bridge-free reference: MC with DISCRETE KI at n = 240/480/960/1920 equally
spaced dates (one SDE step per date, no bridge), extrapolated to continuous monitoring linearly in
sqrt(dt) (Broadie-Glasserman-Kou: discrete -> continuous error ~ c*sigma*sqrt(dt)).
Compared with: MC continuous KI (bridge) at 240/960 fine steps; PDE continuous KI; PDE discrete KI at
240/960 dates. Errors in bp of notional vs PDE continuous 3200x64.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "test" / "volmodels"))
sys.path.insert(0, str(Path(__file__).parent))
import test_andreasen_huge_repricing as rep  # noqa: E402
from diagnose_gap import pde_params, snowball  # noqa: E402

from quantark.asset.equity.engine.mc import LocalVolSnowballMCEngine  # noqa: E402
from quantark.asset.equity.engine.pde import LocalVolSnowballPDESolver  # noqa: E402
from quantark.asset.equity.param import MCParams  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import BarrierConfig  # noqa: E402
from quantark.asset.equity.product.option.snowball_option import SnowballOption  # noqa: E402
from quantark.param import FlatRateCurve, GridVolSurface  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol, calibrate_andreasen_huge  # noqa: E402

HERE = Path(__file__).parent
PATHS = 1 << 17
SEEDS = (11, 23, 37, 53, 71, 89, 101, 131)
T = 14 / 365


def a1_discrete(spot, n):
    initial = spot / 0.86
    return SnowballOption(
        initial_price=initial, strike=initial, maturity=T, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=1.03 * initial, ko_rate=0.15,
                                     ko_observation_type=ObservationType.DISCRETE, ko_observation_dates=[T],
                                     ki_barrier=0.80 * initial, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=list(np.linspace(0.0, T, n + 1)[1:])))


def mc(lv, product, env, substeps=1):
    pvs, ses = [], []
    for seed in SEEDS:
        eng = LocalVolSnowballMCEngine(MCParams(num_paths=PATHS, seed=seed, use_antithetic=True),
                                       local_vol_surface=lv, substeps_per_interval=substeps)
        pvs.append(eng.price(product, env))
        ses.append(eng.get_last_std_error())
    return float(np.mean(pvs)), math.sqrt(sum(s * s for s in ses)) / len(ses)


def main():
    grid, spot, q = rep._fixture()
    R = rep.R
    env = rep._flat_env(spot, q, grid)
    bday = 1.0 / float(env.bus_days_in_year)
    flat_grid = GridVolSurface(list(np.asarray(grid.strikes, float)), list(np.asarray(grid.maturities, float)),
                               np.full(np.asarray(grid.iv_grid).shape, 0.25))
    surfaces = {
        "Dupire": build_dupire_local_vol(grid, spot=spot, rate_curve=FlatRateCurve(R), div_yield=lambda t: q),
        "AH": calibrate_andreasen_huge(grid, spot, FlatRateCurve(R), lambda t: q).to_local_vol_surface(),
        "flat": build_dupire_local_vol(flat_grid, spot=spot, rate_curve=FlatRateCurve(R), div_yield=lambda t: q),
    }
    cont = snowball(spot, T, i_over_s=0.86, ki="continuous", ko_times=[T], bday=bday)
    notional = cont.initial_price
    out = {}
    for tag, lv in surfaces.items():
        ref = LocalVolSnowballPDESolver(pde_params(3200, 64), local_vol_surface=lv).price(cont, env)
        bp = lambda v: 1e4 * (v - ref) / notional  # noqa: E731
        row = {"pde_cont_6400x128": bp(LocalVolSnowballPDESolver(pde_params(6400, 128),
                                                                 local_vol_surface=lv).price(cont, env))}
        for n in (240, 960):
            row[f"pde_discrete_{n}"] = bp(LocalVolSnowballPDESolver(pde_params(3200, 64),
                                                                    local_vol_surface=lv).price(a1_discrete(spot, n), env))
        n_contract = len(LocalVolSnowballMCEngine(MCParams(), local_vol_surface=lv)._build_time_grid(cont, env, T)[1])
        for sub in (24, 96):
            pv, se = mc(lv, cont, env, substeps=sub)
            row[f"mc_bridge_{n_contract * sub}"] = (bp(pv), 1e4 * se / notional)
        xs, ys = [], []
        for n in (240, 480, 960, 1920):
            pv, se = mc(lv, a1_discrete(spot, n), env)
            row[f"mc_discrete_{n}"] = (bp(pv), 1e4 * se / notional)
            xs.append(math.sqrt(T / n)); ys.append(bp(pv))
            print(tag, n, row[f"mc_discrete_{n}"], flush=True)
        slope, intercept = np.polyfit(xs, ys, 1)
        row["mc_discrete_extrapolated_continuous"] = float(intercept)
        row["bgk_slope_bp_per_sqrt_yr"] = float(slope)
        out[tag] = row
        print(tag, json.dumps(row), flush=True)
        (HERE / "diagnose_ki.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
