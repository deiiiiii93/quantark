"""Phase-1 evidence: where does the fine-step LV MC vs LV PDE gap come from? (no quantark edits)

Fine MC = uniform substeps so every SDE step is <= ~0.06 day. Each row is the mean over 4 seeds
of 2^18 antithetic paths; err is MC - PDE(3200x64) in bp of notional (initial_price).

Controls:
  surface   flat 25% (constant LV; BSM engines give an independent reference), AH, Dupire
  product   A1 continuous KI | A1d daily DISCRETE KI on the same nodes (no bridge) | B (discrete, 1 node)
            | European puts at A1's KI level (14d) and B's KI level (30d)
  sampling  left (default) and mid at the same fine grid
  reference PDE 3200x64 and 6400x128 rungs
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/fuxinyao/quant-ark/test/volmodels")
import test_andreasen_huge_repricing as rep  # noqa: E402

from quantark.asset.equity.engine.mc import LocalVolSnowballMCEngine, SnowballMCEngine  # noqa: E402
from quantark.asset.equity.engine.pde import LocalVolSnowballPDESolver, SnowballPDESolver  # noqa: E402
from quantark.asset.equity.engine.pde.grid.config import GridConfig  # noqa: E402
from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver  # noqa: E402
from quantark.asset.equity.param import MCParams, PDEParams  # noqa: E402
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import BarrierConfig  # noqa: E402
from quantark.asset.equity.product.option.snowball_option import SnowballOption  # noqa: E402
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import ObservationType, OptionType  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol, calibrate_andreasen_huge  # noqa: E402
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc  # noqa: E402

HERE = Path(__file__).parent
PATHS = 1 << 18
SEEDS = (2029, 4111, 6337, 8443)
FINE_DT = 0.06 / 365.0
FLAT = 0.25


def pde_params(pts, spd):
    return PDEParams(accuracy="high", grid=GridConfig(points=pts, steps_per_day=spd,
                                                      max_points=20000, max_steps=800000))


def snowball(spot, T, *, i_over_s, ki, ko_times, bday):
    initial = spot / i_over_s
    if ki == "continuous":
        kw = dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    elif ki == "daily":
        n = int(T / bday) + 1                       # the continuous-KI node set of _build_time_grid
        kw = dict(ki_observation_type=ObservationType.DISCRETE,
                  ki_observation_dates=list(np.linspace(0.0, T, n + 1)[1:]))
    else:
        kw = dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=[T])
    return SnowballOption(
        initial_price=initial, strike=initial, maturity=T, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=1.03 * initial, ko_rate=0.15,
                                     ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_times), ki_barrier=0.80 * initial, **kw))


def mc_snowball(lv, product, env, sampling, substeps):
    pvs, ses = [], []
    for seed in SEEDS:
        eng = LocalVolSnowballMCEngine(MCParams(num_paths=PATHS, seed=seed, use_antithetic=True),
                                       local_vol_surface=lv, lv_time_sampling=sampling,
                                       substeps_per_interval=substeps)
        pvs.append(eng.price(product, env))
        ses.append(eng.get_last_std_error())
    return float(np.mean(pvs)), math.sqrt(sum(s * s for s in ses)) / len(ses)


def mc_bsm_snowball(product, env, substeps_unused):
    pvs, ses = [], []
    for seed in SEEDS:
        eng = SnowballMCEngine(MCParams(num_paths=PATHS, seed=seed, use_antithetic=True))
        pvs.append(eng.price(product, env))
        ses.append(eng.get_last_std_error())
    return float(np.mean(pvs)), math.sqrt(sum(s * s for s in ses)) / len(ses)


def mc_put(lv, spot, k, T, r, q, sampling):
    M = max(1, math.ceil(T / FINE_DT))
    pvs, ses = [], []
    for seed in SEEDS:
        p, se = price_european_lv_mc(spot, k, False, lv, np.full(M, T / M), np.full(M, r), np.full(M, q),
                                     math.exp(-r * T), num_paths=PATHS, seed=seed, use_antithetic=True,
                                     return_stderr=True, time_sampling=sampling)
        pvs.append(p); ses.append(se)
    return float(np.mean(pvs)), math.sqrt(sum(s * s for s in ses)) / len(ses), M


def main():
    grid, spot, q = rep._fixture()
    R = rep.R
    flat_grid = GridVolSurface(list(np.asarray(grid.strikes, float)), list(np.asarray(grid.maturities, float)),
                               np.full(np.asarray(grid.iv_grid).shape, FLAT))
    env = rep._flat_env(spot, q, grid)
    env_bsm = PricingEnvironment(rate_curve=FlatRateCurve(R), valuation_date=rep.VALUATION,
                                 spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(FLAT),
                                 div_yield=ContinuousDividendYield(div_yield=q))
    bday = 1.0 / float(env.bus_days_in_year)
    surfaces = {
        "flat": build_dupire_local_vol(flat_grid, spot=spot, rate_curve=FlatRateCurve(R), div_yield=lambda t: q),
        "AH": calibrate_andreasen_huge(grid, spot, FlatRateCurve(R), lambda t: q).to_local_vol_surface(),
        "Dupire": build_dupire_local_vol(grid, spot=spot, rate_curve=FlatRateCurve(R), div_yield=lambda t: q),
    }
    T_a, T_b = 14 / 365, 30 / 365
    products = {
        "A1 cont KI": snowball(spot, T_a, i_over_s=0.86, ki="continuous", ko_times=[T_a], bday=bday),
        "A1d daily discrete KI": snowball(spot, T_a, i_over_s=0.86, ki="daily", ko_times=[T_a], bday=bday),
        "B maturity-only KI": snowball(spot, T_b, i_over_s=0.90, ki="maturity", ko_times=[T_b], bday=bday),
    }
    puts = {"put 14d K=A1 KI": (T_a, 0.80 * spot / 0.86), "put 30d K=B KI": (T_b, 0.80 * spot / 0.90),
            "put 14d ATM": (T_a, spot)}
    out = {}
    for tag, lv in surfaces.items():
        for name, product in products.items():
            n_contract = len(LocalVolSnowballMCEngine(MCParams(), local_vol_surface=lv)
                             ._build_time_grid(product, env, float(product.maturity))[1])
            substeps = max(1, math.ceil(float(product.maturity) / n_contract / FINE_DT))
            ref = {}
            for pts, spd in ((3200, 64), (6400, 128)):
                ref[f"{pts}x{spd}"] = LocalVolSnowballPDESolver(pde_params(pts, spd), local_vol_surface=lv).price(
                    product, env)
            base = ref["3200x64"]
            row = dict(pde_6400_minus_3200_bp=1e4 * (ref["6400x128"] - base) / product.initial_price,
                       fine_steps=n_contract * substeps)
            for sampling in ("left", "mid"):
                t0 = time.perf_counter()
                pv, se = mc_snowball(lv, product, env, sampling, substeps)
                row[f"mc_{sampling}_bp"] = 1e4 * (pv - base) / product.initial_price
                row[f"mc_{sampling}_se_bp"] = 1e4 * se / product.initial_price
                row[f"mc_{sampling}_wall"] = time.perf_counter() - t0
            if tag == "flat":
                bsm_pde = SnowballPDESolver(pde_params(3200, 64)).price(product, env_bsm)
                pv, se = mc_bsm_snowball(product, env_bsm, None)
                row["bsm_pde_minus_lv_pde_bp"] = 1e4 * (bsm_pde - base) / product.initial_price
                row["bsm_mc_bp"] = 1e4 * (pv - base) / product.initial_price
                row["bsm_mc_se_bp"] = 1e4 * se / product.initial_price
            out[f"{tag} | {name}"] = row
            print(tag, name, {k: round(v, 3) for k, v in row.items()}, flush=True)
            (HERE / "diagnose_gap.json").write_text(json.dumps(out, indent=1))
        for name, (T, k) in puts.items():
            product = EuropeanVanillaOption(strike=float(k), option_type=OptionType.PUT, maturity=float(T))
            base = LocalVolPDESolver(params=pde_params(3200, 64), local_vol_surface=lv).price(product, env)
            fine = LocalVolPDESolver(params=pde_params(6400, 128), local_vol_surface=lv).price(product, env)
            row = dict(pde_6400_minus_3200_bp=1e4 * (fine - base) / spot)
            for sampling in ("left", "mid"):
                pv, se, M = mc_put(lv, spot, float(k), float(T), R, q, sampling)
                row[f"mc_{sampling}_bp"] = 1e4 * (pv - base) / spot
                row[f"mc_{sampling}_se_bp"] = 1e4 * se / spot
                row["fine_steps"] = M
            if tag == "flat":
                row["bs_minus_lv_pde_bp"] = 1e4 * (rep._bs_put(spot, float(k), float(T), R, q, FLAT) - base) / spot
            out[f"{tag} | {name}"] = row
            print(tag, name, {k_: round(v, 3) for k_, v in row.items()}, flush=True)
            (HERE / "diagnose_gap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
