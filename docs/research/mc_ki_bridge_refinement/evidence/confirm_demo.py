"""Confirmation demo (no quantark edits): two-part partial refinement for LV snowball MC.

Two-part rule (runtime subclass of LocalVolSnowballMCEngine):
  1. inside [0, W]: SDE steps of at most W/F;
  2. after W: SDE steps of at most max_dt (one business day here).
Every contractual interval is split independently; recorded nodes, payoffs and the
continuous-KI bridge variance stay on the contractual grid. W=0, max_dt=inf is today's default.

Surfaces: Andreasen-Huge and Dupire local vol of the iv_surface_20240208 fixture.
Reference: LocalVolSnowballPDESolver on the same surface (rungs 1600x16, 3200x64).
Noise: 4 independent seeds x 2^18 antithetic paths per arm.
"""
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/fuxinyao/quant-ark/test/volmodels")
import test_andreasen_huge_repricing as rep  # noqa: E402

from quantark.asset.equity.engine.mc import LocalVolSnowballMCEngine  # noqa: E402
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import _ArrayPathGenerator  # noqa: E402
from quantark.asset.equity.engine.pde import LocalVolSnowballPDESolver  # noqa: E402
from quantark.asset.equity.engine.pde.grid.config import GridConfig  # noqa: E402
from quantark.asset.equity.param import MCParams, PDEParams  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import BarrierConfig  # noqa: E402
from quantark.asset.equity.product.option.snowball_option import SnowballOption  # noqa: E402
from quantark.param import FlatRateCurve  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol, calibrate_andreasen_huge  # noqa: E402

HERE = Path(__file__).parent
PATHS = 1 << 18
SEEDS = (2029, 4111, 6337, 8443)
DAY = 1.0 / 365.0


class TwoPartLV(LocalVolSnowballMCEngine):
    def __init__(self, *args, window=0.0, front_steps=1, max_dt=math.inf, **kwargs):
        super().__init__(*args, **kwargs)
        self.window, self.front_steps, self.max_dt = float(window), int(front_steps), float(max_dt)
        self._counts = None

    def _refined_dt_array(self, dt_array):
        dt = np.asarray(dt_array, dtype=float)
        edges = np.concatenate([[0.0], np.cumsum(dt)])
        h_front = self.window / self.front_steps if self.window > 0 else math.inf
        fine, counts = [], []
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            pieces = []
            m = min(b, self.window)
            if m > a:
                k = max(1, math.ceil((m - a) / h_front - 1e-9))
                pieces += [(m - a) / k] * k
            lo = max(a, m)
            rest = b - lo
            if rest > 1e-12 * b:
                k = max(1, math.ceil(rest / self.max_dt - 1e-9)) if math.isfinite(self.max_dt) else 1
                pieces += [rest / k] * k
            if len(pieces) == 1:
                pieces = [float(dt[i])]          # unrefined interval: the contractual dt, bit for bit
            fine += pieces
            counts.append(len(pieces))
        self._counts = np.asarray(counts, dtype=int)
        return np.asarray(fine, dtype=float)

    def _make_path_generator(self, simulate, n_eff, batch_id):
        cols = np.concatenate([[0], np.cumsum(self._counts)])

        def simulate_contractual(batch_id=None, seed=None):
            return simulate(batch_id=batch_id, seed=seed)[:, cols]

        return _ArrayPathGenerator(simulate_contractual, n_eff, batch_id=batch_id)

    def _record_step_log_variance(self, fine):
        if fine is not None:
            fine = np.add.reduceat(fine, np.concatenate([[0], np.cumsum(self._counts)[:-1]]), axis=1)
        self._bridge_variance_store()[threading.get_ident()] = fine


def snowball(spot, T, *, i_over_s, continuous_ki, ko_times):
    initial = spot / i_over_s
    ki = dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True) if continuous_ki else dict(
        ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=[T])
    return SnowballOption(
        initial_price=initial, strike=initial, maturity=T, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=1.03 * initial, ko_rate=0.15,
                                     ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_times), ki_barrier=0.80 * initial, **ki))


def arms_for(case, bday):
    two = lambda w, f, cap: dict(window=w * DAY, front_steps=f, max_dt=cap)  # noqa: E731
    arms = [("base", None, {}), ("noop two-part (=base bitwise)", "two", two(0, 1, math.inf))]
    arms += [(f"two-part W10/F{f}", "two", two(10, f, bday)) for f in (32, 64, 128)]
    arms += [("two-part W5/F64", "two", two(5, 64, bday)), ("two-part W20/F64", "two", two(20, 64, bday))]
    arms += [("cap 1bd only", "two", two(0, 1, bday))]
    if case.startswith(("A1", "B")):
        arms += [(f"uniform x{n}", "uniform", dict(substeps_per_interval=n)) for n in (4, 7, 13, 26)]
    return arms


def run_surface(tag, lv, env, cases, bday, results):
    for name, product in cases.items():
        T = float(product.maturity)
        pde = {}
        for pts, spd in ((1600, 16), (3200, 64)):
            p = PDEParams(accuracy="high", grid=GridConfig(points=pts, steps_per_day=spd,
                                                           max_points=20000, max_steps=400000))
            pde[f"{pts}x{spd}"] = LocalVolSnowballPDESolver(p, local_vol_surface=lv).price(product, env)
        ref = pde["3200x64"]
        notional = product.initial_price
        contractual = LocalVolSnowballMCEngine(MCParams(), local_vol_surface=lv)._build_time_grid(product, env, T)[1]
        rows = []
        for label, kind, kw in arms_for(name, bday):
            pvs, ses, walls = [], [], []
            for seed in SEEDS:
                params = MCParams(num_paths=PATHS, seed=seed, use_antithetic=True)
                eng = (TwoPartLV(params, local_vol_surface=lv, **kw) if kind == "two"
                       else LocalVolSnowballMCEngine(params, local_vol_surface=lv, **kw))
                t0 = time.perf_counter()
                pvs.append(eng.price(product, env))
                walls.append(time.perf_counter() - t0)
                ses.append(eng.get_last_std_error())
            steps = len(eng._refined_dt_array(contractual))
            errs = [1e4 * (v - ref) / notional for v in pvs]
            row = dict(arm=label, steps=steps, err_bp=float(np.mean(errs)),
                       se_bp=1e4 * math.sqrt(sum(s * s for s in ses)) / len(ses) / notional,
                       seed_spread_bp=float(np.std(errs, ddof=1)), per_seed_bp=errs,
                       wall_per_seed=float(np.mean(walls)), pvs=pvs)
            rows.append(row)
            print(tag, name, {k: row[k] for k in ("arm", "steps", "err_bp", "se_bp", "seed_spread_bp",
                                                 "wall_per_seed")}, flush=True)
        results[f"{tag} | {name}"] = dict(T=T, contractual_steps=len(contractual),
                                          pde_rung_gap_bp=1e4 * (pde["1600x16"] - ref) / notional, rows=rows)
        (HERE / "confirm_demo.json").write_text(json.dumps(results, indent=1))


def main():
    grid, spot, q = rep._fixture()
    env = rep._flat_env(spot, q, grid)
    bday = 1.0 / float(env.bus_days_in_year)
    cases = {
        "A1 2w cont-KI, spot 7% above KI": snowball(spot, 14 / 365, i_over_s=0.86, continuous_ki=True,
                                                    ko_times=[14 / 365]),
        "A2 2w cont-KI, spot 2% below KO": snowball(spot, 14 / 365, i_over_s=1.01, continuous_ki=True,
                                                    ko_times=[14 / 365]),
        "B 1m discrete KI+KO at maturity": snowball(spot, 30 / 365, i_over_s=0.90, continuous_ki=False,
                                                    ko_times=[30 / 365]),
        "C 1y cont-KI, monthly KO (fresh)": snowball(spot, 1.0, i_over_s=1.0, continuous_ki=True,
                                                     ko_times=[m / 12 for m in range(1, 13)]),
    }
    surfaces = {
        "AH": calibrate_andreasen_huge(grid, spot, FlatRateCurve(rep.R), lambda t: q).to_local_vol_surface(),
        "Dupire": build_dupire_local_vol(grid, spot=spot, rate_curve=FlatRateCurve(rep.R), div_yield=lambda t: q),
    }
    results = {}
    for tag, lv in surfaces.items():
        run_surface(tag, lv, env, cases, bday, results)


if __name__ == "__main__":
    main()
