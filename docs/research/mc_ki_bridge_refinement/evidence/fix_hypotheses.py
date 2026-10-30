"""Phase-3 hypothesis tests for the LV continuous-KI bridge bias (runtime patches, no quantark edits).

H1 finer bridge nodes: the continuous-KI node set uses `factor` x bus_days_in_year (bridge intervals
   shrink; the SDE steps on those nodes). Expect convergence to the PDE as factor grows.
H2 barrier-local bridge variance: h2_k = sigma_LV(B, t_k + dt_k/2)^2 * dt_k instead of the path's own
   integrated variance. Expect a much smaller bias already at daily nodes.
Product A1 (2w continuous KI, spot 7% above KI); err in bp of notional vs PDE continuous 3200x64.
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
from quantark.param import FlatRateCurve, GridVolSurface  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol, calibrate_andreasen_huge  # noqa: E402

HERE = Path(__file__).parent
PATHS = 1 << 17
SEEDS = (11, 23, 37, 53, 71, 89, 101, 131)
T = 14 / 365


class _EnvProxy:
    """The environment with a denser business-day clock, read only by the continuous-KI node set."""

    def __init__(self, env, factor):
        self._env, self._factor = env, factor

    def __getattr__(self, name):
        value = getattr(self._env, name)
        return value * self._factor if name == "bus_days_in_year" else value


class Patched(LocalVolSnowballMCEngine):
    def __init__(self, *args, ki_factor=1, barrier_local=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ki_factor, self.barrier_local = int(ki_factor), bool(barrier_local)

    def _build_time_grid(self, product, pricing_env, T):
        env = pricing_env if self.ki_factor == 1 else _EnvProxy(pricing_env, self.ki_factor)
        return super()._build_time_grid(product, env, T)

    def _check_ki_barriers_continuous_with_bridge(self, paths, all_times, ki_barrier, sigma, is_reverse,
                                                  rng_seed, horizon_steps=None):
        self._bridge_ctx = (np.asarray(all_times, dtype=float), float(ki_barrier))
        return super()._check_ki_barriers_continuous_with_bridge(paths, all_times, ki_barrier, sigma, is_reverse,
                                                                 rng_seed, horizon_steps=horizon_steps)

    def _ki_bridge_step_log_variance(self, paths, horizon_steps=None):
        if not self.barrier_local:
            return super()._ki_bridge_step_log_variance(paths, horizon_steps=horizon_steps)
        times, barrier = self._bridge_ctx
        left = np.concatenate([[0.0], times[:-1]])
        dt = times - left
        var = np.array([float(self._prebuilt.local_vol(np.array([barrier]), t + 0.5 * h)[0]) ** 2 * h
                        for t, h in zip(left, dt)])
        return np.broadcast_to(var, (paths.shape[0], var.size))


def run(lv, product, env, **kw):
    pvs, ses = [], []
    for seed in SEEDS:
        eng = Patched(MCParams(num_paths=PATHS, seed=seed, use_antithetic=True), local_vol_surface=lv, **kw)
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
    product = snowball(spot, T, i_over_s=0.86, ki="continuous", ko_times=[T], bday=bday)
    arms = [
        ("base daily", {}),
        ("base daily, substeps 24", dict(substeps_per_interval=24)),
        ("H1 KI nodes x4", dict(ki_factor=4)),
        ("H1 KI nodes x16", dict(ki_factor=16)),
        ("H1 KI nodes x64", dict(ki_factor=64)),
        ("H2 barrier-local, daily", dict(barrier_local=True)),
        ("H2 barrier-local, daily, substeps 24", dict(barrier_local=True, substeps_per_interval=24)),
        ("H1x4 + H2", dict(ki_factor=4, barrier_local=True)),
        ("H1x16 + H2", dict(ki_factor=16, barrier_local=True)),
    ]
    out = {}
    for tag, lv in surfaces.items():
        ref = LocalVolSnowballPDESolver(pde_params(3200, 64), local_vol_surface=lv).price(product, env)
        rows = {}
        for label, kw in arms:
            pv, se = run(lv, product, env, **kw)
            rows[label] = (1e4 * (pv - ref) / product.initial_price, 1e4 * se / product.initial_price)
            print(tag, label, rows[label], flush=True)
        out[tag] = rows
        (HERE / "fix_hypotheses.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
