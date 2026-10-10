"""Step 1b: raise the paths and the RQMC replications until the dated-vs-numeric gap is many standard errors or
vanishes. Both forms in one process, paired by seed.

PSEUDO: 16 independent seeds (7 + 1000 i: the engine offsets a batch's seed by 1000 * batch_id, so 1000 spacing
never shares a stream), 65536 paths each -> 1,048,576 paths per form and day.
RANDOMIZED_QUASI: 8 replications (base seeds 7 + 1000 i, far more than the 16-batch count apart: consecutive
seeds share scrambles), 16 batches x 8192 paths each (target_std 1e-12 so all 16 batches run).

Days: 14 (the DATED grid holds a spurious 1e-16 step, grid_scan.py), 92 (the NUMERIC grid does), 45 and 200
(controls: identical node counts)."""
import sys
import time

import numpy as np

import common as c
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod

DAYS = (14, 45, 92, 200)


def run(method, seed, paths, form, day):
    if method == "pseudo":
        eng = SnowballMCEngine(params=MCParams(seed=seed, num_paths=paths))
    else:
        eng = SnowballMCEngine(
            params=MCParams(seed=seed, num_paths=paths, rqmc_min_batches=16, rqmc_max_batches=16,
                            rqmc_target_std=1e-12),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.RANDOMIZED_QUASI))
    v = eng.price(c.build("snowball", form, day), c.env(100.0, c.on(day)))
    return float(v), float(eng.get_last_std_error())


def summarize(label, values):
    a = np.asarray(values, float)
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else float("nan")
    return f"{label} mean={m:.5f} se={se:.5f}"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "pseudo"
    if which == "pseudo":
        seeds = [7 + 1000 * i for i in range(16)]
        paths = 65536
    else:
        seeds = [7 + 1000 * i for i in range(8)]
        paths = 8192
    for day in DAYS:
        per = {"dated": [], "numeric": []}
        gaps = []
        t0 = time.perf_counter()
        for s in seeds:
            d, dse = run(which, s, paths, "dated", day)
            n, nse = run(which, s, paths, "numeric", day)
            per["dated"].append(d)
            per["numeric"].append(n)
            gaps.append(d - n)
            print(f"  {which} day={day} seed={s} dated={d:.6f} (se {dse:.5f}) numeric={n:.6f} (se {nse:.5f}) "
                  f"gap={d - n:+.3e}", flush=True)
        g = np.asarray(gaps)
        gse = g.std(ddof=1) / np.sqrt(g.size)
        print(f"{which} day={day}: {summarize('dated', per['dated'])} | {summarize('numeric', per['numeric'])} | "
              f"paired gap mean={g.mean():+.3e} se={gse:.3e} max|gap|={np.abs(g).max():.3e} "
              f"| dated-numeric of the means in pooled se: "
              f"{(np.mean(per['dated']) - np.mean(per['numeric'])) / np.hypot(*[np.std(per[f], ddof=1) / np.sqrt(len(seeds)) for f in per]) if g.any() else 0.0:+.2f} "
              f"bitwise-equal seeds={int(np.sum(g == 0.0))}/{g.size} [{time.perf_counter() - t0:.0f}s]", flush=True)
