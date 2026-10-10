"""D01 (W7b): library Monte Carlo references for the reverse continuous-KI Heston snowballs the W7b fix cures.

Reverse snowball KO 97, continuous KI 125 / 139.80 / 114.45, 15% rebate, 24 monthly KO dates, T = 2, W3's Heston
(v0 = theta = 0.04, kappa 1.5, sigma 0.3, rho -0.5), r = 0.03, q = 0. HestonSnowballMCEngine, QE scheme, its
continuous-KI Brownian bridge on the path's own quadratic variation, daily business-day grid, 4 substeps, 200,000
antithetic paths x 8 batches. No MC enters a PDE path: the references are pinned in
test/test_vol_pde_reverse_continuous_ki.py with this provenance.
"""
import json
import sys

from common import HP, env, snowball
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import HestonSnowballMCEngine
from quantark.asset.equity.param.engine_params import MCParams
from quantark.util.enum.engine_enums import HestonMCScheme

if __name__ == "__main__":
    out = {}
    for ki in (125.0, 139.80, 114.45):
        eng = HestonSnowballMCEngine(HP, params=MCParams(num_paths=200_000, seed=20261010, use_antithetic=True,
                                                          use_business_day_grid=True),
                                     scheme=HestonMCScheme.QUADEXP, num_batches=8, substeps_per_interval=4)
        pv = eng.price(snowball(ko=97.0, ki=ki, continuous=True, rebate_rate=0.15, reverse=True), env())
        out[f"KI {ki:.2f}"] = [float(pv), float(eng.get_last_std_error())]
        print(f"reverse continuous KI {ki:.2f}: MC {pv!r} +- {eng.get_last_std_error():.3e} (s.e.)", flush=True)
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)
