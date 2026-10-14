"""E06: an independent Heston QE-M MC value for the same contract (evidence only; the PDE engine never imports MC).
Sobol, 64 batches (32,768 paths each: per-batch arrays stay under ~0.5 GB); substeps 2 and 4 per daily interval to show the time bias. It says which PDE limit is right, not
how fast the PDE converges: the PDE arms' n_x 4801 values sit within ~10 of each other for the cell-average arms and
~170 / ~115 away for the two nodal arms.
"""
import time

from common import FELLER, env, product
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine
from quantark.util.enum.engine_enums import HestonMCScheme

if __name__ == "__main__":
    prod, en = product(), env()
    for sub in (2, 4):
        t0 = time.time()
        r = HestonDCNMCEngine(model_params=FELLER, scheme=HestonMCScheme.QUADEXP_M, substeps_per_interval=sub,
                              num_paths=2 ** 21, seed=42, use_sobol=True, num_batches=64).price_detailed(prod, en)
        print(f"QE-M substeps {sub}, 2^21 paths x 64 batches: pv {r.pv:.4f}  s.e. {r.std_error:.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
