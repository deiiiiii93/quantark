"""How much does the price move between ADJACENT grid sizes?

Option (d) would choose the grid so that both spot and the barrier land on
nodes, which makes the readout exact -- but then the grid size depends on
spot, so the scheme's own discretisation error becomes spot-dependent.  The
induced delta jump is (price change between adjacent grids) / (spot interval
over which the grid stays fixed).  Measure the numerator.
"""
from __future__ import annotations
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams

TTM = 0.295890
M_REF = 200.0


def main():
    product = repro.build(TTM)
    S = repro.SPOT0
    base = None
    rows = []
    for n in range(683, 700):
        eng = SnowballQuadEngine(
            QuadParams(grid_points=n, min_diffusion_stddev_cells=0.0)
        )
        px = eng.price(product, repro.env_at(S))
        rows.append((n, px))
    pxs = np.array([r[1] for r in rows])
    steps = np.abs(np.diff(pxs))
    print(f"# price at spot {S}, grid_points 683..699, adaptive floor off")
    for (n, px), step in zip(rows[:-1], steps):
        print(f"  n={n:4d}  {px:18.6f}   step to n+1 {step:12.6f}")
    print()
    cell_pts = 19.697
    print(f"# adjacent-grid price step: mean {steps.mean():.4f} "
          f"max {steps.max():.4f} currency")
    print(f"# if the grid changed once per cell ({cell_pts:.1f} points), the "
          f"induced delta jump would be")
    print(f"#   mean {steps.mean() / cell_pts / M_REF:.4f} "
          f"max {steps.max() / cell_pts / M_REF:.4f} reference hands")
    print(f"# compare: the CURRENT staircase riser is ~0.25 hands")


if __name__ == "__main__":
    main()
