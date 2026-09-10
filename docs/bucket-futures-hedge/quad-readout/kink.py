"""Where does the kink actually live?

A kink at a barrier is created at an observation date and then SMOOTHED by
the backward transition over the next positive-variance interval.  If the
t=0 surface is smooth at the barrier node, then the near-barrier degradation
is large curvature, not a kink -- and the objection to a high-order readout
disappears entirely.

Inspect the retained t=0 surface directly.
"""
from __future__ import annotations
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams

TTM = 0.295890


def main():
    product = repro.build(TTM)
    eng = SnowballQuadEngine(
        QuadParams(grid_points=401, min_diffusion_stddev_cells=2.5)
    )
    eng.price(product, repro.env_at(repro.SPOT0))
    spots, values = eng._last_spot_greeks_grid
    j = int(np.argmin(np.abs(spots - repro.KI)))
    print(f"# KI barrier {repro.KI:.2f}; nearest node {spots[j]:.4f} "
          f"(offset {spots[j] - repro.KI:+.2e})")

    x = np.log(spots)
    d1 = np.diff(values) / np.diff(x)
    d2 = np.diff(d1) / np.diff(x[:-1])
    d3 = np.diff(d2) / np.diff(x[:-2])
    print(f"\n{'node':>6} {'spot':>10} {'dV/dx':>16} {'d2V/dx2':>16} {'d3V/dx3':>16}")
    for i in range(j - 6, j + 7):
        mark = "  <-- barrier node" if i == j else ""
        print(f"{i:6d} {spots[i]:10.3f} {d1[i]:16.2f} {d2[i]:16.2f} "
              f"{d3[i]:16.2f}{mark}")

    band = slice(j - 30, j + 30)
    far = slice(j + 120, j + 180)
    print(f"\n# |d3V/dx3| near the barrier node: "
          f"{np.abs(d3[band]).mean():.3e}")
    print(f"# |d3V/dx3| well above it:          {np.abs(d3[far]).mean():.3e}")
    print("# a surviving kink would show a spike in d2 at the barrier node;")
    print("# a smoothed one shows only elevated, continuous curvature.")


if __name__ == "__main__":
    main()
