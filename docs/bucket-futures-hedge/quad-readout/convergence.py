"""Is the cubic readout more ACCURATE, or only smoother?

Smoothness alone would make it a matter of taste.  If instead the cubic at a
working grid agrees with the refined limit that the linear readout only
reaches much later, then the linear readout is losing real accuracy and the
change is a fix rather than a preference.

Prices the same state on a ladder of grids under both readouts and reports
the distance to the finest common grid.
"""
from __future__ import annotations
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams
from candidates import cubic_interpolate

TTM = 0.295890
NOTIONAL = 50_000_000.0
_ORIGINAL = QuadratureMath.interpolate
GRIDS = (683, 1001, 1501, 2001, 3001, 4001, 5001)


def prices(interp, spot):
    if interp is not None:
        QuadratureMath.interpolate = interp
    try:
        product = repro.build(TTM)
        out = []
        for n in GRIDS:
            eng = SnowballQuadEngine(
                QuadParams(grid_points=n, min_diffusion_stddev_cells=2.5)
            )
            out.append(eng.price(product, repro.env_at(spot)))
        return np.asarray(out)
    finally:
        QuadratureMath.interpolate = _ORIGINAL


def main():
    for spot, tag in ((5306.986, "12 cells above the barrier"),
                      (5060.0, "0.5 cells above the barrier")):
        lin = prices(None, spot)
        cub = prices(cubic_interpolate, spot)
        ref = cub[-1]
        print(f"\n## spot {spot} ({tag}); reference = cubic at n={GRIDS[-1]}")
        print(f"{'n':>6} {'linear':>16} {'cubic':>16} "
              f"{'linear err (bp)':>16} {'cubic err (bp)':>15}")
        for n, a, b in zip(GRIDS, lin, cub):
            print(f"{n:6d} {a:16.4f} {b:16.4f} "
                  f"{(a - ref) / NOTIONAL * 1e4:16.4f} "
                  f"{(b - ref) / NOTIONAL * 1e4:15.4f}")


if __name__ == "__main__":
    main()
