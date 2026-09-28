"""Standalone readout comparison -- runtime patched, no engine edits.

Established: the QUAD price is `np.interp(0.0, grid, values)`, and barrier
alignment pins the lattice to the barrier rather than to spot, so the
readout position sweeps across cells and delta comes out as a staircase.

Two things follow, and this script measures both:

  A. causality -- with alignment disabled the lattice is spot-anchored, an
     odd grid puts spot exactly on the centre node, and the sawtooth must
     disappear entirely (no interpolation happens at all);
  B. candidates -- if alignment stays, a higher-order readout on the SAME
     node values removes the staircase.  Reported against the engine's own
     grid-gradient delta, which is staircase-free by construction.

Nothing here changes library behaviour; every variant is a monkeypatch
applied inside this process.
"""
from __future__ import annotations
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams

TTM = 0.295890
M_REF = 200.0
BUMP = 0.0005
SPOTS = np.linspace(5300.0, 5341.0, 42)

_ORIGINAL_INTERP = QuadratureMath.interpolate
_ORIGINAL_ALIGN = SnowballQuadEngine._select_alignment_log


def cubic_interpolate(self, values, x: float = 0.0) -> float:
    """Four-point centred Lagrange on the uniform log grid."""
    grid = self.grid
    k = int(np.searchsorted(grid, x)) - 1
    k = max(1, min(k, len(grid) - 3))
    xs, ys = grid[k - 1 : k + 3], values[k - 1 : k + 3]
    total = 0.0
    for i in range(4):
        w = 1.0
        for j in range(4):
            if i != j:
                w *= (x - xs[j]) / (xs[i] - xs[j])
        total += w * ys[i]
    return float(total)


def measure(label, *, interp=None, align=None):
    if interp is not None:
        QuadratureMath.interpolate = interp
    if align is not None:
        SnowballQuadEngine._select_alignment_log = align
    try:
        product = repro.build(TTM)
        eng = SnowballQuadEngine(
            QuadParams(grid_points=401, min_diffusion_stddev_cells=2.5)
        )
        errs, prices = [], []
        for s in SPOTS:
            s = float(s)
            row = eng.calculate_spot_greeks_curve(product, repro.env_at(s), [s])[0]
            ref = row["delta"] / M_REF
            up = eng.price(product, repro.env_at(s * (1 + BUMP)))
            dn = eng.price(product, repro.env_at(s * (1 - BUMP)))
            prices.append(eng.price(product, repro.env_at(s)))
            errs.append((up - dn) / (2 * s * BUMP) / M_REF - ref)
        errs = np.asarray(errs)
        print(f"{label:<34} ptp {np.ptp(errs):7.4f}  mean|.| "
              f"{np.abs(errs).mean():7.4f}  max|.| {np.abs(errs).max():7.4f}")
        return np.asarray(prices)
    finally:
        QuadratureMath.interpolate = _ORIGINAL_INTERP
        SnowballQuadEngine._select_alignment_log = _ORIGINAL_ALIGN


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ttms", type=float, nargs="+", default=[0.295890])
    ap.add_argument("--with-no-align", action="store_true")
    args = ap.parse_args()

    global TTM
    print("# sub-cell finite-difference delta error vs the engine's grid "
          "delta, in reference hands")
    print(f"# bump {BUMP:.2%}, {SPOTS.size} spots over ~2 cells")
    for ttm in args.ttms:
        TTM = ttm
        print(f"\n## ttm = {ttm:.4f}")
        base = measure("linear readout (current)")
        if args.with_no_align:
            measure("alignment disabled", align=lambda *a, **k: None)
        cubic = measure("cubic readout, alignment kept", interp=cubic_interpolate)
        rel = np.abs(cubic - base) / np.abs(base)
        print(f"   price move vs current: max {rel.max():.3e} rel "
              f"({rel.max() * 1e4:.3f} bp of price)")


if __name__ == "__main__":
    main()
