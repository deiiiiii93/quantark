"""Which interpolant, given a kink sitting on a node?

Compares the readout candidates on the same node values: the current linear
rule, a 3-point centred Lagrange (quadratic, narrower stencil so it crosses
the kink less often), a 4-point centred Lagrange (cubic), and PCHIP, which
is shape-preserving and cannot overshoot by construction.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams

TTM = 0.295890
M_REF = 200.0
BUMP = 0.0005

_ORIGINAL = QuadratureMath.interpolate


def lagrange(order):
    span = order + 1

    def interp(self, values, x: float = 0.0) -> float:
        grid = self.grid
        k = int(np.searchsorted(grid, x)) - 1
        lo = max(0, min(k - (span // 2 - 1), len(grid) - span))
        xs, ys = grid[lo : lo + span], values[lo : lo + span]
        total = 0.0
        for i in range(span):
            w = 1.0
            for j in range(span):
                if i != j:
                    w *= (x - xs[j]) / (xs[i] - xs[j])
            total += w * ys[i]
        return float(total)

    return interp


def pchip(self, values, x: float = 0.0) -> float:
    grid = self.grid
    k = int(np.searchsorted(grid, x)) - 1
    lo = max(0, min(k - 3, len(grid) - 8))
    return float(PchipInterpolator(grid[lo : lo + 8], values[lo : lo + 8])(x))


def run(label, interp, spots):
    if interp is not None:
        QuadratureMath.interpolate = interp
    try:
        product = repro.build(TTM)
        eng = SnowballQuadEngine(
            QuadParams(grid_points=401, min_diffusion_stddev_cells=2.5)
        )
        errs = []
        for s in spots:
            s = float(s)
            row = eng.calculate_spot_greeks_curve(product, repro.env_at(s), [s])[0]
            ref = row["delta"] / M_REF
            up = eng.price(product, repro.env_at(s * (1 + BUMP)))
            dn = eng.price(product, repro.env_at(s * (1 - BUMP)))
            errs.append((up - dn) / (2 * s * BUMP) / M_REF - ref)
        errs = np.asarray(errs)
        print(f"  {label:<32} ptp {np.ptp(errs):8.4f}  mean|.| "
              f"{np.abs(errs).mean():8.4f}")
    finally:
        QuadratureMath.interpolate = _ORIGINAL


def main():
    ki = repro.KI
    for lo, hi, tag in (
        (ki + 2.0, ki + 42.0, "0.1 to 2.1 cells above the KI barrier"),
        (ki + 240.0, ki + 280.0, "12 to 14 cells above (control)"),
    ):
        spots = np.linspace(lo, hi, 42)
        print(f"{tag}")
        run("linear (current)", None, spots)
        run("3-point Lagrange (quadratic)", lagrange(2), spots)
        run("4-point Lagrange (cubic)", lagrange(3), spots)
        run("PCHIP (shape preserving)", pchip, spots)
        print()


if __name__ == "__main__":
    main()
