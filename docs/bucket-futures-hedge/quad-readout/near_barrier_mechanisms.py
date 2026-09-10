"""If a near-barrier delta lobe is real, is it the event smoothing?

`event_smoothing_cells` defaults to 1, so the discrete KI event is spread
over a kernel one cell wide either side of the barrier. That would bias
delta in a smooth, cell-invariant way over a few cells centred on the
barrier -- which is the shape reported, and exactly what a detrended
sub-cell metric cannot see.

Vary the smoothing and the projection against the PDE and see what moves.
"""
from __future__ import annotations

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams

import near_barrier_vs_pde as P


def main():
    p = P.product()
    pde = SnowballPDESolver(PDEParams(accuracy="high"))
    spots = [P.KI * m for m in (0.985, 1.0084, 1.02)]
    refs = {s: P.delta_of(pde, p, s) for s in spots}

    variants = [
        ("transition, smoothing 1 (default)", dict(readout="transition")),
        ("transition, smoothing 0", dict(readout="transition",
                                         event_smoothing_cells=0)),
        ("transition, NODAL projection", dict(readout="transition",
                                              event_projection="nodal")),
        ("transition, grid 1601", dict(readout="transition", grid_points=1601)),
        ("transition, smoothing 0 + grid 1601",
         dict(readout="transition", event_smoothing_cells=0, grid_points=1601)),
    ]
    header = "  ".join(f"{100 * (s / P.KI - 1):+7.2f}%" for s in spots)
    print(f"# delta gap vs PDE, study hands.  spots: {header}")
    for label, kwargs in variants:
        kwargs.setdefault("grid_points", 401)
        eng = SnowballQuadEngine(QuadParams(**kwargs))
        gaps = [P.delta_of(eng, p, s) - refs[s] for s in spots]
        cells = "  ".join(f"{g:8.3f}" for g in gaps)
        print(f"  {label:36s} {cells}")


if __name__ == "__main__":
    main()
