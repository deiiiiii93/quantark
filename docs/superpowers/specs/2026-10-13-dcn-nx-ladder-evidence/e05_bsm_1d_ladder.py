"""E05: the same contract on the 1D BSM DCN PDE (DCNPDEEngine, flat sigma 0.184 from FLAT), spatial-points ladder in
both event treatments the 1D route offers: event_projection None (its default, CELL_AVERAGE) and "nodal" (the 0.7.0
masks the 2D Heston DCN still uses). Time resolution is the route's default (steps_per_day from the preset).
Separates a generic (1D, any model) nodal-event effect from a 2D/Heston-specific one.
"""
import time

from common import env, product
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import PDEParams

LADDER = (301, 401, 501, 601, 801, 1001, 1201, 1601, 2401, 3201, 4801)

if __name__ == "__main__":
    prod, en = product(), env()
    for proj in (None, "nodal"):
        print(f"1D BSM DCNPDEEngine, event_projection={proj!r}", flush=True)
        prev = None
        for n in LADDER:
            t0 = time.time()
            params = PDEParams(grid=GridConfig(points=n, max_points=max(n, 2000)), event_projection=proj)
            v = DCNPDEEngine(params=params).price(prod, en)
            d = "" if prev is None else f"d {v - prev:+11.4f}"
            print(f"  points {n:>5}  {v!r:>22}  {d}   ({time.time() - t0:.0f} s)", flush=True)
            prev = v
