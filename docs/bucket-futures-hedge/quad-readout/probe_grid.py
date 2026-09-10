"""Does --quad-grid actually change the grid the engine uses?

_resolve_grid_points returns max(requested, required) where `required` comes
from min_diffusion_stddev_cells.  If `required` dominates on the dates that
fail, the grid ladder measured nothing on those dates.
"""
import math
import numpy as np
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams

VOL = 0.155
ACT = 365.0

def daily_times(ttm_years):
    """KI observed every trading day: 5 on, 2 off, ACT/365."""
    n_days = int(round(ttm_years * ACT))
    out = []
    for d in range(1, n_days + 1):
        if d % 7 in (6, 0):          # crude weekend mask
            continue
        out.append(d / ACT)
    if not out or abs(out[-1] - ttm_years) > 1e-12:
        out.append(ttm_years)
    return sorted(set(out))

print(f"{'ttm':>6} {'min dt':>9} {'req=401':>8} {'req=801':>8} {'req=1201':>9}")
for ttm in (1.0, 0.9, 0.75, 0.6, 0.5, 0.35, 0.25, 0.15, 0.08, 0.04):
    times = daily_times(ttm)
    dts = np.diff(np.concatenate(([0.0], times)))
    row = []
    for req in (401, 801, 1201):
        eng = SnowballQuadEngine(QuadParams(grid_points=req))
        row.append(eng._resolve_grid_points(ttm, VOL, times))
    print(f"{ttm:6.2f} {dts.min():9.6f} {row[0]:8d} {row[1]:8d} {row[2]:9d}")
