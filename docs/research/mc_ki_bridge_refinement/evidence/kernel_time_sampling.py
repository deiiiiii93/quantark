"""Standalone: LV MC European put bias vs total time steps on short expiries (AH surface, fixture)."""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "test" / "volmodels"))
import test_andreasen_huge_repricing as rep
from quantark.param import FlatRateCurve
from quantark.volmodels.localvol import calibrate_andreasen_huge
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc

grid, spot, q = rep._fixture()
R = rep.R
lv = calibrate_andreasen_huge(grid, spot, FlatRateCurve(R), lambda t: q).to_local_vol_surface()
mats = np.asarray(grid.maturities, float); ks = np.asarray(grid.strikes, float); iv = np.asarray(grid.iv_grid, float)
STEPS = [1, 2, 4, 8, 16, 32, 64, 128]
PATHS = 1 << 19
out = []
for ei in (0, 1):
    t = float(mats[ei])
    sel = [int(np.argmin(np.abs(ks - spot * m))) for m in (0.85, 0.93, 1.0, 1.03)]
    for ki in sel:
        k, v = float(ks[ki]), float(iv[ei, ki])
        sd = v * math.sqrt(t); d1 = (math.log(spot / k) + (R - q + 0.5 * v * v) * t) / sd
        vega = spot * math.exp(-q * t) * math.sqrt(t) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        for mode in ("left", "integrated"):
            for M in STEPS:
                dt = np.full(M, t / M)
                t0 = time.perf_counter()
                p, se = price_european_lv_mc(spot, k, False, lv, dt, np.full(M, R), np.full(M, q),
                                             math.exp(-R * t), num_paths=PATHS, seed=7,
                                             use_antithetic=True, return_stderr=True, time_sampling=mode)
                wall = time.perf_counter() - t0
                gap = 100 * (rep._implied(p, spot, k, t, R, q) - v)
                row = dict(days=round(t * 365), k=k, mny=round(k / spot, 3), mode=mode, M=M,
                           gap_vp=gap, se_vp=100 * se / vega, wall=wall)
                out.append(row); print(row, flush=True)
Path(__file__).with_name("measure.json").write_text(json.dumps(out, indent=1))
