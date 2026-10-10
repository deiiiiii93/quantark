"""D04 (user ruling 2026-10-10, "(a) first"): PDE refinement ladder for the Feller-violating 2D Heston DCN.

DCN_A (test/dcn_fixtures.py), Heston v0 0.0249, kappa 9.92, theta 0.045, sigma 1.583, rho -0.025 (Feller ratio 0.36),
HestonDCNPDESolver on its defaults: n_x 301, n_v 101, substeps_per_interval 1 (time = daily nodes x substeps). From
the default grid, double n_x, then n_v, then the time steps, each alone, then all together; 4 levels each (n_x and
n_v doubled as 2n - 1). Values per 1,000,000 notional. Two arms back to back in one process: the patched core (W7 Q1,
HestonSLVADICore._A1 carries the degenerate v = 0 column) and the unpatched one (runtime: that column zeroed again,
exactly the pre-Q1 arithmetic; checked against main's c10 value at (151, 41) bit for bit). Successive differences d_k
= V_k - V_{k-1} and the observed order p = log2(|d_{k-1}| / |d_k|). No Monte Carlo (ruling).

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python d04_dcn_refinement_ladder.py
"""
import math
import time

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver
from quantark.volmodels import adi_core
from quantark.volmodels.heston import HestonParams

FELLER = HestonParams(v0=0.0249, kappa=9.92, theta=0.045, sigma=1.583, rho=-0.025)
MAIN_151_41 = -89176.7328541755          # main 0d1e4133, outputs/c10_main.txt
PATCHED_A1 = adi_core.HestonSLVADICore._A1


def _unpatched_a1(self, U, t):
    out = PATCHED_A1(self, U, t)
    if self._degenerate_v0:
        out[:, 0] = 0.0                    # pre-Q1: A1 filled the interior columns only
    return out


def price(n_x, n_v, sub):
    return HestonDCNPDESolver(FELLER, n_x=n_x, n_v=n_v, substeps_per_interval=sub).price(make_dcn(DCN_A),
                                                                                       flat_env(**FLAT))


def ladder(levels):
    return [(lv, price(*lv)) for lv in levels]


def show(name, rows):
    print(f"  {name}", flush=True)
    prev = prev_d = None
    for (n_x, n_v, sub), v in rows:
        d = None if prev is None else v - prev
        p = (math.log2(abs(prev_d) / abs(d)) if prev_d not in (None, 0.0) and d not in (None, 0.0) else None)
        print(f"    (n_x {n_x:>4}, n_v {n_v:>3}, substeps {sub})  {v:14.4f}  "
              f"{'' if d is None else f'd {d:+11.4f}'}  {'' if p is None else f'p {p:5.2f}'}", flush=True)
        prev_d, prev = d, v


LADDERS = {
    "n_x": [(301, 101, 1), (601, 101, 1), (1201, 101, 1), (2401, 101, 1)],
    "n_v": [(301, 101, 1), (301, 201, 1), (301, 401, 1), (301, 801, 1)],
    "time": [(301, 101, 1), (301, 101, 2), (301, 101, 4), (301, 101, 8)],
    "together": [(301, 101, 1), (601, 201, 2), (1201, 401, 4), (2401, 801, 8)],
}

if __name__ == "__main__":
    adi_core.HestonSLVADICore._A1 = _unpatched_a1
    try:
        check = price(151, 41, 1)
    finally:
        adi_core.HestonSLVADICore._A1 = PATCHED_A1
    print(f"unpatched arm at (151, 41): {check!r} vs main {MAIN_151_41!r}: "
          f"{'bit-identical' if float(check).hex() == float(MAIN_151_41).hex() else 'DIFFERS'}", flush=True)
    print(f"patched (151, 41): {price(151, 41, 1)!r}", flush=True)
    for name, levels in LADDERS.items():
        for arm in ("patched", "unpatched"):
            t0 = time.time()
            adi_core.HestonSLVADICore._A1 = PATCHED_A1 if arm == "patched" else _unpatched_a1
            try:
                rows = ladder(levels)
            finally:
                adi_core.HestonSLVADICore._A1 = PATCHED_A1
            show(f"{name} ladder, {arm} core ({time.time() - t0:.0f} s)", rows)
