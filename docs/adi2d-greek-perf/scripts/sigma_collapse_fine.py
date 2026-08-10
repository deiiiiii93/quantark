"""Is the shipped scheme's sigma-collapse PV gap a convergence lag or a bias?

Prices the same synthetic sigma-collapse cell on the certification ladder's
coarse / target / fine rows for the shipped and legacy schemes, so the
trajectory toward the RQMC-QE reference is visible rather than inferred.
"""
from __future__ import annotations

import importlib.util
import sys
import time

spec = importlib.util.spec_from_file_location(
    "cert16",
    "/private/tmp/quant-ark-adi-greek-certification/example/mo_volmodels/16_adi_greek_certification.py",
)
cert = importlib.util.module_from_spec(spec)
sys.modules["cert16"] = cert
spec.loader.exec_module(cert)

from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams

# from sigma_collapse_reference.py, 4096 x 64 scrambles
REF_PRICE = 1.511243
REF_SE = 0.011054

CASE = [c for c in cert.certification_cases(quick=False) if c.name == "sigma_collapse"][0]
product = cert.make_snowball(CASE)
env = cert.make_environment(CASE.spot, 0.20)
ladders = cert.grid_ladders(CASE.maturity, quick=False)

# full 3-point ladder: coarse -> target -> fine on every axis at once
COARSE = ladders["n_x"][0].__class__(200, 60, 1200)
TARGET = ladders["target"]
FINE = ladders["n_x"][0].__class__(450, 135, 4800)

SCHEMES = {
    "shipped (path_focused + adaptive_upwind)": dict(
        variance_grid_mode="auto", v_drift_scheme="adaptive_upwind"
    ),
    "legacy grid + centered": dict(
        variance_grid_mode="legacy", v_drift_scheme="centered", v_grid_power=0.0
    ),
}

print(f"RQMC-QE reference = {REF_PRICE:+.6f} +/- {REF_SE:.6f}\n")
for label, controls in SCHEMES.items():
    print(f"--- {label} ---", flush=True)
    prices = []
    for g in (COARSE, TARGET, FINE):
        eng = HestonSnowballPDESolver(
            CASE.params,
            n_x=g.n_x, n_v=g.n_v, n_t=g.n_t,
            grid_style="concentrated",
            v0_boundary="degenerate_pde",
            params=PDEParams(cache_enabled=False),
            barrier_greek_steps_per_tick=0,
            greek_min_n_x=0, greek_min_n_v=0,
            greek_min_steps_per_year=0, barrier_greek_min_n_x=0,
            **controls,
        )
        t0 = time.time()
        px = eng.price(product, env)
        prices.append(px)
        err = px - REF_PRICE
        print(
            f"    n_x={g.n_x:3d} n_v={g.n_v:3d} n_t={g.n_t:4d}  price={px:+.6f}  "
            f"err={err:+.6f} ({err/REF_SE:+6.1f} SE, {100*err/abs(REF_PRICE):+6.2f}% of PV, "
            f"{err/100*100:+.3f}% of notional)   ({time.time()-t0:.0f}s)",
            flush=True,
        )
    moves = [prices[1] - prices[0], prices[2] - prices[1]]
    print(f"    successive moves: {moves[0]:+.6f} -> {moves[1]:+.6f}"
          f"   {'CONTRACTING' if abs(moves[1]) < abs(moves[0]) else 'NOT CONTRACTING'}\n",
          flush=True)
