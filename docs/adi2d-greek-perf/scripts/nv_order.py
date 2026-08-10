"""Isolate the ORDER cost of the donor-cell fallback on the variance axis.

Same path_focused grid geometry, same n_x/n_t; only v_drift_scheme differs.
Any difference in convergence RATE is attributable to the stencil alone.
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

REF_PRICE, REF_SE = 1.511243, 0.011054
N_X, N_T = 300, 2400
NV_LADDER = [60, 90, 135, 200, 300]

CASE = [c for c in cert.certification_cases(quick=False) if c.name == "sigma_collapse"][0]
product = cert.make_snowball(CASE)
env = cert.make_environment(CASE.spot, 0.20)

SCHEMES = {
    "path_focused + adaptive_upwind (SHIPPED)": dict(
        variance_grid_mode="path_focused", v_drift_scheme="adaptive_upwind"
    ),
    "path_focused + centered": dict(
        variance_grid_mode="path_focused", v_drift_scheme="centered"
    ),
    "legacy grid + centered": dict(
        variance_grid_mode="legacy", v_drift_scheme="centered", v_grid_power=0.0
    ),
}

print(f"RQMC-QE reference = {REF_PRICE:+.6f} +/- {REF_SE:.6f}")
print(f"n_x={N_X} n_t={N_T} FIXED; n_v ladder {NV_LADDER}")
print("(n_v x1.5 per step: 2nd order -> error ratio 2.25, 1st order -> 1.5)\n")

for label, controls in SCHEMES.items():
    print(f"--- {label} ---", flush=True)
    prev_err = None
    for n_v in NV_LADDER:
        eng = HestonSnowballPDESolver(
            CASE.params,
            n_x=N_X, n_v=n_v, n_t=N_T,
            grid_style="concentrated",
            v0_boundary="degenerate_pde",
            params=PDEParams(cache_enabled=False),
            barrier_greek_steps_per_tick=0,
            greek_min_n_x=0, greek_min_n_v=0,
            greek_min_steps_per_year=0, barrier_greek_min_n_x=0,
            **controls,
        )
        if n_v == NV_LADDER[0]:
            d = eng._make_core(product, env, float(CASE.maturity)).variance_operator_diagnostics()
            print(f"    [{d['fallback_nodes']}/{d['interior_nodes']} rows on donor-cell, "
                  f"monotone={d['monotone']}, maxPe={d['max_local_peclet']:.3g}]")
        t0 = time.time()
        px = eng.price(product, env)
        err = px - REF_PRICE
        ratio = (abs(prev_err) / abs(err)) if prev_err not in (None, 0) and err != 0 else float("nan")
        print(f"    n_v={n_v:4d}  price={px:+.6f}  err={err:+.6f} ({err/REF_SE:+6.1f} SE)"
              f"   ratio={ratio:5.2f}   ({time.time()-t0:.0f}s)", flush=True)
        prev_err = err
    print(flush=True)
