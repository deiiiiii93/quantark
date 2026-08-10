"""LEVER 1 — time-axis Richardson, driver-level (no engine change at all).

Per cell: production one-march Greek readout at n_t/4, n_t/2, n_t (target
spatial grid). Combine pairs with the p=2 recipe U* = (4 U_fine - U_coarse)/3:

  cheap pair  (n_t/4, n_t/2): cost ~0.75x a target solve
  quality pair(n_t/2, n_t  ): cost ~1.5x a target solve

Errors in hedge contracts vs the banked MC references. C kernel patched in
(bitwise-safe). Extrapolation commutes with the delta/gamma readout because
both are linear in the surface.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time

SCRATCH = "/private/tmp/claude-501/-Users-fuxinyao-quant-ark/0b6e4fbe-10d5-4787-b996-5c7815bd68e3/scratchpad"
WORKTREE = "/private/tmp/quant-ark-adi-greek-certification"
sys.path.insert(0, SCRATCH)
sys.path.insert(0, WORKTREE)

spec = importlib.util.spec_from_file_location(
    "cert16", f"{WORKTREE}/example/mo_volmodels/16_adi_greek_certification.py"
)
cert = importlib.util.module_from_spec(spec)
sys.modules["cert16"] = cert
spec.loader.exec_module(cert)

import quantark.volmodels.adi_core as adi_core
from boosted_tridiag_c import solve_tridiag_batch_c
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams

adi_core.solve_tridiag_batch = solve_tridiag_batch_c

CKPT = f"{WORKTREE}/output/adi_greek_certification/checkpoints"
ORDER = ["ordinary_full", "ordinary_decayed", "near_ko", "near_ki",
         "low_feller", "sigma_collapse", "near_expiry"]

def solve_row(case, n_x, n_v, n_t):
    product = cert.make_snowball(case)
    env = cert.make_environment(case.spot, 0.20)
    eng = HestonSnowballPDESolver(
        case.params,
        n_x=n_x, n_v=n_v, n_t=n_t,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        variance_grid_mode="auto",
        v_drift_scheme="adaptive_upwind",
        params=PDEParams(cache_enabled=False),
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0, greek_min_n_v=0,
        greek_min_steps_per_year=0, barrier_greek_min_n_x=0,
    )
    t0 = time.perf_counter()
    g = eng.calculate_greeks(product, env)
    return g, time.perf_counter() - t0

def rich(f, c):
    return (4.0 * f - c) / 3.0

summary = []
for name in ORDER:
    doc = json.load(open(f"{CKPT}/heston__{name}.json"))
    ev = doc["evidence"]
    q = ev["economic_scale"]["delta_quantum_per_contract"]
    ref_d = ev["certifications"]["delta"]["reference"]
    ref_g = ev["certifications"]["gamma"]["reference"]
    tgt_err = ev["certifications"]["delta"]["difference_economic_contracts"]

    case = [c for c in cert.certification_cases(quick=False) if c.name == name][0]
    lad = cert.grid_ladders(case.maturity, quick=False,
                            dense_ki_stencil=(name == "near_ki"))
    tg = lad["target"]
    rows = {}
    for frac, n_t in (("t/4", tg.n_t // 4), ("t/2", tg.n_t // 2), ("t", tg.n_t)):
        g, secs = solve_row(case, tg.n_x, tg.n_v, n_t)
        rows[frac] = {"d": g["delta"], "g": g["gamma"], "s": secs, "n_t": n_t}
        print(f"  {name:16s} n_t={n_t:5d}  dErr={(g['delta']-ref_d)/q:+8.4f}c"
              f"  ({secs:.1f}s)", flush=True)

    cheap_d = rich(rows["t/2"]["d"], rows["t/4"]["d"])
    qual_d = rich(rows["t"]["d"], rows["t/2"]["d"])
    cheap_g = rich(rows["t/2"]["g"], rows["t/4"]["g"])
    qual_g = rich(rows["t"]["g"], rows["t/2"]["g"])
    summary.append({
        "name": name,
        "e_t": (rows["t"]["d"] - ref_d) / q,
        "e_cheap": (cheap_d - ref_d) / q,
        "e_qual": (qual_d - ref_d) / q,
        "eg_t": (rows["t"]["g"] - ref_g) / q,
        "eg_cheap": (cheap_g - ref_g) / q,
        "eg_qual": (qual_g - ref_g) / q,
        "cost_t": rows["t"]["s"],
        "cost_cheap": rows["t/4"]["s"] + rows["t/2"]["s"],
        "cost_qual": rows["t/2"]["s"] + rows["t"]["s"],
        "cert_err": tgt_err,
    })

print("\n" + "=" * 108)
print(f"{'cell':16s} {'delta err @ n_t (cert)':>24s} {'cheap pair':>12s} {'quality pair':>13s}"
      f" {'cost n_t':>9s} {'cheap':>7s} {'qual':>7s}")
print("-" * 108)
for r in summary:
    print(f"{r['name']:16s} {r['e_t']:+11.4f}c ({r['cert_err']:+.4f}) "
          f"{r['e_cheap']:+11.4f}c {r['e_qual']:+12.4f}c "
          f"{r['cost_t']:8.1f}s {r['cost_cheap']:6.1f}s {r['cost_qual']:6.1f}s")
print("\ngamma errors (target / cheap pair / quality pair):")
for r in summary:
    print(f"  {r['name']:16s} {r['eg_t']:+8.4f}c  {r['eg_cheap']:+8.4f}c  {r['eg_qual']:+8.4f}c")
