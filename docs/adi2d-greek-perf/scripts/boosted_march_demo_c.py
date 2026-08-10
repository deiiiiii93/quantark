"""Four-arm A/B demo on REAL production marches — engine source untouched.

Arms (kernel injected into adi_core's namespace per arm, restored after):
  stock    pure-NumPy Thomas with per-iteration guards (current engine)
  numpy+   hoisted-guard pure NumPy       (bitwise, zero deps, zero toolchain)
  scipy    LAPACK concat / multi-RHS      (~1e-15, pivoted semantics)
  C        transposed compiled Thomas     (bitwise, needs a C toolchain)

Cases cover both variance regimes plus a short-dated sanity check; near_ko
runs the production one-march Greek readout (price+delta+gamma).
"""
from __future__ import annotations

import importlib.util
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
from boosted_tridiag import solve_tridiag_batch_boosted
from boosted_tridiag_c import solve_tridiag_batch_c
from boosted_tridiag_np import solve_tridiag_batch_hoisted
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams

QUANTUM = 0.018130080
CASES = ["near_ko", "sigma_collapse", "near_expiry"]
ARMS = [
    ("stock", None),
    ("numpy+", solve_tridiag_batch_hoisted),
    ("scipy", solve_tridiag_batch_boosted),
    ("C", solve_tridiag_batch_c),
]
STOCK_FN = adi_core.solve_tridiag_batch


def make_engine(case, grid):
    return HestonSnowballPDESolver(
        case.params,
        n_x=grid.n_x, n_v=grid.n_v, n_t=grid.n_t,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        variance_grid_mode="auto",
        v_drift_scheme="adaptive_upwind",
        params=PDEParams(cache_enabled=False),
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0, greek_min_n_v=0,
        greek_min_steps_per_year=0, barrier_greek_min_n_x=0,
    )


results = {}
for label, fn in ARMS:
    if fn is not None:
        adi_core.solve_tridiag_batch = fn
    try:
        print(f"=== arm: {label} ===", flush=True)
        out = {}
        for name in CASES:
            case = [c for c in cert.certification_cases(quick=False) if c.name == name][0]
            product = cert.make_snowball(case)
            env = cert.make_environment(case.spot, 0.20)
            grid = cert.grid_ladders(case.maturity, quick=False)["target"]
            eng = make_engine(case, grid)
            t0 = time.perf_counter()
            if name == "near_ko":
                g = eng.calculate_greeks(product, env)
                secs = time.perf_counter() - t0
                out[name] = {"price": g["price"], "delta": g["delta"],
                             "gamma": g["gamma"], "secs": secs, "grid": grid}
            else:
                px = eng.price(product, env)
                secs = time.perf_counter() - t0
                out[name] = {"price": px, "secs": secs, "grid": grid}
            print(f"  {name:15s} price={out[name]['price']:+.12f}  ({secs:.1f}s)",
                  flush=True)
        results[label] = out
    finally:
        adi_core.solve_tridiag_batch = STOCK_FN

print("\n" + "=" * 100)
hdr = f"{'case':15s} {'grid':>14s}" + "".join(f" {a:>10s}" for a, _ in ARMS) + \
      f" {'best spd':>9s}"
print(hdr)
print("-" * 100)
for name in CASES:
    g = results["stock"][name]["grid"]
    secs = [results[a][name]["secs"] for a, _ in ARMS]
    print(f"{name:15s} {f'{g.n_x}x{g.n_v}x{g.n_t}':>14s}"
          + "".join(f" {s:9.1f}s" for s in secs)
          + f" {secs[0]/min(secs[1:]):8.2f}x")

print("\nPV agreement vs stock (per case):")
for name in CASES:
    base = results["stock"][name]["price"]
    row = []
    for a, _ in ARMS[1:]:
        p = results[a][name]["price"]
        row.append(f"{a}: {'BITWISE' if p == base else f'{abs(p-base):.2e}'}")
    print(f"  {name:15s} " + "   ".join(row))

print("\nnear_ko production Greek readout:")
base = results["stock"]["near_ko"]
for a, _ in ARMS:
    r = results[a]["near_ko"]
    dtag = "BITWISE" if r["delta"] == base["delta"] else f"{(r['delta']-base['delta'])/QUANTUM:+.1e} contracts"
    gtag = "BITWISE" if r["gamma"] == base["gamma"] else f"{(r['gamma']-base['gamma'])/QUANTUM:+.1e} contracts"
    print(f"  {a:7s} delta={r['delta']:+.12f} ({dtag})   gamma={r['gamma']:+.12f} ({gtag})")
