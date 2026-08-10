"""A/B demo: stock engine vs boosted tridiagonal kernel on REAL production marches.

The engine source is untouched. The boosted kernel is injected at runtime into
quantark.volmodels.adi_core's namespace for arm B only, then restored. Fresh
engine instances per arm; identical constructor arguments; certification
target grids.

Cases cover both variance regimes:
  near_ko        power grid + (mostly) centered stencil, T=1
  sigma_collapse path_focused grid + donor-cell fallback,  T=3  (worst cost)
  near_expiry    short-dated sanity,                        T=0.25
Greeks (one-march delta/gamma readout) A/B'd on near_ko.
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
from boosted_tridiag import CALLS, solve_tridiag_batch_boosted
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams

QUANTUM = 0.018130080  # hedge contracts per unit delta (cert EconomicGreekScale)
CASES = ["near_ko", "sigma_collapse", "near_expiry"]
STOCK = adi_core.solve_tridiag_batch


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


def run_arm(label):
    out = {}
    for name in CASES:
        case = [c for c in cert.certification_cases(quick=False) if c.name == name][0]
        product = cert.make_snowball(case)
        env = cert.make_environment(case.spot, 0.20)
        grid = cert.grid_ladders(case.maturity, quick=False)["target"]
        eng = make_engine(case, grid)
        t0 = time.perf_counter()
        if name == "near_ko":
            greeks = eng.calculate_greeks(product, env)   # one march: price+delta+gamma
            secs = time.perf_counter() - t0
            out[name] = {"price": greeks["price"], "delta": greeks["delta"],
                         "gamma": greeks["gamma"], "secs": secs, "grid": grid}
        else:
            px = eng.price(product, env)
            secs = time.perf_counter() - t0
            out[name] = {"price": px, "secs": secs, "grid": grid}
        g = grid
        print(f"  [{label}] {name:15s} ({g.n_x}x{g.n_v}x{g.n_t})  "
              f"price={out[name]['price']:+.9f}  ({secs:.1f}s)", flush=True)
    return out


print("=== ARM A: stock engine (pure-NumPy Thomas) ===", flush=True)
A = run_arm("stock")

print("\n=== ARM B: boosted kernel patched into adi_core (engine source untouched) ===",
      flush=True)
adi_core.solve_tridiag_batch = solve_tridiag_batch_boosted
try:
    assert adi_core.solve_tridiag_batch is solve_tridiag_batch_boosted
    B = run_arm("boost")
finally:
    adi_core.solve_tridiag_batch = STOCK
assert CALLS["identical"] > 0 and CALLS["concat"] > 0, "boosted kernel was not exercised"

print(f"\nboosted kernel calls: {CALLS['concat']} concatenated (S-sweep), "
      f"{CALLS['identical']} multi-RHS (V-sweep)")

print("\n" + "=" * 96)
print(f"{'case':15s} {'grid':>14s} {'stock s':>8s} {'boost s':>8s} {'speedup':>8s} "
      f"{'|PV diff|':>11s} {'rel':>9s}")
print("-" * 96)
for name in CASES:
    a, b = A[name], B[name]
    g = a["grid"]
    dpv = abs(b["price"] - a["price"])
    rel = dpv / abs(a["price"]) if a["price"] != 0 else float("nan")
    print(f"{name:15s} {f'{g.n_x}x{g.n_v}x{g.n_t}':>14s} {a['secs']:8.1f} {b['secs']:8.1f} "
          f"{a['secs']/b['secs']:7.2f}x {dpv:11.3e} {rel:9.2e}")

a, b = A["near_ko"], B["near_ko"]
print("\nnear_ko one-march Greeks (production readout path):")
print(f"  delta  stock {a['delta']:+.9f}   boost {b['delta']:+.9f}   "
      f"diff {b['delta']-a['delta']:+.3e}  = {(b['delta']-a['delta'])/QUANTUM:+.2e} contracts")
print(f"  gamma  stock {a['gamma']:+.9f}   boost {b['gamma']:+.9f}   "
      f"diff {b['gamma']-a['gamma']:+.3e}  = {(b['gamma']-a['gamma'])/QUANTUM:+.2e} contracts")
