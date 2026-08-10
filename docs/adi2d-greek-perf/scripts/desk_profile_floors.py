"""DESK DEMO 1 — desk-tolerance grid profiles vs the banked MC references.

The certification contract is 0.5c/cell with a 0.1c mean-bias budget; a
hedging desk works to ~1.0c. This sweeps three desk profiles (fractions of
each cell's joint-coarse certification rung) across all 7 Heston cells,
through the production one-march Greek readout with the C kernel patched in.
References come from the finished checkpoints — no MC re-run, no engine edit.

Profiles (factors applied to the cell's coarse rung n_x/n_v/n_t):
  coarse  1.00/1.00/1.00   the WS-B joint-coarse baseline (re-measured)
  desk-A  0.80/0.67/0.75
  desk-B  0.60/0.50/0.50
  desk-C  0.50/0.36/0.33   aggressive probe of the floor
"""
from __future__ import annotations

import importlib.util
import inspect
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
from quantark.util.exceptions import QuantArkException

CKPT = f"{WORKTREE}/output/adi_greek_certification/checkpoints"
ORDER = ["ordinary_full", "ordinary_decayed", "near_ko", "near_ki",
         "low_feller", "sigma_collapse", "near_expiry"]
LADDER_KW = inspect.signature(cert.grid_ladders).parameters
DESK_BOUND = 1.0    # hedge-contract cents per cell, each Greek
CERT_BOUND = 0.5

adi_core.solve_tridiag_batch = solve_tridiag_batch_c   # C kernel, bitwise-safe

PROFILES = [
    ("coarse", 1.00, 1.00, 1.00),
    ("desk-A", 0.80, 0.67, 0.75),
    ("desk-B", 0.60, 0.50, 0.50),
    ("desk-C", 0.50, 0.36, 1.0 / 3.0),
]


def cell_setup(name):
    doc = json.load(open(f"{CKPT}/heston__{name}.json"))
    ev = doc["evidence"]
    quantum = ev["economic_scale"]["delta_quantum_per_contract"]
    refs = (ev["certifications"]["delta"]["reference"],
            ev["certifications"]["gamma"]["reference"])
    case = [c for c in cert.certification_cases(quick=False) if c.name == name][0]
    kwargs = {}
    if "dense_ki_stencil" in LADDER_KW:
        kwargs["dense_ki_stencil"] = (name == "near_ki")
    ladders = cert.grid_ladders(case.maturity, quick=False, **kwargs)
    coarse = (ladders["n_x"][0].n_x, ladders["n_v"][0].n_v, ladders["n_t"][0].n_t)
    return case, quantum, refs, coarse


def run_cell(case, grid, product, env):
    eng = HestonSnowballPDESolver(
        case.params,
        n_x=grid[0], n_v=grid[1], n_t=grid[2],
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


results = {}          # (cell, profile) -> (ed, eg, secs, status)
for name in ORDER:
    case, quantum, (ref_d, ref_g), (nx_c, nv_c, nt_c) = cell_setup(name)
    product = cert.make_snowball(case)
    env = cert.make_environment(case.spot, 0.20)
    print(f"\n{name}  (coarse rung {nx_c}x{nv_c}x{nt_c})", flush=True)
    for pname, fx, fv, ft in PROFILES:
        grid = (max(100, round(fx * nx_c)),
                max(24, round(fv * nv_c)),
                max(60, round(ft * nt_c)))
        try:
            g, secs = run_cell(case, grid, product, env)
        except QuantArkException as exc:
            print(f"  {pname:7s} {grid[0]:3d}x{grid[1]:3d}x{grid[2]:<5d} "
                  f"GUARD/ERROR: {exc}", flush=True)
            results[(name, pname)] = (None, None, None, "GUARD")
            continue
        ed = (g["delta"] - ref_d) / quantum
        eg = (g["gamma"] - ref_g) / quantum
        ok_desk = abs(ed) < DESK_BOUND and abs(eg) < DESK_BOUND
        ok_cert = abs(ed) < CERT_BOUND and abs(eg) < CERT_BOUND
        status = "PASS+cert" if ok_cert else ("PASS" if ok_desk else "FAIL")
        results[(name, pname)] = (ed, eg, secs, status)
        print(f"  {pname:7s} {grid[0]:3d}x{grid[1]:3d}x{grid[2]:<5d} "
              f"dErr={ed:+8.4f}c  gErr={eg:+8.4f}c  {secs:6.2f}s  {status}",
              flush=True)

print("\n=== profile summaries (7-cell book sweep, single core) ===", flush=True)
for pname, *_ in PROFILES:
    rows = [results[(n, pname)] for n in ORDER]
    if any(r[3] == "GUARD" for r in rows):
        print(f"{pname:7s} — grid guard tripped on some cells")
        continue
    tot = sum(r[2] for r in rows)
    worst_d = max(abs(r[0]) for r in rows)
    worst_g = max(abs(r[1]) for r in rows)
    bias = sum(r[0] for r in rows) / len(rows)
    n_desk = sum(r[3] != "FAIL" for r in rows)
    print(f"{pname:7s} total={tot:6.1f}s  worst|dErr|={worst_d:.3f}c  "
          f"worst|gErr|={worst_g:.3f}c  mean delta bias={bias:+.3f}c  "
          f"desk-pass {n_desk}/7")

print("\n=== cheapest desk-passing profile per cell (regime mix) ===", flush=True)
mix_total = 0.0
for name in ORDER:
    for pname, *_ in reversed(PROFILES):        # cheapest first
        ed, eg, secs, status = results[(name, pname)]
        if status in ("PASS", "PASS+cert"):
            mix_total += secs
            print(f"{name:16s} -> {pname}  ({secs:5.2f}s, "
                  f"dErr={ed:+7.4f}c gErr={eg:+7.4f}c)")
            break
    else:
        print(f"{name:16s} -> NONE passes the 1.0c desk bound")
print(f"\nregime-mix book sweep, all 7 cells: {mix_total:.1f}s "
      f"(joint-coarse cert profile was ~55.7s; certified target grids 246.6s)")
