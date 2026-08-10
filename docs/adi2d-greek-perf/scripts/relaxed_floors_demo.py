"""Config-lever demo: JOINT-coarse grids vs the banked MC references.

The certification ladders varied one axis at a time; this runs the joint
relaxation (n_x, n_v, n_t all at their coarse rows simultaneously) for every
Heston cell, through the production one-march Greek readout, with the C
kernel patched in. References come from the finished checkpoints — no MC is
re-run. Errors are reported in hedge contracts against the certification
bounds (cell 0.5, mean signed bias budget 0.1).
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

CKPT = f"{WORKTREE}/output/adi_greek_certification/checkpoints"
ORDER = ["ordinary_full", "ordinary_decayed", "near_ko", "near_ki",
         "low_feller", "sigma_collapse", "near_expiry"]
LADDER_KW = inspect.signature(cert.grid_ladders).parameters

adi_core.solve_tridiag_batch = solve_tridiag_batch_c   # C kernel, bitwise-safe

print(f"grid_ladders params: {list(LADDER_KW)}\n")
rows = []
total_secs = 0.0
for name in ORDER:
    doc = json.load(open(f"{CKPT}/heston__{name}.json"))
    ev = doc["evidence"]
    quantum = ev["economic_scale"]["delta_quantum_per_contract"]
    ref_d = ev["certifications"]["delta"]["reference"]
    ref_g = ev["certifications"]["gamma"]["reference"]
    tgt_d = ev["certifications"]["delta"]["pde"]
    tgt_g = ev["certifications"]["gamma"]["pde"]

    case = [c for c in cert.certification_cases(quick=False) if c.name == name][0]
    kwargs = {}
    if "dense_ki_stencil" in LADDER_KW:
        kwargs["dense_ki_stencil"] = (name == "near_ki")
    ladders = cert.grid_ladders(case.maturity, quick=False, **kwargs)
    coarse = ladders["target"].__class__(
        ladders["n_x"][0].n_x, ladders["n_v"][0].n_v, ladders["n_t"][0].n_t
    )

    product = cert.make_snowball(case)
    env = cert.make_environment(case.spot, 0.20)
    eng = HestonSnowballPDESolver(
        case.params,
        n_x=coarse.n_x, n_v=coarse.n_v, n_t=coarse.n_t,
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
    secs = time.perf_counter() - t0
    total_secs += secs

    ed = (g["delta"] - ref_d) / quantum
    eg = (g["gamma"] - ref_g) / quantum
    ed_t = (tgt_d - ref_d) / quantum
    eg_t = (tgt_g - ref_g) / quantum
    ok = abs(ed) < 0.5 and abs(eg) < 0.5
    rows.append((name, coarse, ed, eg, ed_t, eg_t, secs, ok))
    print(f"{name:16s} {coarse.n_x}x{coarse.n_v}x{coarse.n_t:<5d} "
          f"dErr={ed:+7.4f}c (target was {ed_t:+7.4f}c)  "
          f"gErr={eg:+7.4f}c (target {eg_t:+7.4f}c)  "
          f"{secs:5.1f}s  {'PASS' if ok else 'FAIL'}", flush=True)

mean_bias = sum(r[2] for r in rows) / len(rows)
mean_bias_target = sum(r[4] for r in rows) / len(rows)
worst = max(rows, key=lambda r: abs(r[2]))
print(f"\nmean signed delta bias: {mean_bias:+.4f} contracts "
      f"(budget 0.1; certified target grid was {mean_bias_target:+.4f})")
print(f"worst cell |delta err|: {abs(worst[2]):.4f}c ({worst[0]}; cell bound 0.5)")
print(f"total Greek-mark cost, all 7 cells: {total_secs:.1f} s "
      f"(certified target grids on the stock kernel were 246.6 s)")
