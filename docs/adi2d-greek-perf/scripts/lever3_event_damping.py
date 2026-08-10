"""LEVER 3 — event-damping economics, pure PDEParams study (no engine change).

The per-event Rannacher restart runs `event_rannacher_steps` damped Douglas
steps at `event_theta`. Since the damped-step count scales with the event
count (not n_t), their O(dt^2) local errors already sum to a second-order
global contribution — so the knob should move the CONSTANT, not the order.
This measures that constant and the wall-clock across configurations, plus
the interaction with the time-Richardson pair on near_ko.

Configs: production (eks=2, theta=1.0) / eks=1 / eks=4 / damping OFF /
event_theta=0.5. Errors in hedge contracts vs banked MC references.
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

CONFIGS = {
    "production (eks=2, th=1.0)": dict(),
    "eks=1": dict(event_rannacher_steps=1),
    "eks=4": dict(event_rannacher_steps=4),
    "damping OFF": dict(rannacher_at_events=False),
    "event_theta=0.5": dict(event_theta=0.5),
}
CELLS = ["near_ko", "sigma_collapse", "near_ki"]


def greeks(case, n_x, n_v, n_t, pde_kwargs):
    eng = HestonSnowballPDESolver(
        case.params,
        n_x=n_x, n_v=n_v, n_t=n_t,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        variance_grid_mode="auto",
        v_drift_scheme="adaptive_upwind",
        params=PDEParams(cache_enabled=False, **pde_kwargs),
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0, greek_min_n_v=0,
        greek_min_steps_per_year=0, barrier_greek_min_n_x=0,
    )
    product = cert.make_snowball(case)
    env = cert.make_environment(case.spot, 0.20)
    t0 = time.perf_counter()
    g = eng.calculate_greeks(product, env)
    return g, time.perf_counter() - t0


refs = {}
for name in CELLS:
    ev = json.load(open(f"{CKPT}/heston__{name}.json"))["evidence"]
    refs[name] = (
        ev["certifications"]["delta"]["reference"],
        ev["certifications"]["gamma"]["reference"],
        ev["economic_scale"]["delta_quantum_per_contract"],
    )

for label, kw in CONFIGS.items():
    print(f"=== {label} ===", flush=True)
    for name in CELLS:
        case = [c for c in cert.certification_cases(quick=False) if c.name == name][0]
        tg = cert.grid_ladders(case.maturity, quick=False,
                               dense_ki_stencil=(name == "near_ki"))["target"]
        ref_d, ref_g, q = refs[name]
        g, secs = greeks(case, tg.n_x, tg.n_v, tg.n_t, kw)
        line = (f"  {name:15s} dErr={(g['delta']-ref_d)/q:+8.4f}c "
                f"gErr={(g['gamma']-ref_g)/q:+8.4f}c  ({secs:5.1f}s)")
        if name == "near_ko":
            g2, secs2 = greeks(case, tg.n_x, tg.n_v, tg.n_t // 2, kw)
            rich = (4.0 * g["delta"] - g2["delta"]) / 3.0
            line += (f"   [pair(n_t/2,n_t): {(rich-ref_d)/q:+8.4f}c, "
                     f"+{secs2:.1f}s]")
        print(line, flush=True)
