"""LEVER 4 — spot-axis nodes: sweep n_x and test a KI-band-preserving thinned
mesh, all on the near_ki dense cell. No engine change: the custom mesh rides
in through a demo-local solver subclass overriding _layer_x_nodes.

The auto grid focus already concentrates at the KI barrier; the question is
whether ~400 well-placed nodes match the certified n_x=600. The custom mesh
keeps the n_x=600 layout's density inside a +/-12% log-band around KI (which
also covers spot and the bump stencil here) and thins to every other node
outside, preserving pinned criticals by construction of the band.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time

import numpy as np

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

NAME = "near_ki"
ev = json.load(open(f"{CKPT}/heston__{NAME}.json"))["evidence"]
REF_D = ev["certifications"]["delta"]["reference"]
REF_G = ev["certifications"]["gamma"]["reference"]
Q = ev["economic_scale"]["delta_quantum_per_contract"]

CASE = [c for c in cert.certification_cases(quick=False) if c.name == NAME][0]
TG = cert.grid_ladders(CASE.maturity, quick=False, dense_ki_stencil=True)["target"]
PRODUCT = cert.make_snowball(CASE)
ENV = cert.make_environment(CASE.spot, 0.20)


def make_solver(cls, n_x):
    return cls(
        CASE.params,
        n_x=n_x, n_v=TG.n_v, n_t=TG.n_t,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        variance_grid_mode="auto",
        v_drift_scheme="adaptive_upwind",
        params=PDEParams(cache_enabled=False),
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0, greek_min_n_v=0,
        greek_min_steps_per_year=0, barrier_greek_min_n_x=0,
    )


def run(label, solver):
    t0 = time.perf_counter()
    g = solver.calculate_greeks(PRODUCT, ENV)
    secs = time.perf_counter() - t0
    core = solver._make_core(PRODUCT, ENV, float(CASE.maturity))
    print(f"  {label:34s} nodes={core.X_grid.size:4d}  "
          f"dErr={(g['delta']-REF_D)/Q:+8.4f}c  gErr={(g['gamma']-REF_G)/Q:+8.4f}c"
          f"  ({secs:5.1f}s)", flush=True)


print(f"near_ki: n_v={TG.n_v} n_t={TG.n_t}, auto grid focus (= KI)", flush=True)
if "--sweep" in sys.argv:   # sweep already banked in lever4.log
    for n_x in (300, 400, 450, 600, 750):
        tag = " <- certified dense floor" if n_x == 600 else ""
        run(f"uniform-budget n_x={n_x}{tag}", make_solver(HestonSnowballPDESolver, n_x))


class ThinnedMeshSolver(HestonSnowballPDESolver):
    """n_x=600 layout, thinned to ~65% outside the KI band (demo-only subclass)."""

    BAND = 0.12  # +/- band in log-spot around the KI barrier

    def _layer_x_nodes(self, product, env, T):
        base = np.asarray(super()._layer_x_nodes(product, env, T), dtype=float)
        ki = float(self._primary_ki_barrier(product))
        x_ki = np.log(ki)
        keep = np.zeros(base.size, dtype=bool)
        keep[0] = keep[-1] = True
        keep[np.abs(base - x_ki) <= self.BAND] = True
        outside = np.where(~keep)[0]
        keep[outside[::2]] = True   # every other node outside the band
        nodes = base[keep]
        # the core validates len(x_nodes) == n_x; follow the thinned count
        self.n_x = int(nodes.size)
        return nodes


thin = make_solver(ThinnedMeshSolver, 600)   # 600-budget layout, then thinned
run("KI-band mesh (600-density in band)", thin)
