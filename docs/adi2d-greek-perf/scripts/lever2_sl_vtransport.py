"""LEVER 2 — semi-Lagrangian v-transport (demo-local subclass, no engine change).

Replaces the v-drift discretization entirely:
  * the shared v-generator keeps ONLY diffusion (monotone by construction,
    local Peclet identically 0 — no upwind fallback, no centered wiggle);
  * drift is transported along the EXACT CIR characteristics
        v_foot = theta + (v - theta) * exp(-kappa * dt)
    by cubic-Lagrange interpolation with linear-bracket clipping (monotone);
    mean reversion contracts, so feet are always interior — no outflow BC;
  * Strang splitting: advect dt/2, parent ADI step (drift-free A2), advect dt/2;
  * the degenerate v=0 row (pure drift) becomes advection + implicit identity.

Gate: sigma_collapse n_v ladder vs banked MC reference — the shipped
donor-cell scheme is measured first-order (p=1.19); if SL restores
second-order behaviour, n_v=60 should match upwind's n_v>=300.
Injected by patching the class name in the solver module for arm B only.
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
import quantark.asset.equity.engine.pde.snowball_vol_pde_solvers as solver_mod
from boosted_tridiag_c import solve_tridiag_batch_c
from quantark.asset.equity.param import PDEParams

adi_core.solve_tridiag_batch = solve_tridiag_batch_c
CKPT = f"{WORKTREE}/output/adi_greek_certification/checkpoints"
BaseCore = solver_mod.HestonSLVADICore


class SLCore(BaseCore):
    """Semi-Lagrangian v-advection + diffusion-only implicit v-solve."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adv_cache = {}

    # ---- diffusion-only shared generator (drift removed everywhere) ----
    def _build_v_generator_coefficients(self):
        cached = self._v_operator_cache
        if cached is not None:
            return cached
        v_int = self.V_grid[1:-1]
        diffusion = 0.5 * self.sig_eff2 * v_int
        if self._uniform:
            inv_h2 = 1.0 / (self.dV * self.dV)
            sub = diffusion * inv_h2
            sup = diffusion * inv_h2
        else:
            wm2, _w02, wp2 = self._vv
            sub = diffusion * wm2
            sup = diffusion * wp2
        sub = np.maximum(sub, 0.0)
        sup = np.maximum(sup, 0.0)
        diag = -(sub + sup)
        mono = np.ones(sub.shape, dtype=bool)
        self._v_operator_cache = (sub, diag, sup, mono, np.zeros_like(mono))
        return self._v_operator_cache

    # ---- explicit v-operator: no degenerate drift row (advection owns drift)
    def _A2(self, U):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3:
            return out
        sub, diag, sup, _m, _f = self._build_v_generator_coefficients()
        out[1:-1, 1:-1] = (
            U[1:-1, :-2] * sub + U[1:-1, 1:-1] * diag + U[1:-1, 2:] * sup
        )
        return out

    # ---- implicit v-rows: diffusion + identity degenerate row
    def _tri_V(self, dt_step, theta_loc):
        key = (float(dt_step), float(theta_loc))
        cached = self._V_tri_cache.get(key)
        if cached is not None:
            return cached
        N = self.N_V
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
        sub, diag, sup, _m, _f = self._build_v_generator_coefficients()
        a[1:-1] = -theta_loc * dt_step * sub
        b[1:-1] = 1.0 - theta_loc * dt_step * diag
        c[1:-1] = -theta_loc * dt_step * sup
        if self._degenerate_v0:
            b[0] = 1.0                     # pure-advection row: implicit identity
        else:
            b[0] = 1.0; c[0] = -1.0
        a[-1] = -1.0; b[-1] = 1.0
        self._V_tri_cache[key] = (a, b, c)
        return a, b, c

    # ---- exact-characteristic advection ----
    def _adv_weights(self, dt_sub):
        key = round(float(dt_sub), 15)
        got = self._adv_cache.get(key)
        if got is not None:
            return got
        V = self.V_grid
        N = V.size
        decay = np.exp(-self.kappa * dt_sub)
        feet = self.theta + (V - self.theta) * decay
        feet = np.clip(feet, V[0], V[-1])
        j = np.clip(np.searchsorted(V, feet) - 1, 0, N - 2)
        W = np.zeros((N, N))
        for i in range(N):
            k = j[i]
            f = feet[i]
            if 1 <= k <= N - 3:
                xs = V[k - 1:k + 3]
                for a_ in range(4):
                    w = 1.0
                    for b_ in range(4):
                        if a_ != b_:
                            w *= (f - xs[b_]) / (xs[a_] - xs[b_])
                    W[i, k - 1 + a_] = w
            else:  # edge: linear
                t = (f - V[k]) / (V[k + 1] - V[k])
                W[i, k] = 1.0 - t
                W[i, k + 1] = t
        got = (W.T.copy(), j)
        self._adv_cache[key] = got
        return got

    def _advect_v(self, U, dt_sub):
        if dt_sub <= 0.0:
            return U
        WT, j = self._adv_weights(dt_sub)
        U_new = U @ WT
        lo = np.minimum(U[:, j], U[:, j + 1])   # linear-bracket clip -> monotone
        hi = np.maximum(U[:, j], U[:, j + 1])
        return np.clip(U_new, lo, hi)

    # ---- Strang split around the parent steps ----
    def _douglas_step(self, U, dt_step, tau, theta_loc, t_mid):
        U = self._advect_v(U, 0.5 * dt_step)
        U = super()._douglas_step(U, dt_step, tau, theta_loc, t_mid)
        U = self._advect_v(U, 0.5 * dt_step)
        self._bc(U, tau)
        return U

    def _cs_step(self, U, dt_step, tau, theta_loc, t_mid):
        U = self._advect_v(U, 0.5 * dt_step)
        U = super()._cs_step(U, dt_step, tau, theta_loc, t_mid)
        U = self._advect_v(U, 0.5 * dt_step)
        self._bc(U, tau)
        return U


def greeks(name, n_x, n_v, n_t, grid_mode="auto", scheme="adaptive_upwind"):
    case = [c for c in cert.certification_cases(quick=False) if c.name == name][0]
    eng = solver_mod.HestonSnowballPDESolver(
        case.params,
        n_x=n_x, n_v=n_v, n_t=n_t,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        variance_grid_mode=grid_mode,
        v_drift_scheme=scheme,
        params=PDEParams(cache_enabled=False),
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0, greek_min_n_v=0,
        greek_min_steps_per_year=0, barrier_greek_min_n_x=0,
    )
    product = cert.make_snowball(case)
    env = cert.make_environment(case.spot, 0.20)
    t0 = time.perf_counter()
    g = eng.calculate_greeks(product, env)
    return g, time.perf_counter() - t0


ev = json.load(open(f"{CKPT}/heston__sigma_collapse.json"))["evidence"]
REF_D = ev["certifications"]["delta"]["reference"]
Q = ev["economic_scale"]["delta_quantum_per_contract"]
RQMC_PV, RQMC_SE = 1.511243, 0.011054   # session RQMC-QE reference (4096x64)

print("=== sigma_collapse: n_v ladder at n_x=300, n_t=2400 (800/yr) ===")
for arm in ("shipped", "SL"):
    if arm == "SL":
        solver_mod.HestonSLVADICore = SLCore
    try:
        for n_v in (45, 60, 90, 135):
            g, secs = greeks("sigma_collapse", 300, n_v, 2400)
            print(f"  [{arm:7s}] n_v={n_v:3d}  dErr={(g['delta']-REF_D)/Q:+8.4f}c  "
                  f"PVerr={(g['price']-RQMC_PV)/RQMC_SE:+6.1f} SE  ({secs:.1f}s)",
                  flush=True)
    finally:
        solver_mod.HestonSLVADICore = BaseCore

print("\n=== SL n_t stability (n_v=60): the Rannacher-crutch divergence check ===")
solver_mod.HestonSLVADICore = SLCore
try:
    prev = None
    for n_t in (600, 1200, 2400):
        g, secs = greeks("sigma_collapse", 300, 60, n_t)
        move = "" if prev is None else f"  move={(g['price']-prev):+.6f}"
        print(f"  n_t={n_t:5d}  PV={g['price']:+.6f}  "
              f"PVerr={(g['price']-RQMC_PV)/RQMC_SE:+6.1f} SE{move}  ({secs:.1f}s)",
              flush=True)
        prev = g["price"]
finally:
    solver_mod.HestonSLVADICore = BaseCore

print("\n=== ordinary_full sanity (n_x=300, n_v=90, n_t=2400) ===")
ev2 = json.load(open(f"{CKPT}/heston__ordinary_full.json"))["evidence"]
ref2 = ev2["certifications"]["delta"]["reference"]
for arm in ("shipped", "SL"):
    if arm == "SL":
        solver_mod.HestonSLVADICore = SLCore
    try:
        g, secs = greeks("ordinary_full", 300, 90, 2400)
        print(f"  [{arm:7s}] dErr={(g['delta']-ref2)/Q:+8.4f}c  ({secs:.1f}s)", flush=True)
    finally:
        solver_mod.HestonSLVADICore = BaseCore
