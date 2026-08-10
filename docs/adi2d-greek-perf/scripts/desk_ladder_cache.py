"""DESK DEMO 2 — ladder-cache prototype: one cached live-surface march per
position, intraday Greeks as surface lookups, three invalidation triggers.

The production `calculate_greeks` already prices its bump stencil by bilinear
interpolation of ONE solved surface (`_solve_live_surface` -> `core.interpolate`).
This cache keeps that (core, surface) pair alive across marks and re-reads it
at any (spot, v0) through the IDENTICAL stencil readout, so:

  * at the build spot a lookup is bitwise-identical to production output;
  * an intraday spot move or v0 remark is a microsecond lookup, no re-solve;
  * a re-solve happens only on the three desk triggers:
      (1) recalibration  — kappa/theta/sigma/rho changed,
      (2) date roll      — the surface is a t=0 snapshot for one date,
      (3) lifecycle      — valuation KI/KO state changed (continuous breach
                           or the overnight `_otc_lifecycle_knocked_in` flag).

No engine change: engine internals are reused read-only, C kernel patched in.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta

import json

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

import numpy as np

import quantark.volmodels.adi_core as adi_core
from boosted_tridiag_c import solve_tridiag_batch_c
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams

adi_core.solve_tridiag_batch = solve_tridiag_batch_c

CKPT = f"{WORKTREE}/output/adi_greek_certification/checkpoints"
QUANTUM = json.load(open(f"{CKPT}/heston__ordinary_full.json"))[
    "evidence"]["economic_scale"]["delta_quantum_per_contract"]

# desk grid profile certified by desk_profile_floors.py (fractions of the
# joint-coarse rung): desk-A passed all 7 cells at the 1.0c desk bound
DESK_FX, DESK_FV, DESK_FT = 0.80, 0.67, 0.75            # desk-A


def desk_grid(case, *, dense_ki: bool):
    ladders = cert.grid_ladders(case.maturity, quick=False,
                                dense_ki_stencil=dense_ki)
    nx, nv, nt = (ladders["n_x"][0].n_x, ladders["n_v"][0].n_v,
                  ladders["n_t"][0].n_t)
    return (max(100, round(DESK_FX * nx)), max(24, round(DESK_FV * nv)),
            max(60, round(DESK_FT * nt)))


def make_engine(model_params, grid):
    return HestonSnowballPDESolver(
        model_params,
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


class Invalid:
    def __init__(self, reason):
        self.reason = reason

    def __repr__(self):
        return f"INVALID({self.reason})"


class SnowballLadderCache:
    """Prototype position cache: build = one live-surface march; mark = lookup."""

    def __init__(self, model_params, grid, product, env):
        self.grid = grid
        self.product = product
        self.build_secs = None
        self.rebuild(model_params, env)

    # -- build ------------------------------------------------------------
    def rebuild(self, model_params, env):
        t0 = time.perf_counter()
        self.model_params = model_params
        self.env0 = env
        eng = make_engine(model_params, self.grid)
        T = float(self.product.get_maturity(env))
        sig = eng._valuation_state_signature(self.product, env)
        if sig[0] != "live":
            raise RuntimeError(f"cannot cache a '{sig[0]}' position")
        bump_engine = eng.create_bump_context(self.product, env)
        if bump_engine is None:
            bump_engine = eng
        core, surface = bump_engine._solve_live_surface(
            self.product, env, T, knocked_in=bool(sig[1])
        )
        bc = eng.params.get_effective_bump_config()
        gb = bc.gamma_spot_bump
        self.eng, self.core, self.surface, self.sig, self.T = eng, core, surface, sig, T
        self.db = float(bc.spot_bump)
        self.gb = float(gb) if gb is not None else self.db
        self.valuation_date = env.valuation_date
        self.build_secs = time.perf_counter() - t0
        return self.build_secs

    # -- invalidation -----------------------------------------------------
    @staticmethod
    def _surface_key(p):
        return (p.kappa, p.theta, p.sigma, p.rho)   # v0 is a readout arg

    def is_valid(self, *, model_params=None, env=None):
        if model_params is not None and (
            self._surface_key(model_params) != self._surface_key(self.model_params)
        ):
            return False, ("recalibration: kappa/theta/sigma/rho changed -> "
                           "surface stale")
        if env is not None:
            if env.valuation_date != self.valuation_date:
                return False, (f"date roll: surface is a t=0 snapshot for "
                               f"{self.valuation_date.date()}")
            sig = self.eng._valuation_state_signature(self.product, env)
            if sig != self.sig:
                return False, (f"lifecycle: valuation state {self.sig} -> {sig}")
        return True, "valid"

    # -- mark -------------------------------------------------------------
    def lookup(self, spot, v0=None):
        env_s = cert.bumped_environment(self.env0, float(spot))
        ok, reason = self.is_valid(env=env_s)
        if not ok:
            return Invalid(reason)
        spot = float(spot)
        lo, hi = float(self.core.S_grid[0]), float(self.core.S_grid[-1])
        b_max = max(self.db, self.gb)
        if not (lo <= spot * (1.0 - b_max) and spot * (1.0 + b_max) <= hi):
            return Invalid(f"stencil outside cached S-grid [{lo:.1f}, {hi:.1f}]")
        v0r = float(self.model_params.v0 if v0 is None else v0)
        if not (self.core.V_grid[0] <= v0r <= self.core.V_grid[-1]):
            return Invalid("v0 outside cached variance grid")
        px = {
            s: float(self.core.interpolate(self.surface, np.log(spot * (1.0 + s)), v0r))
            for s in (0.0, self.db, -self.db, self.gb, -self.gb)
        }
        dh, gh = spot * self.db, spot * self.gb
        return {
            "price": px[0.0],
            "delta": (px[self.db] - px[-self.db]) / (2.0 * dh),
            "gamma": (px[self.gb] - 2.0 * px[0.0] + px[-self.gb]) / (gh * gh),
        }


def truth(model_params, grid, product, env):
    eng = make_engine(model_params, grid)
    t0 = time.perf_counter()
    g = eng.calculate_greeks(product, env)
    return g, time.perf_counter() - t0


def compare(tag, look, full, secs_full, lookup_us):
    ed = (look["delta"] - full["delta"]) / QUANTUM
    eg = (look["gamma"] - full["gamma"]) / QUANTUM
    ep = look["price"] - full["price"]
    ok = abs(ed) < 1.0 and abs(eg) < 1.0
    print(f"  {tag:26s} dDiff={ed:+8.4f}c  gDiff={eg:+8.4f}c  "
          f"pvDiff={ep:+9.5f}  lookup {lookup_us:7.1f}us vs re-solve "
          f"{secs_full:5.2f}s  {'PASS' if ok else 'FAIL'}", flush=True)


def timed_lookup(cache, spot, v0=None, n=200):
    res = cache.lookup(spot, v0)
    t0 = time.perf_counter()
    for _ in range(n):
        cache.lookup(spot, v0)
    us = (time.perf_counter() - t0) / n * 1e6
    return res, us


HP = None  # HestonParams class, bound lazily from the case


def heston(p, **over):
    kw = dict(v0=p.v0, kappa=p.kappa, theta=p.theta, sigma=p.sigma, rho=p.rho)
    kw.update(over)
    return HP(**kw)


# ======================================================================
for cell in ("ordinary_full", "near_ki"):
    case = [c for c in cert.certification_cases(quick=False) if c.name == cell][0]
    HP = type(case.params)
    dense = cell == "near_ki"
    grid = desk_grid(case, dense_ki=dense)
    product = cert.make_snowball(case)
    env = cert.make_environment(case.spot, 0.20)

    print(f"\n=== {cell}: spot={case.spot}, desk grid "
          f"{grid[0]}x{grid[1]}x{grid[2]} ===", flush=True)
    cache = SnowballLadderCache(case.params, grid, product, env)
    print(f"  build (one live-surface march): {cache.build_secs:.2f}s   "
          f"S-grid [{cache.core.S_grid[0]:.1f}, {cache.core.S_grid[-1]:.1f}]  "
          f"V-grid [{cache.core.V_grid[0]:.4f}, {cache.core.V_grid[-1]:.4f}]",
          flush=True)

    # -- parity at the build spot ---------------------------------------
    look, us = timed_lookup(cache, case.spot)
    full, secs = truth(case.params, grid, product, env)
    exact = all(look[k] == full[k] for k in ("price", "delta", "gamma"))
    print(f"  parity at build spot: bitwise-identical to production "
          f"calculate_greeks = {exact}", flush=True)
    if not exact:
        compare("(parity detail)", look, full, secs, us)

    # -- intraday spot ladder ---------------------------------------------
    moves = ((-0.03, -0.01, -0.005, 0.005, 0.01, 0.03, 0.05)
             if cell == "ordinary_full" else
             (-0.01, -0.005, 0.005, 0.01, 0.03))
    for m in moves:
        s = case.spot * (1.0 + m)
        look, us = timed_lookup(cache, s)
        if isinstance(look, Invalid):
            print(f"  move {m:+.1%} -> {look}", flush=True)
            continue
        full, secs = truth(case.params, grid, product,
                           cert.bumped_environment(env, s))
        compare(f"spot {m:+5.1%} ({s:7.3f})", look, full, secs, us)

    # -- v0 remark (surface is v0-independent; readout line moves) --------
    if cell == "ordinary_full":
        for nv0 in (0.03, 0.06):
            look, us = timed_lookup(cache, case.spot, v0=nv0)
            full, secs = truth(heston(case.params, v0=nv0), grid, product, env)
            compare(f"v0 remark 0.04->{nv0:.2f}", look, full, secs, us)

# ======================================================================
print("\n=== invalidation triggers (near_ki position) ===", flush=True)
case = [c for c in cert.certification_cases(quick=False) if c.name == "near_ki"][0]
HP = type(case.params)
grid = desk_grid(case, dense_ki=True)
product = cert.make_snowball(case)
env = cert.make_environment(case.spot, 0.20)
cache = SnowballLadderCache(case.params, grid, product, env)

# (1) recalibration
recal = heston(case.params, sigma=case.params.sigma * 1.1)
ok, reason = cache.is_valid(model_params=recal)
print(f"  [1] recalibration sigma 0.30->0.33: valid={ok}  ({reason})", flush=True)
secs = cache.rebuild(recal, env)
look, us = timed_lookup(cache, case.spot)
full, secs_f = truth(recal, grid, product, env)
print(f"      rebuild {secs:.2f}s; post-rebuild parity: "
      f"{all(look[k] == full[k] for k in look)}", flush=True)

# (2) date roll
env_tom = replace(env, valuation_date=env.valuation_date + timedelta(days=1))
ok, reason = cache.is_valid(env=env_tom)
print(f"  [2] date roll +1d: valid={ok}  ({reason})", flush=True)

# (3a) lifecycle — overnight KI flag on the discrete-KI product
try:
    product._otc_lifecycle_knocked_in = True
    ok, reason = cache.is_valid(env=env)
    print(f"  [3a] lifecycle flag _otc_lifecycle_knocked_in: valid={ok}  "
          f"({reason})", flush=True)
    product._otc_lifecycle_knocked_in = False
except Exception as exc:  # frozen dataclass etc.
    print(f"  [3a] lifecycle flag not settable on this product: {exc}", flush=True)

# (3b) lifecycle — intraday continuous-KI breach variant
product_c = cert.make_snowball(case, dense_ki=False)     # CONTINUOUS KI
cache_c = SnowballLadderCache(case.params, grid, product_c, env)
print(f"  [3b] continuous-KI variant: build {cache_c.build_secs:.2f}s "
      f"(live, KI={cache_c.sig[1]})", flush=True)
look = cache_c.lookup(74.6)
print(f"      lookup at 74.6 (below KI 75.0) -> {look}", flush=True)
env_ki = cert.bumped_environment(env, 74.6)
secs = cache_c.rebuild(case.params, env_ki)
look, us = timed_lookup(cache_c, 74.6)
print(f"      rebuild as knocked-in (single V1 march): {secs:.2f}s; "
      f"lookup now delta={look['delta']:+.5f} ({us:.0f}us)", flush=True)
