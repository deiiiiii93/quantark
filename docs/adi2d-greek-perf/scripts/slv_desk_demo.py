"""DESK DEMO 3 — do the desk levers hold on SLV? (round-2 tricks re-measured
on the heston_slv cells, which finished certifying after round 1.)

Part 0: C-kernel bitwise A/B on one full SLV march (the kernel claim was
        Heston-measured only).
Part A: desk-A profile (0.80/0.67/0.75 of joint-coarse) on all 7 SLV cells.
        Two error columns per Greek: vs the banked SLV MC reference, and vs
        the certified target-grid PDE value (pure grid-degradation drift).
        SLV near_ki certified INCONCLUSIVE (ref half-widths ±0.63c/±1.41c),
        so its desk verdict rides the PDE-drift column.
Part B: SnowballLadderCache on SLV (ordinary_full, near_ki): build parity,
        spot-move lookups vs fresh re-solves, the v0-remark subtlety, and the
        SLV-specific trigger — the fingerprint must include the LEVERAGE
        surface, not just kappa/theta/sigma/rho.

Engines come from cert16.make_pde_engine (the exact certified construction,
floors zeroed). No engine change; C kernel patched in for A/B and speed.
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
from quantark.volmodels.slv.leverage import LeverageSurface

_STOCK = adi_core.solve_tridiag_batch
CKPT = f"{WORKTREE}/output/adi_greek_certification/checkpoints"
ORDER = ["ordinary_full", "ordinary_decayed", "near_ko", "near_ki",
         "low_feller", "sigma_collapse", "near_expiry"]
DESK = (0.80, 0.67, 0.75)          # desk-A, certified on Heston in round 2


def cell(name):
    return [c for c in cert.certification_cases(quick=False) if c.name == name][0]


def desk_grid(case, name):
    ladders = cert.grid_ladders(case.maturity, quick=False,
                                dense_ki_stencil=(name == "near_ki"))
    nx, nv, nt = (ladders["n_x"][0].n_x, ladders["n_v"][0].n_v,
                  ladders["n_t"][0].n_t)
    return cert.GridPoint(max(100, round(DESK[0] * nx)),
                          max(24, round(DESK[1] * nv)),
                          max(60, round(DESK[2] * nt)))


def slv_engine(case, grid, leverage):
    return cert.make_pde_engine("heston_slv", case, grid, leverage)


def evidence(name):
    ev = json.load(open(f"{CKPT}/heston_slv__{name}.json"))["evidence"]
    q = ev["economic_scale"]["delta_quantum_per_contract"]
    c = ev["certifications"]
    return q, c


# ---------------------------------------------------------------- Part 0
print("=== Part 0: C-kernel bitwise A/B on one full SLV march ===", flush=True)
case = cell("ordinary_decayed")
grid = desk_grid(case, "ordinary_decayed")
lev = cert.make_leverage_surface(case.maturity)
product = cert.make_snowball(case)
env = cert.make_environment(case.spot, 0.20)

adi_core.solve_tridiag_batch = _STOCK
t0 = time.perf_counter()
g_stock = slv_engine(case, grid, lev).calculate_greeks(product, env)
s_stock = time.perf_counter() - t0
adi_core.solve_tridiag_batch = solve_tridiag_batch_c
t0 = time.perf_counter()
g_c = slv_engine(case, grid, lev).calculate_greeks(product, env)
s_c = time.perf_counter() - t0
bitwise = all(g_stock[k] == g_c[k] for k in ("price", "delta", "gamma"))
print(f"  ordinary_decayed desk-A {grid.n_x}x{grid.n_v}x{grid.n_t}: "
      f"stock {s_stock:.2f}s vs C {s_c:.2f}s ({s_stock / s_c:.2f}x)  "
      f"bitwise-identical={bitwise}", flush=True)
# keep the C kernel for the rest of the demo


# ---------------------------------------------------------------- Part A
print("\n=== Part A: desk-A profile on all 7 SLV cells ===", flush=True)
print("  columns: err vs banked MC ref | drift vs certified target-grid PDE",
      flush=True)
total = 0.0
worst_ref, worst_drift = 0.0, 0.0
for name in ORDER:
    case = cell(name)
    q, certs = evidence(name)
    ref_d, ref_g = certs["delta"]["reference"], certs["gamma"]["reference"]
    pde_d, pde_g = certs["delta"]["pde"], certs["gamma"]["pde"]
    hw_d = certs["delta"].get("verdict", {}).get("reference_half_width")
    inconclusive = (
        certs["delta"].get("verdict", {}).get("status") != "PASS"
        or certs["gamma"].get("verdict", {}).get("status") != "PASS"
    )
    grid = desk_grid(case, name)
    lev = cert.make_leverage_surface(case.maturity)
    product = cert.make_snowball(case)
    env = cert.make_environment(case.spot, 0.20)
    t0 = time.perf_counter()
    g = slv_engine(case, grid, lev).calculate_greeks(product, env)
    secs = time.perf_counter() - t0
    total += secs
    edr = (g["delta"] - ref_d) / q
    egr = (g["gamma"] - ref_g) / q
    edp = (g["delta"] - pde_d) / q
    egp = (g["gamma"] - pde_g) / q
    if inconclusive:
        ok = abs(edp) < 1.0 and abs(egp) < 1.0      # verdict on PDE drift
        note = f"  [ref INCONCLUSIVE, hw ±{hw_d:.2f}c -> verdict on drift]"
    else:
        ok = abs(edr) < 1.0 and abs(egr) < 1.0
        note = ""
        worst_ref = max(worst_ref, abs(edr), abs(egr))
    worst_drift = max(worst_drift, abs(edp), abs(egp))
    print(f"  {name:16s} {grid.n_x:3d}x{grid.n_v:2d}x{grid.n_t:<5d} "
          f"ref d={edr:+7.3f}c g={egr:+7.3f}c | drift d={edp:+7.3f}c "
          f"g={egp:+7.3f}c  {secs:6.2f}s  {'PASS' if ok else 'FAIL'}{note}",
          flush=True)
print(f"  total 7-cell SLV desk sweep: {total:.1f}s  "
      f"(worst |ref err| conclusive cells {worst_ref:.3f}c; "
      f"worst |grid drift| {worst_drift:.3f}c)", flush=True)


# ---------------------------------------------------------------- Part B
print("\n=== Part B: ladder cache on SLV ===", flush=True)


def lev_key(lv: LeverageSurface):
    return (lv.time_grid.tobytes(), lv.strike_grid.tobytes(),
            lv.leverage_grid.tobytes())


class SLVLadderCache:
    """Round-2 cache with the SLV fingerprint: (kappa,theta,sigma,rho, L)."""

    def __init__(self, case, grid, leverage, product, env):
        self.case, self.grid, self.product = case, grid, product
        self.rebuild(case.params, leverage, env)

    def rebuild(self, params, leverage, env):
        t0 = time.perf_counter()
        self.params, self.leverage, self.env0 = params, leverage, env
        # prototype: the engine always prices case.params; a Heston-params
        # change is exercised only through the fingerprint check below.
        eng = slv_engine(self.case, self.grid, leverage)
        T = float(self.product.get_maturity(env))
        sig = eng._valuation_state_signature(self.product, env)
        if sig[0] != "live":
            raise RuntimeError(f"cannot cache a '{sig[0]}' position")
        bump_engine = eng.create_bump_context(self.product, env) or eng
        self.core, self.surface = bump_engine._solve_live_surface(
            self.product, env, T, knocked_in=bool(sig[1])
        )
        bc = eng.params.get_effective_bump_config()
        self.eng, self.sig, self.T = eng, sig, T
        self.db = float(bc.spot_bump)
        gb = bc.gamma_spot_bump
        self.gb = float(gb) if gb is not None else self.db
        self.fp = ((params.kappa, params.theta, params.sigma, params.rho),
                   lev_key(leverage), env.valuation_date)
        self.build_secs = time.perf_counter() - t0
        return self.build_secs

    def is_valid(self, *, params=None, leverage=None, env=None):
        if params is not None and (
            (params.kappa, params.theta, params.sigma, params.rho) != self.fp[0]
        ):
            return False, "recalibration: kappa/theta/sigma/rho changed"
        if leverage is not None and lev_key(leverage) != self.fp[1]:
            return False, "recalibration: LEVERAGE SURFACE changed (SLV-only trigger)"
        if env is not None:
            if env.valuation_date != self.fp[2]:
                return False, "date roll"
            sig = self.eng._valuation_state_signature(self.product, env)
            if sig != self.sig:
                return False, f"lifecycle: {self.sig} -> {sig}"
        return True, "valid"

    def lookup(self, spot, v0=None):
        env_s = cert.bumped_environment(self.env0, float(spot))
        ok, reason = self.is_valid(env=env_s)
        if not ok:
            return reason
        spot = float(spot)
        v0r = float(self.params.v0 if v0 is None else v0)
        px = {
            s: float(self.core.interpolate(self.surface,
                                           np.log(spot * (1.0 + s)), v0r))
            for s in (0.0, self.db, -self.db, self.gb, -self.gb)
        }
        dh, gh = spot * self.db, spot * self.gb
        return {
            "price": px[0.0],
            "delta": (px[self.db] - px[-self.db]) / (2.0 * dh),
            "gamma": (px[self.gb] - 2.0 * px[0.0] + px[-self.gb]) / (gh * gh),
        }


for name, moves in (("ordinary_full", (-0.03, -0.01, 0.01, 0.03)),
                    ("near_ki", (-0.01, -0.005, 0.005, 0.01, 0.03))):
    case = cell(name)
    q, _ = evidence(name)
    grid = desk_grid(case, name)
    lev = cert.make_leverage_surface(case.maturity)
    product = cert.make_snowball(case)
    env = cert.make_environment(case.spot, 0.20)
    cache = SLVLadderCache(case, grid, lev, product, env)
    print(f"\n  {name}: build {cache.build_secs:.2f}s "
          f"({grid.n_x}x{grid.n_v}x{grid.n_t})", flush=True)

    look = cache.lookup(case.spot)
    full = slv_engine(case, grid, lev).calculate_greeks(product, env)
    print(f"    parity at build spot: bitwise = "
          f"{all(look[k] == full[k] for k in look)}", flush=True)

    for m in moves:
        s = case.spot * (1.0 + m)
        t0 = time.perf_counter()
        look = cache.lookup(s)
        us = (time.perf_counter() - t0) * 1e6
        t0 = time.perf_counter()
        full = slv_engine(case, grid, lev).calculate_greeks(
            product, cert.bumped_environment(env, s))
        secs = time.perf_counter() - t0
        ed = (look["delta"] - full["delta"]) / q
        eg = (look["gamma"] - full["gamma"]) / q
        ok = abs(ed) < 1.0 and abs(eg) < 1.0
        print(f"    spot {m:+5.1%}  dDiff={ed:+8.4f}c  gDiff={eg:+8.4f}c  "
              f"lookup {us:6.0f}us vs {secs:5.2f}s  {'PASS' if ok else 'FAIL'}",
              flush=True)

    if name == "ordinary_full":
        # v0 remark WITH THE SAME LEVERAGE: mechanically valid lookup...
        nv0 = 0.06
        look = cache.lookup(case.spot, v0=nv0)
        HP = type(case.params)
        p2 = HP(v0=nv0, kappa=case.params.kappa, theta=case.params.theta,
                sigma=case.params.sigma, rho=case.params.rho)
        case2 = type(case)(case.name, p2, case.maturity, case.spot, case.tags)
        full = slv_engine(case2, grid, lev).calculate_greeks(product, env)
        ed = (look["delta"] - full["delta"]) / q
        eg = (look["gamma"] - full["gamma"]) / q
        print(f"    v0 remark 0.04->{nv0:.2f} (SAME leverage): "
              f"dDiff={ed:+7.4f}c gDiff={eg:+7.4f}c — surface v0-independent "
              f"given L", flush=True)
        # ...but a REAL SLV v0 remark recalibrates L -> trigger must fire:
        lev2 = LeverageSurface(lev.time_grid, lev.strike_grid,
                               lev.leverage_grid * 1.02)
        ok, reason = cache.is_valid(leverage=lev2)
        print(f"    leverage recalibration: valid={ok}  ({reason})", flush=True)
        ok, reason = cache.is_valid(params=case.params, leverage=lev)
        print(f"    unchanged params+leverage: valid={ok}", flush=True)
