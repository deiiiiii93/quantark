"""C05 (Q3): the reverse continuous-KI Heston snowball KO 97 / KI 125 in the production regime, and where the proposed
floor (C02, half-cell floor measured at the barrier) must be inert.

1. n_x = PRODUCTION_BARRIER_GREEK_MIN_N_X = 600 at n_v 48 / n_t 192 and n_v 60 / n_t 800: as-is, floor, correction off.
2. The production barrier-Greek path for this contract: ``ADI_2D_PRODUCTION_ENGINE_CONTROLS`` (example/mo_volmodels/
   11_pde_convergence_gate.py) on the medium PV grid (200, 60, ceil(400 T)); ``calculate_greeks`` resolves the Greek
   grid through ``greek_time_grid_policy`` (printed). Under continuous KI the policy returns before its dense-KI
   branch, so ``barrier_greek_min_n_x`` = 600 never applies; the Greek grid is the generic floor (300, 135, 3200).
3. Inertness of the floor, float.hex(): the standard orientation (KO 103 / KI 75), the discrete twin, the BSM limit
   at (384, 48, 192), the 1D flat-BSM snowball (continuous KI, both orientations) and the 2D Heston Phoenix reverse.
"""
import math
import time

from c02_q3_common import BSM_LIMIT, REV_C, REV_D, STD_C, barrier_cell_floor, correction_off, px
from common import HP, T, env, snowball, solver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSnowballPDESolver

PROD = dict(grid_style="concentrated", v0_boundary="degenerate_pde", variance_grid_mode="auto", v_drift_scheme="auto",
            barrier_greek_steps_per_tick=HestonSnowballPDESolver.PRODUCTION_BARRIER_GREEK_STEPS_PER_TICK,
            greek_min_n_x=HestonSnowballPDESolver.PRODUCTION_GREEK_MIN_N_X,
            greek_min_n_v=HestonSnowballPDESolver.PRODUCTION_GREEK_MIN_N_V,
            greek_min_steps_per_year=HestonSnowballPDESolver.PRODUCTION_GREEK_MIN_STEPS_PER_YEAR,
            barrier_greek_min_n_x=HestonSnowballPDESolver.PRODUCTION_BARRIER_GREEK_MIN_N_X)
MEDIUM = (200, 60, int(math.ceil(400 * T)))


def fmt(v):
    return f"{v:.6f}" if math.isfinite(v) and abs(v) < 1e6 else f"DIVERGED {v:.3e}"


def greeks(**kw):
    s = solver(HP, *MEDIUM, **PROD, **kw)
    p = snowball(**REV_C)
    t0 = time.time()
    try:
        g = s.calculate_greeks(p, env())
        out = "  ".join(f"{k} {fmt(float(v))}" for k, v in g.items())
    except Exception as exc:  # noqa: BLE001 - probe
        out = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out, s.greek_time_grid_policy(p, env()), time.time() - t0


def hexeq(f):
    a = f()
    with barrier_cell_floor():
        b = f()
    return f"{a!r} vs {b!r}  {'bit-identical' if float(a).hex() == float(b).hex() else f'MOVED {b - a:+.3e}'}"


if __name__ == "__main__":
    print("1. n_x = 600", flush=True)
    for n in ((600, 48, 192), (600, 60, 800)):
        a = px(n)
        with barrier_cell_floor():
            b = px(n)
        with correction_off():
            c = px(n)
        print(f"   n={n}: as-is {a}  barrier-cell floor {b}  correction off {c}", flush=True)

    print("2. production path", flush=True)
    for label, ctx in (("as-is", None), ("barrier-cell floor", barrier_cell_floor)):
        if ctx is None:
            pv = px(MEDIUM, **PROD)
            out, policy, sec = greeks()
        else:
            with ctx():
                pv = px(MEDIUM, **PROD)
                out, policy, sec = greeks()
        print(f"   {label}: PV on {MEDIUM} {pv}; Greeks {out}  ({sec:.0f} s)", flush=True)
    print(f"   Greek grid policy: resolved ({policy['resolved_n_x']}, {policy['resolved_n_v']}, "
          f"{policy['resolved_n_t']}), barrier_minimum_n_x {policy['barrier_minimum_n_x']}, reason: "
          f"{policy['reason']}", flush=True)

    print("3. inertness of the floor (as-is vs floor)", flush=True)
    n = (384, 48, 192)
    print("   2D std continuous   ", hexeq(lambda: solver(HP, *n).price(snowball(**STD_C), env())), flush=True)
    print("   2D rev discrete     ", hexeq(lambda: solver(HP, *n).price(snowball(**REV_D), env())), flush=True)
    print("   2D rev BSM limit    ", hexeq(lambda: solver(BSM_LIMIT, *n).price(snowball(**REV_C), env())), flush=True)
    print("   2D std (96,24,48)   ", hexeq(lambda: solver(HP).price(snowball(**STD_C), env())), flush=True)
    print("   1D std continuous   ", hexeq(lambda: SnowballPDESolver().price(snowball(**STD_C), env())), flush=True)
    print("   1D rev continuous   ", hexeq(lambda: SnowballPDESolver().price(snowball(**REV_C), env())), flush=True)
    print("   1D rev KI 115       ", hexeq(lambda: SnowballPDESolver().price(
        snowball(**dict(REV_C, ki=115.0)), env())), flush=True)
