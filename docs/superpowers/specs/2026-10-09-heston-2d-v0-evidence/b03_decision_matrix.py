"""B03: decision matrix for the 2D Heston certain-V0 control gap (PDE - closed form, price units per 100 notional).

Each row changes ONE knob from the base (W3's control: standard, flat rebate 10%, r=0.03, q=0, HP, n=(96,24,48),
Craig-Sneyd, the 2D default stepper = CrankNicolson resolved to DAMPING_DEFAULTS["2D"] = (full_step, 1 terminal,
2 event damping steps), 24 KO dates). All rows run back to back in one process on one machine.

Also prints the prediction of the "damped steps are implicit Euler on the -rU reaction" hypothesis:
    n_damped x c x (r dt)^2 / 2      (c = the V0 constant, ~103.59)
which is the local error of 1/(1 + r dt) against exp(-r dt), per damped step.
"""
import math
import sys

import numpy as np

from common import HP, certain_v0, closed_form, env, solver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import event_damped_step_keys
from quantark.asset.equity.param.engine_params import PDEParams
from quantark.volmodels.heston import HestonParams

BSM_LIMIT = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=-0.5)


def run(label, reverse=False, annualized=False, r=0.03, q=0.0, hp=HP, n=(96, 24, 48), stepper=None, n_obs=24, **kw):
    e = env(r=r, q=q)
    p = certain_v0(0.10, annualized, reverse=reverse, n_obs=n_obs)
    params = PDEParams(stepper=stepper) if stepper is not None else None
    s = solver(hp, *n, params=params, **kw)
    price = s.price(p, e)
    cf = closed_form(p, e, 0.10, annualized, r=r)
    # damped-step count of the march actually run
    st = s._resolved_stepper_for_probe if hasattr(s, "_resolved_stepper_for_probe") else None
    from quantark.asset.equity.engine.pde.core.components import resolve_stepper
    st = resolve_stepper(s.params, DAMPING_DEFAULTS["2D"])
    core = s._make_core(p, e, 2.0)
    maps = s._build_event_maps(p, e, 2.0, core.dt)
    keys = event_damped_step_keys(st, maps, core.N_T) or set()
    n_damped = len(keys)
    pred = n_damped * cf * (r * core.dt) ** 2 / 2.0 if r else 0.0
    print(f"{label:<44} PDE-cf={price - cf:+.4e}  damped={n_damped:>3}/{core.N_T:<4} "
          f"implicit-Euler prediction={pred:+.4e}", flush=True)
    return price - cf


if __name__ == "__main__":
    rev = "--reverse" in sys.argv
    tag = "rev " if rev else "std "
    no_event = CrankNicolson(event_damping_steps=0)
    no_damp = CrankNicolson(event_damping_steps=0, terminal_damping_steps=0)
    run(tag + "base", rev)
    run(tag + "r=q=0", rev, r=0.0)
    run(tag + "r=q=0.03 (carry 0, discount on)", rev, r=0.03, q=0.03)
    run(tag + "annualized rebate", rev, annualized=True)
    run(tag + "event damping 0", rev, stepper=no_event)
    run(tag + "event+terminal damping 0", rev, stepper=no_damp)
    run(tag + "event damping 1", rev, stepper=CrankNicolson(event_damping_steps=1))
    run(tag + "event_theta 0.5 (damped steps = CN)", rev, stepper=CrankNicolson(event_theta=0.5))
    run(tag + "n_obs 4 (fewer events)", rev, n_obs=4)
    run(tag + "BSM limit (sigma 1e-4, v0=theta)", rev, hp=BSM_LIMIT)
    run(tag + "BSM limit, event damping 0", rev, hp=BSM_LIMIT, stepper=no_event)
    run(tag + "scheme DOUGLAS", rev, scheme="douglas")
    run(tag + "v_drift centered", rev, v_drift_scheme="centered")
    run(tag + "v_drift semi_lagrangian", rev, v_drift_scheme="semi_lagrangian")
    run(tag + "v0_boundary neumann", rev, v0_boundary="neumann")
    for n_t in (96, 192, 384):
        run(tag + f"n_t={n_t}", rev, n=(96, 24, n_t))
    for n_v in (48, 96, 192):
        run(tag + f"n_v={n_v}", rev, n=(96, n_v, 48))
    for n_x in (192, 384):
        run(tag + f"n_x={n_x}", rev, n=(n_x, 24, 48))
    for n_t in (96, 192, 384):
        run(tag + f"n_t={n_t}, event damping 0", rev, n=(96, 24, n_t), stepper=no_event)
    for n_v in (48, 96, 192):
        run(tag + f"n_v={n_v}, event damping 0", rev, n=(96, n_v, 48), stepper=no_event)
    for n_x in (192, 384):
        run(tag + f"n_x={n_x}, event damping 0", rev, n=(n_x, 24, 48), stepper=no_event)
