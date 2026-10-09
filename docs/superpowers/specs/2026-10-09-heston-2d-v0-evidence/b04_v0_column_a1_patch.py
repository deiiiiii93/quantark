"""B04: runtime patch -- the explicit S-operator A1 on the v = 0 column (degenerate_pde boundary).

``HestonSLVADICore._A1`` fills only ``out[1:-1, 1:-1]``: on the v = 0 column it returns 0, while the implicit S-solve
(``_tri_S``) carries the S-drift and the -rU reaction on EVERY variance column, the v = 0 one included. A theta-step
of the S-direction is  (I - theta dt A1) Y1 = Y0 - theta dt A1 U  with  Y0 = U + dt (A0 + A1 + A2) U.  With A1 U := 0
on that column it becomes  (I - theta dt A1) Y1 = U + dt A2 U, i.e. the v = 0 row evolves under theta x A1 instead of
A1 -- a consistent operator only at theta = 1 (the damped steps). The degenerate v = 0 PDE is
    U_tau = (r - q) U_x + kappa theta U_v - r U        (the v-diffusion and the S-diffusion vanish at v = 0)
so the column's A1 is (r - q) U_x - r U  (the L^2 v terms are 0 at v = 0).

The patch adds exactly that column to A1 (the formula of the interior rows, evaluated at v = V_grid[0]). Both arms run
back to back in one process.
"""
import sys

import numpy as np

from common import HP, certain_v0, closed_form, env, solver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.param.engine_params import PDEParams
from quantark.volmodels import adi_core

_ORIG_A1 = adi_core.HestonSLVADICore._A1


def _a1_with_v0_column(self, U, t):
    out = _ORIG_A1(self, U, t)
    if not self._degenerate_v0 or self.N_S < 3 or self.N_V < 3:
        return out
    r_step, q_step = self._rq_at(t)
    L2v = (self._L(t) ** 2) * float(self.V_grid[0])
    if self._uniform:
        U_xx = (U[2:, 0] - 2.0 * U[1:-1, 0] + U[:-2, 0]) / (self.dx * self.dx)
        U_x = (U[2:, 0] - U[:-2, 0]) / (2.0 * self.dx)
    else:
        wm2, w02, wp2 = self._xx
        wm1, w01, wp1 = self._x1
        U_xx = U[:-2, 0] * wm2 + U[1:-1, 0] * w02 + U[2:, 0] * wp2
        U_x = U[:-2, 0] * wm1 + U[1:-1, 0] * w01 + U[2:, 0] * wp1
    out[1:-1, 0] = 0.5 * L2v * U_xx + ((r_step - q_step) - 0.5 * L2v) * U_x - r_step * U[1:-1, 0]
    return out


def gap(reverse=False, n=(96, 24, 48), stepper=None, r=0.03, q=0.0, **kw):
    e = env(r=r, q=q)
    p = certain_v0(0.10, False, reverse=reverse)
    s = solver(HP, *n, params=PDEParams(stepper=stepper) if stepper else None, **kw)
    return s.price(p, e) - closed_form(p, e, 0.10, False, r=r)


def both(label, **kw):
    adi_core.HestonSLVADICore._A1 = _ORIG_A1
    a = gap(**kw)
    adi_core.HestonSLVADICore._A1 = _a1_with_v0_column
    b = gap(**kw)
    adi_core.HestonSLVADICore._A1 = _ORIG_A1
    print(f"{label:<46} as-is {a:+.4e}   patched {b:+.4e}", flush=True)


if __name__ == "__main__":
    nd = CrankNicolson(event_damping_steps=0)
    for rev in (False, True):
        t = "rev " if rev else "std "
        both(t + "base (96,24,48)", reverse=rev)
        both(t + "event damping 0", reverse=rev, stepper=nd)
        both(t + "r=q=0.03, event damping 0", reverse=rev, stepper=nd, q=0.03)
        both(t + "r=0, q=0.03, event damping 0", reverse=rev, stepper=nd, r=0.0, q=0.03)
        for n_v in (48, 96, 192):
            both(t + f"n_v={n_v}, event damping 0", reverse=rev, stepper=nd, n=(96, n_v, 48))
        for n_t in (96, 192, 384):
            both(t + f"n_t={n_t}", reverse=rev, n=(96, 24, n_t))
        both(t + "(192,24,192)", reverse=rev, n=(192, 24, 192))
        both(t + "(192,192,192)", reverse=rev, n=(192, 192, 192))
