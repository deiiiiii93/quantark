"""B07: with the v = 0 column patched (b04), what is left of the standard certain-V0 gap, and is it the damping?

Prediction for the damped steps: each fully implicit (theta = 1) step discounts the constant c by 1 / (1 + r dt)
instead of exp(-r dt): + c (r dt)^2 / 2 per step. The terminal Rannacher start is two theta = 1 half steps:
+ c (r dt)^2 / 4 in total. Everything else (Craig-Sneyd at theta = 1/2 on a constant) is O((r dt)^3) per step.
Standard orientation, KO 500 / KI 1, flat 10% rebate, r = 0.03, n_x = 96, n_v = 24.
"""
from b04_v0_column_a1_patch import _a1_with_v0_column, gap
from common import HP, certain_v0, closed_form, env
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.volmodels import adi_core

R, T, C = 0.03, 2.0, closed_form(certain_v0(), env(), 0.10, False)

if __name__ == "__main__":
    adi_core.HestonSLVADICore._A1 = _a1_with_v0_column
    for n_t in (48, 96, 192):
        dt = T / n_t
        rann = C * (R * dt) ** 2 / 4
        for label, st, pred in (
                ("default (46 event-damped steps + Rannacher)", None, 46 * C * (R * dt) ** 2 / 2 + rann),
                ("event damping 0 (Rannacher only)", CrankNicolson(event_damping_steps=0), rann),
                ("event + terminal damping 0", CrankNicolson(event_damping_steps=0, terminal_damping_steps=0), 0.0)):
            g = gap(n=(96, 24, n_t), stepper=st)
            print(f"n_t={n_t:<4} {label:<46} patched PDE-cf {g:+.4e}  predicted {pred:+.4e}  residual {g - pred:+.2e}",
                  flush=True)
