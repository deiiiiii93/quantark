"""B01: reproduce W3's 2D Heston certain-V0 control gap and print the grid it ran on.

W3 reported +3.8e-3 standard / +8.7e-3 reverse at n_x=96, n_t=48 and +2.1e-3 / +6.9e-3 at 192 x 192.
"""
import numpy as np

from common import HP, certain_v0, closed_form, env, solver


def run(reverse, annualized, n_x, n_v, n_t):
    e = env()
    p = certain_v0(0.10, annualized, reverse=reverse)
    s = solver(HP, n_x, n_v, n_t)
    price = s.price(p, e)
    cf = closed_form(p, e, 0.10, annualized)
    core = s._make_core(p, e, 2.0)
    lo, up = s._rows_v0_stash[4], s._rows_v0_stash[5]
    return price - cf, core, lo, up


if __name__ == "__main__":
    for n_x, n_v, n_t in ((96, 24, 48), (192, 24, 192), (192, 192, 192)):
        for reverse in (False, True):
            for annualized in (False, True):
                d, core, lo, up = run(reverse, annualized, n_x, n_v, n_t)
                print(f"n=({n_x},{n_v},{n_t}) {'rev' if reverse else 'std'} {'ann ' if annualized else 'flat'} "
                      f"PDE-cf={d:+.3e}  S=[{core.S_grid[0]:.3f},{core.S_grid[-1]:.3f}] V=[{core.V_grid[0]:.4f},"
                      f"{core.V_grid[-1]:.4f}] rows {lo.branch}={lo.value:.5f} | {up.branch}={up.value:.5f}")
