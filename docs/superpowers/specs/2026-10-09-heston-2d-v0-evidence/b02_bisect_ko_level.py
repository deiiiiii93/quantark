"""B02: does the certain-V0 gap come from the knock-out region on the grid?

Moves the KO level (standard: up, reverse: down) from W3's 500 / 20 to beyond the grid's edge. The closed form is the
same for every row (the contract never knocks out under BSM); under Heston the true KO probability from (S0, v0) is
tiny (see b04 for MC). Also prints V(S0, v) - cf along the variance axis at t=0 (n_x=96, n_v=24, n_t=48).
"""
import numpy as np

from common import HP, certain_v0, closed_form, env, solver


def gap(reverse, ko, n=(96, 24, 48), annualized=False):
    e = env()
    p = certain_v0(0.10, annualized, reverse=reverse, ko=ko)
    s = solver(HP, *n)
    price = s.price(p, e)
    core, surf = s._solve_live_surface(p, e, 2.0, knocked_in=False)
    cf = closed_form(p, e, 0.10, annualized)
    i = int(np.argmin(abs(core.S_grid - 100.0)))
    prof = surf[i, :] - cf
    return price - cf, core, prof, s._rows_v0_stash


if __name__ == "__main__":
    for reverse, kos in ((False, (300.0, 500.0, 700.0, 900.0, 5000.0)), (True, (40.0, 20.0, 15.0, 11.0, 2.0))):
        for ko in kos:
            d, core, prof, st = gap(reverse, ko)
            print(f"{'rev' if reverse else 'std'} KO={ko:7.1f} PDE-cf={d:+.3e}  S=[{core.S_grid[0]:.2f},"
                  f"{core.S_grid[-1]:.2f}] rows {st[4].branch} | {st[5].branch}")
    d, core, prof, _ = gap(False, 500.0)
    print("standard KO 500, V(S~100, v)-cf by v:")
    for v, x in zip(core.V_grid, prof):
        print(f"   v={v:.5f}  {x:+.3e}")
    d, core, prof, _ = gap(True, 20.0)
    print("reverse KO 20, V(S~100, v)-cf by v:")
    for v, x in zip(core.V_grid, prof):
        print(f"   v={v:.5f}  {x:+.3e}")
