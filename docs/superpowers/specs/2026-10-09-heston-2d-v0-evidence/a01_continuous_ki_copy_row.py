"""A01: the 2D continuous-KI copy row. Before / after a runtime patch of the 2D boundary hook, back to back.

Under continuous KI the not-knocked-in (V0) edge that lies at or beyond the KI level copies the knocked-in (V1) row
(`SnowballPDESolver._boundary_rows_v0`, branch `v0_{edge}_copy_ki`), read from `self._grid_v1[row, t_idx]`. The 1D
`SnowballAdapter.boundaries` sets V1's rows of the step first and points `_grid_v1` at them; the 2D Heston/SLV hook
(`_Heston2DSnowballPDEBase._boundary_hook`) never sets `_grid_v1`, so the row is 0.0.

The patch (same as the proposed engine change): the 2D V0 hook computes V1's Dirichlet rows at the same tau
(`_boundary_rows_v1`, as the 1D adapter does first) and exposes them as `_grid_v1` for the row, restoring it after.

Contracts: KO 103 monthly / KI 75 continuous (standard) and KO 97 / KI 125 (reverse), ko_rate = rebate = 15%, T=2.
Reference for the BSM limit: Heston sigma=1e-4, v0=theta=0.04 against the 1D SnowballPDESolver at 20% flat vol
(fine grid) and the closed-form-free refinement of the 2D solver.
"""
import numpy as np

from common import HP, env, snowball, solver
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import _Heston2DSnowballPDEBase
from quantark.asset.equity.param.engine_params import PDEParams
from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.volmodels.heston import HestonParams

BSM_LIMIT = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=-0.5)
_ORIG_HOOK = _Heston2DSnowballPDEBase._boundary_hook
ROWS = []


def _patched_hook(self, core, product, env_, knocked_in):
    if knocked_in or not (self._ki_continuous and product.has_ki_barrier):
        return _ORIG_HOOK(self, core, product, env_, knocked_in)

    def hook(U, tau):
        ki_rows = np.zeros((core.S_grid.size, 1), dtype=float)
        lo, up = self._boundary_rows_v1(0, float(tau), product, env_, core.S_grid)
        ki_rows[0, 0], ki_rows[-1, 0] = lo.value, up.value
        grid = np.zeros((core.S_grid.size, 1), dtype=float)
        saved, self._grid_v1 = self._grid_v1, ki_rows
        try:
            self._set_boundary_conditions_v0(grid, core.X_grid, core.S_grid, 0, float(tau), product, env_)
        finally:
            self._grid_v1 = saved
        U[0, :] = grid[0, 0]
        U[-1, :] = grid[-1, 0]
        return U

    return hook


def _recording(hook_factory):
    def factory(self, core, product, env_, knocked_in):
        inner = hook_factory(self, core, product, env_, knocked_in)

        def hook(U, tau):
            out = inner(U, tau)
            if not knocked_in:
                ROWS.append((float(tau), float(out[0, 0]), float(out[-1, 0]), self._rows_v0_stash[4].branch,
                             self._rows_v0_stash[5].branch))
            return out
        return hook
    return factory


def price2d(product, hp, n, patched):
    _Heston2DSnowballPDEBase._boundary_hook = _recording(_patched_hook if patched else _ORIG_HOOK)
    try:
        ROWS.clear()
        s = solver(hp, *n)
        return s.price(product, env()), list(ROWS)
    finally:
        _Heston2DSnowballPDEBase._boundary_hook = _ORIG_HOOK


if __name__ == "__main__":
    cases = (("std KO103/KI75 cont", dict(ko=103.0, ki=75.0, continuous=True, rebate_rate=0.15)),
             ("rev KO97/KI125 cont", dict(ko=97.0, ki=125.0, continuous=True, rebate_rate=0.15, reverse=True)))
    for name, kw in cases:
        p = snowball(**kw)
        ref1d = SnowballPDESolver(params=PDEParams(grid=GridConfig(points=1600, steps_per_day=8))).price(p, env())
        print(f"{name}: 1D BSM PDE (1600 pts, 8 steps/day) = {ref1d:.6f}", flush=True)
        for label, hp in (("BSM limit", BSM_LIMIT), ("Heston HP", HP)):
            for n in ((96, 24, 48), (192, 48, 192), (384, 48, 384)):
                a, rows_a = price2d(p, hp, n, False)
                b, rows_b = price2d(p, hp, n, True)
                ref = f" 2D-1D: as-is {a - ref1d:+.3e} patched {b - ref1d:+.3e}" if label == "BSM limit" else ""
                print(f"  {label:<9} n={n}: as-is {a:.6f}  patched {b:.6f}  patched-as-is {b - a:+.3e}{ref}",
                      flush=True)
        # the rows the V0 hook wrote at the valuation step (last call) and mid-life
        for tag, rows in (("as-is", rows_a), ("patched", rows_b)):
            last = rows[-1]
            print(f"    {tag:<7} t0 rows: lower={last[1]:.6f} ({last[3]})  upper={last[2]:.6f} ({last[4]})")
