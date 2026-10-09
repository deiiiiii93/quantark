"""A04: which side of the defect-A move is right, where the move is visible (KI just inside the 2D grid's lower edge).

2D Heston in the BSM limit (sigma = 1e-4, v0 = theta = 0.04, i.e. 20% vol) before / after the copy-row fix, against
the 1D SnowballPDESolver at 20% flat vol (its copy row reads the knocked-in grid it marches beside, 1600 points,
8 steps per day). KO 103 monthly, continuous KI at 10.5 / 11 / 12 / 15 (the 2D grid's lower edge is 10.20).
Both 2D arms run back to back in one process.
"""
from common import env, snowball, solver
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver as _S
from quantark.asset.equity.param.engine_params import PDEParams
from quantark.volmodels.heston import HestonParams

BSM_LIMIT = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=-0.5)
FIXED = _S._set_boundary_conditions_v0_beside_v1_rows


def _before(self, grid, x_vec, s_vec, t_idx, tau, product, pricing_env):
    self._set_boundary_conditions_v0(grid, x_vec, s_vec, t_idx, tau, product, pricing_env)


if __name__ == "__main__":
    for ki in (10.5, 11.0, 12.0, 15.0):
        p = snowball(ko=103.0, ki=ki, continuous=True, rebate_rate=0.15)
        ref = SnowballPDESolver(params=PDEParams(grid=GridConfig(points=1600, steps_per_day=8))).price(p, env())
        for n in ((96, 24, 48), (192, 48, 192)):
            _S._set_boundary_conditions_v0_beside_v1_rows = _before
            b = solver(BSM_LIMIT, *n).price(p, env())
            _S._set_boundary_conditions_v0_beside_v1_rows = FIXED
            a = solver(BSM_LIMIT, *n).price(p, env())
            print(f"KI={ki:5.1f} n={n}: 1D ref {ref:.8f}  2D before-ref {b - ref:+.3e}  after-ref {a - ref:+.3e}  "
                  f"after-before {a - b:+.3e}", flush=True)
