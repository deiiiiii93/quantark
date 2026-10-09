"""A07: the PDE byte book's 2D rows (test/pde_bitwise/capture_book.py vol_rows: heston_snowball, slv_snowball,
heston_phoenix -- all continuous KI) before / after the defect-A fix, back to back in one process, as float.hex().
Run with PYTHONPATH=<wt>:<wt>/test.
"""
import numpy as np

from pde_bitwise.capture_book import PHOENIX, env_flat, make_phoenix, snowball
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (HestonSLVSnowballPDESolver,
                                                                         HestonSnowballPDESolver)
from quantark.asset.equity.param import PDEParams
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

FIXED = SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows


def _before(self, grid, x_vec, s_vec, t_idx, tau, product, pricing_env):
    self._set_boundary_conditions_v0(grid, x_vec, s_vec, t_idx, tau, product, pricing_env)


def rows():
    hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
    ks = np.array(list(100.0 * np.exp(np.linspace(-1.2, 1.2, 15))))
    lev = LeverageSurface(time_grid=np.linspace(0, 1, 6), strike_grid=ks, leverage_grid=np.ones((6, ks.size)))
    return {
        "heston_snowball": lambda: HestonSnowballPDESolver(hp, params=PDEParams(), n_x=80, n_v=32, n_t=40).price(
            snowball(), env_flat()),
        "slv_snowball": lambda: HestonSLVSnowballPDESolver(hp, lev, params=PDEParams(), n_x=80, n_v=32,
                                                           n_t=40).price(snowball(), env_flat()),
        "heston_phoenix": lambda: HestonPhoenixPDESolver(hp, params=PDEParams(), n_x=80, n_v=32, n_t=40).price(
            make_phoenix(dict(PHOENIX, memory_coupon=False)), env_flat()),
    }


if __name__ == "__main__":
    for name, f in rows().items():
        SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows = _before
        b = f()
        SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows = FIXED
        a = f()
        print(f"{name:<16} before {b!r} after {a!r} "
              f"{'bit-identical' if float(a).hex() == float(b).hex() else f'MOVED {a - b:+.3e}'}")
