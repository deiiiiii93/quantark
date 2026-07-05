"""Heston ADI PDE pricer for European vanillas (Douglas / Craig-Sneyd).

2D finite-difference solver in (x=ln S, v) with implicit treatment in each direction
and an explicit cross-derivative term, ported from the SLV reference HestonPDE. Dense
Thomas solves by default; optional sparse SuperLU. Rannacher start-up (two backward-Euler
half-steps) damps the payoff kink. Asset-neutral: carry = dividend yield (equity) or
foreign rate (FX).

For a European option the Heston price depends only on the terminal forward and discount
factor, so constant curve-consistent (r, carry) is curve-exact (unlike state-dependent
local vol); the engine supplies the zero rate / yield to maturity.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston.params import HestonParams

# Below this vol-of-vol the variance is effectively deterministic and the ADI grid is
# ill-conditioned; the exact deterministic-variance (BS) limit is used instead.
_SIGMA_MIN = 1e-4


def price_european_heston_pde(
    s0: float,
    strike: float,
    is_call: bool,
    T: float,
    params: HestonParams,
    r: float,
    carry: float,
    n_x: int = 200,
    n_v: int = 100,
    n_t: int = 100,
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD,
    theta: float = 0.5,
    rannacher: bool = True,
    use_sparse: bool = False,
    grid_spot: float = 0.0,
    v0_boundary: str = "neumann",
) -> float:
    """Price a European vanilla under Heston via ADI finite differences.

    grid_spot (> 0) pins the spatial grid centering for clean Greeks across spot bumps;
    when 0 the grid centers on s0. v0_boundary "degenerate_pde" replaces the Neumann
    v=0 row with the degenerate convection PDE row (opt-in; helps Feller-violated cases).
    """
    if s0 <= 0 or strike <= 0 or T <= 0:
        raise ValidationError("s0, strike, T must be positive")
    if n_x < 3 or n_v < 3 or n_t < 1:
        raise ValidationError("require n_x>=3, n_v>=3, n_t>=1")
    if not 0.0 <= theta <= 1.0:
        raise ValidationError("theta must be in [0, 1]")
    if not isinstance(scheme, ADIScheme):
        raise ValidationError("scheme must be an ADIScheme")
    if scheme == ADIScheme.MCS:
        raise ValidationError("MCS is not implemented for the Heston PDE; use CRAIG_SNEYD or DOUGLAS")
    if params.sigma < _SIGMA_MIN:
        return _deterministic_pde_price(s0, strike, is_call, T, params, r, carry)
    solver = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t,
                              leverage=None, eta=1.0, use_sparse=use_sparse,
                              grid_spot=(grid_spot if grid_spot > 0 else None),
                              v0_boundary=v0_boundary)
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid (grid_spot too far from s0)")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price = solver.interpolate(U, float(np.log(s0)), params.v0)
    if not np.isfinite(price):
        raise NumericalError("Heston PDE produced a non-finite price")
    return price


def price_delta_gamma_heston_pde(
    s0: float, strike: float, is_call: bool, T: float, params: HestonParams,
    r: float, carry: float, n_x: int = 200, n_v: int = 100, n_t: int = 100,
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD, theta: float = 0.5, rannacher: bool = True,
    use_sparse: bool = False, grid_spot: float = 0.0, v0_boundary: str = "neumann",
) -> Tuple[float, float, float]:
    """Return (price, spot-delta, spot-gamma) from a single PDE solve.

    Delta/gamma are spatial derivatives of the solved surface (per unit of underlying);
    callers scale by notional/contract size.
    """
    if scheme == ADIScheme.MCS:
        raise ValidationError("MCS is not implemented for the Heston PDE; use CRAIG_SNEYD or DOUGLAS")
    if params.sigma < _SIGMA_MIN:
        from quantark.volmodels.black_scholes import bs_call_price, bs_put_price
        eps = 1e-4 * s0
        f = bs_call_price if is_call else bs_put_price
        v_eff = _deterministic_vol(T, params)
        p = f(s0, strike, T, v_eff, r, carry)
        pu = f(s0 + eps, strike, T, v_eff, r, carry)
        pd = f(s0 - eps, strike, T, v_eff, r, carry)
        return p, (pu - pd) / (2 * eps), (pu - 2 * p + pd) / (eps * eps)
    solver = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t,
                              leverage=None, eta=1.0, use_sparse=use_sparse,
                              grid_spot=(grid_spot if grid_spot > 0 else None),
                              v0_boundary=v0_boundary)
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid (grid_spot too far from s0)")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price, delta, gamma = solver.price_delta_gamma(U, s0)
    if not (np.isfinite(price) and np.isfinite(delta) and np.isfinite(gamma)):
        raise NumericalError("Heston PDE produced non-finite price/greeks")
    return price, delta, gamma


def _deterministic_vol(T, params: HestonParams) -> float:
    if params.kappa > 1e-12:
        integrated = params.theta * T + (params.v0 - params.theta) * (
            -np.expm1(-params.kappa * T) / params.kappa
        )
    else:
        integrated = params.v0 * T
    return float(np.sqrt(max(integrated, 0.0) / T))


def _deterministic_pde_price(s0, strike, is_call, T, params, r, carry) -> float:
    from quantark.volmodels.black_scholes import bs_call_price, bs_put_price
    v_eff = _deterministic_vol(T, params)
    return (bs_call_price if is_call else bs_put_price)(s0, strike, T, v_eff, r, carry)
