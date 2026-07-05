"""Backward SLV ADI PDE pricer for European vanillas (deterministic; no MC).

Heston ADI in (x=ln S, v) with a calibrated leverage L(S, t) entering the x-operators:
    A1 U = 0.5 L^2 v U_xx + ((r - carry) - 0.5 L^2 v) U_x
    A2 U = 0.5 (eta sigma)^2 v U_vv + kappa(theta - v) U_v
    A0 U = rho (eta sigma) L v U_xv
The leverage is supplied as a precomputed LeverageSurface (e.g. MC-binning calibrated);
this engine never invokes Monte Carlo. Douglas / Craig-Sneyd schemes with Rannacher
start-up; dense Thomas solves (S-tridiagonals rebuilt each step since L depends on t).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface


def price_european_slv_pde(
    s0: float, strike: float, is_call: bool, T: float, params: HestonParams,
    lev_surface: LeverageSurface, r: float, carry: float, eta: float = 1.0,
    n_x: int = 200, n_v: int = 100, n_t: int = 100,
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD, theta: float = 0.5, rannacher: bool = True,
) -> float:
    """Price a European vanilla under Heston SLV via backward ADI (given a LeverageSurface)."""
    if s0 <= 0 or strike <= 0 or T <= 0:
        raise ValidationError("s0, strike, T must be positive")
    if n_x < 3 or n_v < 3 or n_t < 1:
        raise ValidationError("require n_x>=3, n_v>=3, n_t>=1")
    if not 0.0 <= theta <= 1.0:
        raise ValidationError("theta must be in [0, 1]")
    if not isinstance(scheme, ADIScheme):
        raise ValidationError("scheme must be an ADIScheme")
    if scheme == ADIScheme.MCS:
        raise ValidationError("MCS is not implemented for the SLV PDE; use CRAIG_SNEYD or DOUGLAS")
    if eta < 0:
        raise ValidationError("eta must be non-negative")
    solver = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t,
                              leverage=lev_surface, eta=eta)
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price = solver.interpolate(U, float(np.log(s0)), params.v0)
    if not np.isfinite(price):
        raise NumericalError("SLV PDE produced a non-finite price")
    return price


def price_delta_gamma_slv_pde(
    s0: float, strike: float, is_call: bool, T: float, params: HestonParams,
    lev_surface: LeverageSurface, r: float, carry: float, eta: float = 1.0,
    n_x: int = 200, n_v: int = 100, n_t: int = 100,
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD, theta: float = 0.5, rannacher: bool = True,
) -> Tuple[float, float, float]:
    """(price, spot-delta, spot-gamma) from a single backward SLV PDE solve."""
    if s0 <= 0 or strike <= 0 or T <= 0:
        raise ValidationError("s0, strike, T must be positive")
    if n_x < 3 or n_v < 3 or n_t < 1:
        raise ValidationError("require n_x>=3, n_v>=3, n_t>=1")
    if not 0.0 <= theta <= 1.0:
        raise ValidationError("theta must be in [0, 1]")
    if not isinstance(scheme, ADIScheme):
        raise ValidationError("scheme must be an ADIScheme")
    if scheme == ADIScheme.MCS:
        raise ValidationError("MCS is not implemented for the SLV PDE; use CRAIG_SNEYD or DOUGLAS")
    if eta < 0:
        raise ValidationError("eta must be non-negative")
    solver = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t,
                              leverage=lev_surface, eta=eta)
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price, delta, gamma = solver.price_delta_gamma(U, s0)
    if not (np.isfinite(price) and np.isfinite(delta) and np.isfinite(gamma)):
        raise NumericalError("SLV PDE produced non-finite price/greeks")
    return price, delta, gamma
