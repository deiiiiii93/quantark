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
    v0_boundary: str = "neumann",
    grid_style: str = "uniform",
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
                              leverage=lev_surface, eta=eta, v0_boundary=v0_boundary, grid_style=grid_style)
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
    v0_boundary: str = "neumann",
    grid_style: str = "uniform",
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
                              leverage=lev_surface, eta=eta, v0_boundary=v0_boundary, grid_style=grid_style)
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price, delta, gamma = solver.price_delta_gamma(U, s0)
    if not (np.isfinite(price) and np.isfinite(delta) and np.isfinite(gamma)):
        raise NumericalError("SLV PDE produced non-finite price/greeks")
    return price, delta, gamma


def _deterministic_slv_barrier(s0, strike, is_call, T, params, lev_surface, r, carry, eta,
                               barrier, is_up, is_out, rebate, pay_at_hit, continuous, observe_taus,
                               n_x, n_t, market_context=None):
    """BSM-limit (sigma -> 0) SLV barrier via the 1-D LV kernel at effective vol L(S,t)*sqrt(v_det)."""
    from quantark.volmodels.localvol.surface import LocalVolSurface
    from quantark.volmodels.localvol.pde_kernel import price_barrier_lv_pde
    from quantark.volmodels.heston.pde_kernel import _deterministic_vol
    v_eff = _deterministic_vol(T, params)
    ks = s0 * np.exp(np.linspace(-2.5, 2.5, 15))
    tg = np.linspace(0.0, max(T, 1e-8), 5)
    lv_grid = np.array([[max(float(np.asarray(lev_surface.leverage(np.array([k]), float(t))).ravel()[0]), 1e-8) * v_eff
                         for k in ks] for t in tg])
    lv = LocalVolSurface(strike_grid=ks, time_grid=tg, lv_grid=lv_grid)
    if market_context is None:
        n = max(int(n_t), 100)
        dt = np.full(n, T / n)
        rf = np.full(n, float(r))
        cf = np.full(n, float(carry))
    else:
        dt = np.diff(market_context.t_grid)
        rf = market_context.fwd_rates
        cf = market_context.fwd_carry
        n = int(dt.size)
    if continuous:
        obs_steps = None
    else:
        t_grid = np.linspace(0.0, T, n + 1)
        obs_steps = [int(np.argmin(np.abs(t_grid - (T - float(o))))) for o in (observe_taus or [])]
    return price_barrier_lv_pde(s0, strike, is_call, T, lv, dt, rf, cf, barrier=float(barrier),
                                is_up=bool(is_up), is_out=bool(is_out), rebate=float(rebate),
                                pay_at_hit=bool(pay_at_hit), continuous=continuous,
                                observe_steps=obs_steps, n_s=max(int(n_x), 400))


def _observation_step_set(observe_taus, dt_grid, n_t):
    return {
        min(max(int(round(float(tau) / dt_grid)), 0), int(n_t))
        for tau in (observe_taus or [])
    }


def _is_observation_tau(tau, dt_grid, obs_ks):
    if obs_ks is None:
        return True
    k_float = float(tau) / dt_grid
    k = int(round(k_float))
    # Rannacher startup evaluates the hook at half steps. Do not let e.g.
    # tau=0.5*dt round into the terminal observation bucket.
    if abs(k_float - k) > 1e-8:
        return False
    return k in obs_ks


def price_barrier_slv_pde(
    s0, strike, is_call, T, params, lev_surface, r, carry, barrier, is_up, is_out,
    rebate=0.0, pay_at_hit=False, continuous=True, observe_taus=None, eta=1.0,
    n_x=200, n_v=100, n_t=100, scheme=ADIScheme.CRAIG_SNEYD, theta=0.5, rannacher=True,
    market_context=None,
):
    """Single-barrier option under Heston-SLV via 2D ADI.

    Continuous monitoring truncates the log-spot domain AT the barrier with a Dirichlet KO
    boundary (exact); discrete monitoring keeps the full domain, CONCENTRATES the grid on the
    barrier (node pinned there), and injects the KO at the observation steps. Same machinery as
    ``price_barrier_heston_pde`` over the leverage-scaled operators.
    Knock-in = Vanilla - KO(rebate=0) + rebate * NoTouch.
    """
    from quantark.volmodels.barrier import BarrierSpec, validate_barrier
    from quantark.volmodels.heston.pde_kernel import _SIGMA_MIN
    spec = BarrierSpec(bool(is_up), bool(is_out), bool(is_call), float(barrier), float(strike),
                       float(rebate), bool(pay_at_hit))
    validate_barrier(spec, s0)

    if params.sigma < _SIGMA_MIN:
        # Deterministic-variance limit: effective local vol = L(S,t) * sqrt(v_det). Price via the
        # exact 1-D local-vol kernel (the ADI grid is ill-conditioned at near-zero vol-of-vol).
        return _deterministic_slv_barrier(s0, strike, is_call, T, params, lev_surface, r, carry, eta,
                                          barrier, is_up, is_out, rebate, pay_at_hit, continuous,
                                          observe_taus, n_x, n_t, market_context=market_context)

    x0, v0 = float(np.log(s0)), params.v0
    dt_grid = float(T) / float(n_t)
    obs_ks = None if continuous else _observation_step_set(observe_taus, dt_grid, n_t)

    def _ko_leg(reb):
        """Knock-OUT leg with rebate ``reb``. Continuous -> domain truncation + Dirichlet KO
        boundary (exact); discrete -> full barrier-concentrated domain, KO injected at obs steps."""
        if continuous:
            core = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t,
                                    leverage=lev_surface, eta=eta, barrier=float(barrier),
                                    barrier_is_up=bool(is_up), rebate=float(reb), pay_at_hit=bool(pay_at_hit),
                                    market_context=market_context)
            U = core.solve(is_call, scheme, theta, rannacher)
            return core.interpolate(U, x0, v0)
        core = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t, leverage=lev_surface,
                                eta=eta, grid_style="concentrated", barrier_concentrate=float(barrier),
                                market_context=market_context)
        tol = 1e-9 * float(barrier)
        # Discrete monitoring has a jump discontinuity at the barrier. The grid is pinned at the
        # barrier for resolution, but forcing that pinned node to the KO value makes the following
        # backward interval behave like a continuous absorbing boundary on coarse grids. Since
        # S == barrier has zero probability under the diffusion, keep the pinned node as the
        # surviving-side left/right limit and zero only nodes strictly beyond the barrier.
        ko_rows = (core.S_grid > barrier + tol) if is_up else (core.S_grid < barrier - tol)

        def hook(U, tau):
            if not _is_observation_tau(tau, dt_grid, obs_ks):
                return U
            U = np.array(U, dtype=float)
            U[ko_rows, :] = float(reb) if pay_at_hit else float(reb) * core.df_to_maturity(tau)
            return U

        U = core.solve(is_call, scheme, theta, rannacher, step_hook=hook)
        return core.interpolate(U, x0, v0)

    if is_out:
        return _ko_leg(rebate)

    van = price_european_slv_pde(s0, strike, is_call, T, params, lev_surface, r, carry, eta,
                                 n_x, n_v, n_t, scheme=scheme, theta=theta, rannacher=rannacher)
    ki = van - _ko_leg(0.0)
    if rebate > 0.0:
        # NoTouch (survival) leg: full-domain KO injection with terminal payoff 1.
        core3 = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t,
                                 leverage=lev_surface, eta=eta,
                                 grid_style=("concentrated" if not continuous else "uniform"),
                                 barrier_concentrate=(float(barrier) if not continuous else 0.0),
                                 market_context=market_context)
        tol3 = 1e-9 * float(barrier)
        ko_rows3 = (core3.S_grid > barrier + tol3) if is_up else (core3.S_grid < barrier - tol3)
        obs3 = None if continuous else obs_ks

        def hook_nt(U, tau):
            if not _is_observation_tau(tau, dt_grid, obs3):
                return U
            U = np.array(U, dtype=float)
            U[ko_rows3, :] = 0.0
            return U

        ones = np.ones((core3.X_grid.size, core3.V_grid.size))
        Un = core3.solve(is_call, scheme, theta, rannacher, step_hook=hook_nt, terminal_override=ones)
        ki += float(rebate) * core3.interpolate(Un, x0, v0)
    return ki
