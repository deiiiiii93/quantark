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
    grid_style: str = "uniform",
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
                              v0_boundary=v0_boundary, grid_style=grid_style)
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
    grid_style: str = "uniform",
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
                              v0_boundary=v0_boundary, grid_style=grid_style)
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


def _deterministic_barrier_price(s0, strike, is_call, T, params, r, carry, barrier, is_up, is_out,
                                 rebate, pay_at_hit, continuous, observe_taus, n_x, n_t,
                                 market_context=None):
    """BSM-limit (sigma -> 0) barrier price via the 1-D local-vol kernel at constant vol."""
    from quantark.volmodels.localvol.surface import LocalVolSurface
    from quantark.volmodels.localvol.pde_kernel import price_barrier_lv_pde
    v_eff = _deterministic_vol(T, params)
    ks = s0 * np.exp(np.linspace(-2.5, 2.5, 7))
    lv = LocalVolSurface(strike_grid=ks, time_grid=np.array([0.0, max(T, 1e-8)]),
                         lv_grid=np.full((2, ks.size), v_eff))
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


def price_barrier_heston_pde(
    s0, strike, is_call, T, params, r, carry, barrier, is_up, is_out,
    rebate=0.0, pay_at_hit=False, continuous=True, observe_taus=None,
    n_x=200, n_v=100, n_t=100, scheme=ADIScheme.CRAIG_SNEYD, theta=0.5, rannacher=True,
    market_context=None,
):
    """Single-barrier option under Heston via 2D ADI.

    Continuous monitoring truncates the log-spot domain AT the barrier and imposes a Dirichlet
    knock-out value on that boundary (exact continuous monitoring, matching the 1-D LV kernel).
    Discrete monitoring keeps the full domain and injects the KO condition at the observation
    steps via a step_hook. Knock-in = Vanilla - KO(rebate=0) + rebate * NoTouch.
    """
    from quantark.volmodels.barrier import BarrierSpec, validate_barrier
    spec = BarrierSpec(bool(is_up), bool(is_out), bool(is_call), float(barrier), float(strike),
                       float(rebate), bool(pay_at_hit))
    validate_barrier(spec, s0)

    if params.sigma < _SIGMA_MIN:
        # Deterministic-variance limit: the ADI grid is ill-conditioned, so price the barrier via
        # the exact 1-D local-vol kernel at the BSM-equivalent constant vol (mirrors the European
        # kernel's _deterministic bypass). Truncation (continuous) / injection (discrete) live there.
        return _deterministic_barrier_price(s0, strike, is_call, T, params, r, carry, barrier, is_up,
                                            is_out, rebate, pay_at_hit, continuous, observe_taus, n_x, n_t,
                                            market_context=market_context)

    x0, v0 = float(np.log(s0)), params.v0
    dt_grid = float(T) / float(n_t)
    obs_ks = None if continuous else _observation_step_set(observe_taus, dt_grid, n_t)

    def _ko_leg(reb):
        """Knock-OUT leg with rebate ``reb``. Continuous -> domain truncation + Dirichlet KO
        boundary (exact); discrete -> full domain with KO injected at the observation steps."""
        if continuous:
            core = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t, leverage=None,
                                    eta=1.0, barrier=float(barrier), barrier_is_up=bool(is_up),
                                    rebate=float(reb), pay_at_hit=bool(pay_at_hit),
                                    market_context=market_context)
            U = core.solve(is_call, scheme, theta, rannacher)
            return core.interpolate(U, x0, v0)
        # Discrete: full domain (the value lives above the barrier between fixings) with the grid
        # CONCENTRATED on the barrier and a node pinned there, so the KO injection is well-resolved.
        core = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t, leverage=None, eta=1.0,
                                grid_style="concentrated", barrier_concentrate=float(barrier),
                                market_context=market_context)
        tol = 1e-9 * float(barrier)
        # Discrete monitoring has a jump discontinuity at the barrier. The grid is pinned at the
        # barrier for resolution, but forcing that pinned node to the KO value makes the following
        # backward interval behave like a continuous absorbing boundary on coarse grids. Since
        # S == barrier has zero probability under the diffusion, keep the pinned node as the
        # surviving-side left/right limit and zero only nodes strictly beyond the barrier.
        ko_rows = (core.S_grid > barrier + tol) if is_up else (core.S_grid < barrier - tol)

        def hook(U, tau):
            # Inject KO only at observation steps; the terminal step (tau=0) is an observation
            # ONLY if maturity is in the schedule (0 in obs_ks), else an unmonitored tail must pay.
            if not _is_observation_tau(tau, dt_grid, obs_ks):
                return U
            U = np.array(U, dtype=float)
            U[ko_rows, :] = float(reb) if pay_at_hit else float(reb) * core.df_to_maturity(tau)
            return U

        U = core.solve(is_call, scheme, theta, rannacher, step_hook=hook)
        return core.interpolate(U, x0, v0)

    if is_out:
        return _ko_leg(rebate)

    van = price_european_heston_pde(s0, strike, is_call, T, params, r, carry, n_x, n_v, n_t,
                                    scheme=scheme, theta=theta, rannacher=rannacher)
    ki = van - _ko_leg(0.0)
    if rebate > 0.0:
        # NoTouch (survival) leg: full-domain KO injection with terminal payoff 1. (Continuous
        # truncation of the survival leg is a distinct boundary problem; kept on the injection path.)
        core3 = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t, leverage=None, eta=1.0,
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
