"""Unified Heston / Heston-SLV ADI PDE core (Douglas / Craig-Sneyd).

2D finite-difference solver in (x = ln S, v). The Heston model is the special case of the
SLV model with leverage ``L ≡ 1`` and vol-of-vol scale ``η = 1``:

    A1 U = 0.5 L^2 v U_xx + ((r - carry) - 0.5 L^2 v) U_x - r U     (S-direction + reaction)
    A2 U = 0.5 (η σ)^2 v U_vv + κ(θ - v) U_v                        (v-direction)
    A0 U = ρ (η σ) L v U_xv                                         (mixed term)

``leverage=None`` selects the constant-``L`` (Heston) path, which enables coefficient
caching and the optional sparse-LU solve; a supplied ``LeverageSurface`` gives the SLV
path (S-coefficients rebuilt each step because ``L`` depends on t). Both paths support
``grid_spot`` pinning for clean bump Greeks. Dense batched-Thomas solves by default;
optional SuperLU for the constant-``L`` case. Deterministic — never invokes Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import (
    solve_tridiag_batch, fd1_interior_coeffs, fd2_interior_coeffs,
)
from quantark.volmodels.heston.params import HestonParams
# NB: concentrated_grid / z_extents are imported lazily inside __init__ (only on the
# opt-in concentrated path) to avoid a circular import via quantark.volmodels.slv.__init__.


def _positive_spot_levels(value) -> list[float]:
    if value is None:
        return []
    try:
        arr = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        return []
    return [float(v) for v in arr.ravel() if np.isfinite(v) and v > 0.0]


def _unique_sorted(values: list[float], tol: float = 1e-12) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values if np.isfinite(v)):
        if not out or abs(value - out[-1]) > tol:
            out.append(value)
    return out


def _pin_grid_targets(grid: np.ndarray, targets: list[float]) -> np.ndarray:
    """Move nearest interior nodes exactly onto requested targets while preserving order."""
    if not targets:
        return grid

    pinned = np.asarray(grid, dtype=float).copy()
    used: set[int] = set()
    n = pinned.size
    for target in _unique_sorted(targets):
        if target <= pinned[0] or target >= pinned[-1]:
            continue
        for idx_raw in np.argsort(np.abs(pinned - target)):
            idx = int(idx_raw)
            if idx == 0 or idx == n - 1 or idx in used:
                continue
            if pinned[idx - 1] < target < pinned[idx + 1]:
                pinned[idx] = target
                used.add(idx)
                break

    if np.any(np.diff(pinned) <= 0):
        raise ValidationError("non-monotone grid after critical point pinning")
    return pinned


class HestonSLVADICore:
    """Shared ADI core for the Heston and Heston-SLV backward PDEs.

    Args:
        s0, strike, T, r, carry: option / market spec (carry = dividend or foreign rate).
        params: HestonParams (κ, θ, σ, ρ, v0).
        n_x, n_v, n_t: grid sizes (log-spot, variance, time).
        leverage: LeverageSurface, or None for constant L ≡ 1 (the Heston case).
        eta: vol-of-vol scale; ``sig_eff = eta * sigma``.
        use_sparse: use per-slice SuperLU in the S/V solves (only valid when L is constant).
        grid_spot: center the log-spot grid here (defaults to s0); pin it across spot
            bumps for clean delta/gamma.
    """

    def __init__(self, s0, strike, T, r, carry, params: HestonParams,
                 n_x, n_v, n_t, *, market_context=None, leverage=None, eta=1.0,
                 use_sparse=False, grid_spot=None, v0_boundary="neumann",
                 grid_style="uniform", barrier=0.0, barrier_is_up=True,
                 rebate=0.0, pay_at_hit=False, barrier_concentrate=0.0,
                 critical_spots=None, v_grid_power=0.0, x_nodes=None):
        self.S0, self.K, self.T, self.r, self.q = s0, strike, T, r, carry
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = (
            params.kappa, params.theta, params.sigma, params.rho, params.v0,
        )
        self.lev = leverage
        self._constant_leverage = leverage is None
        self.sig_eff = float(eta) * self.sigma
        self.sig_eff2 = self.sig_eff ** 2
        self._opt_is_call = True
        self.N_S, self.N_V, self.N_T = n_x, n_v, n_t
        self.market_context = market_context
        self._time_grid = np.linspace(0.0, float(T), int(n_t) + 1)
        if market_context is None:
            self.r_fwd = np.full(int(n_t), float(r))
            self.q_fwd = np.full(int(n_t), float(carry))
            self.node_dfs = np.exp(-float(r) * self._time_grid)
            self.carry_node_dfs = np.exp(-float(carry) * self._time_grid)
        else:
            market_grid = np.asarray(market_context.t_grid, dtype=float)
            if market_grid.shape != (int(n_t) + 1,):
                raise ValidationError("market_context grid must have n_t + 1 nodes")
            if (
                abs(float(market_grid[0])) > 1e-12
                or abs(float(market_grid[-1]) - float(T)) > 1e-12
            ):
                raise ValidationError("market_context grid must run from 0 to T")
            self._time_grid = market_grid
            self.r_fwd = np.asarray(market_context.fwd_rates, dtype=float)
            self.q_fwd = np.asarray(market_context.fwd_carry, dtype=float)
            self.node_dfs = np.asarray(market_context.node_dfs, dtype=float)
            self.carry_node_dfs = np.asarray(market_context.carry_node_dfs, dtype=float)
        if v0_boundary not in ("neumann", "degenerate_pde"):
            raise ValidationError("v0_boundary must be 'neumann' or 'degenerate_pde'")
        self.v0_boundary = v0_boundary
        self._degenerate_v0 = v0_boundary == "degenerate_pde"
        if grid_style not in ("uniform", "concentrated"):
            raise ValidationError("grid_style must be 'uniform' or 'concentrated'")
        self._uniform = grid_style == "uniform"
        self._v_grid_power = float(v_grid_power)
        if self._v_grid_power < 0.0 or not np.isfinite(self._v_grid_power):
            raise ValidationError("v_grid_power must be a finite value >= 0")
        if self._v_grid_power > 0.0:
            if self._v_grid_power < 1.0:
                raise ValidationError("v_grid_power must be >= 1 when set")
            if self._uniform:
                raise ValidationError(
                    "v_grid_power requires grid_style='concentrated'"
                )
        # Sparse SuperLU is only valid when the S-operator is time-independent (L constant)
        # AND the grid is uniform (the concentrated path always uses batched-Thomas).
        self.use_sparse = bool(use_sparse) and self._constant_leverage and self._uniform

        # ---- grid (identical extent logic to both original kernels) ----
        var_eff = max(self.theta, self.v0, 0.25 * self.sig_eff2, 0.04)
        x_width = 8.0 * np.sqrt(var_eff * max(T, 1e-12))
        grid_center = grid_spot if grid_spot is not None else s0
        x_center = float(np.log(max(grid_center, 1e-12)))
        self.x_min, self.x_max = x_center - x_width, x_center + x_width
        self.V_max = max(5.0 * self.theta, 0.5, 2.0 * self.v0)

        # Continuous knock-out barrier (opt-in; barrier==0 leaves grid + boundaries untouched, so
        # European and discrete-injection pricing are byte-identical). Truncate the log-spot domain
        # AT the barrier and impose a Dirichlet KO value on that boundary throughout the solve --
        # exact continuous monitoring, no region-zeroing pollution (the 1-D LV kernel's approach).
        self._barrier_active = bool(barrier and barrier > 0.0)
        self._barrier_is_up = bool(barrier_is_up)
        self._barrier_rebate = float(rebate)
        self._barrier_pay_at_hit = bool(pay_at_hit)
        if self._barrier_active:
            x_b = float(np.log(barrier))
            if self._barrier_is_up:
                self.x_max = x_b
            else:
                self.x_min = x_b

        # Externally supplied S-axis nodes (declarative grid layer, spec
        # §4.6): the log-spot mesh comes from the SAME builder every 1D
        # solver uses — spot/strike/barrier concentration, bump-frozen —
        # replacing the internal var_eff width heuristic for adopters. The
        # nodes are generally non-uniform, so they route through the
        # concentrated-stencil code path; the variance axis stays internal
        # (model-specific by design).
        self._external_x = x_nodes is not None
        if self._external_x:
            x_arr = np.asarray(x_nodes, dtype=float)
            if x_arr.ndim != 1 or x_arr.shape[0] != n_x:
                raise ValidationError(
                    f"x_nodes must be a 1-D array of length n_x={n_x}"
                )
            if np.any(np.diff(x_arr) <= 0.0):
                raise ValidationError("x_nodes must be strictly increasing")
            if self._barrier_active and not (
                abs(float(x_arr[-1]) - self.x_max) < 1e-12
                or abs(float(x_arr[0]) - self.x_min) < 1e-12
            ):
                raise ValidationError(
                    "x_nodes must terminate at the active barrier boundary"
                )
            self.x_min, self.x_max = float(x_arr[0]), float(x_arr[-1])
            self.X_grid = x_arr
            self.use_sparse = False  # injected meshes use batched-Thomas

        if self._uniform and self._external_x:
            # Injected S nodes + uniform variance request: general stencils
            # on the (non-uniform) injected mesh, plain linspace variance.
            self.V_grid = np.linspace(0.0, self.V_max, n_v)
            self._xx = fd2_interior_coeffs(self.X_grid)
            self._x1 = fd1_interior_coeffs(self.X_grid)
            self._vv = fd2_interior_coeffs(self.V_grid)
            self._v1 = fd1_interior_coeffs(self.V_grid)
            self.dx = None  # scalar spacing undefined on an injected mesh
            self.dV = None
            self._uniform = False  # march takes the general-stencil path
        elif self._uniform:
            self.X_grid = np.linspace(self.x_min, self.x_max, n_x)
            self.V_grid = np.linspace(0.0, self.V_max, n_v)
            self.dx = float(self.X_grid[1] - self.X_grid[0])
            self.dV = float(self.V_grid[1] - self.V_grid[0])
        else:
            # x concentrated around ln K (payoff kink); v concentrated around min(v0, theta),
            # widening V_max by a CIR-quantile upper extent when vol-of-vol is live.
            from quantark.volmodels.slv.fokkerplanck.coordinates import (
                concentrated_grid, z_extents,
            )
            concentration_levels = _positive_spot_levels(barrier_concentrate)
            critical_levels = _positive_spot_levels(critical_spots)
            # Concentration center: caller-supplied critical level when provided
            # (for barriers/exotics), else ln K (the vanilla payoff kink).
            if not self._external_x:
                if concentration_levels:
                    xk = float(np.log(concentration_levels[0]))
                    conc = max(0.06 * x_width, 1e-6)   # tight cluster around the selected critical level
                else:
                    xk = float(np.log(max(self.K, 1e-12)))
                    conc = max(0.25 * x_width, 1e-6)
                xk = min(max(xk, self.x_min), self.x_max)
                self.X_grid = concentrated_grid(self.x_min, self.x_max, xk, n_x, concentration=conc)
                pin_targets = [
                    float(np.log(level))
                    for level in concentration_levels + critical_levels
                    if level > 0.0 and self.x_min < float(np.log(level)) < self.x_max
                ]
                self.X_grid = _pin_grid_targets(self.X_grid, pin_targets)
            if self.sig_eff > 0.0:
                t_probe = np.array([0.25 * T, 0.5 * T, T])
                try:
                    _, q_hi = z_extents(params, float(eta), t_probe,
                                        cir_quantile=1e-5, v_floor=1e-8)
                    self.V_max = max(self.V_max, q_hi)
                except ValidationError:
                    pass  # keep the envelope V_max (degenerate CIR)
            if self._v_grid_power > 0.0:
                # Power-graded variance grid v = V_max * u^gamma, clustering
                # nodes at the v = 0 boundary. When Feller is violated
                # (2*kappa*theta < sigma_eff^2) the value function behaves
                # like v^alpha with alpha = 2*kappa*theta/sigma_eff^2 < 1 at
                # the boundary; second-order stencils on ungraded grids then
                # converge like O(h^~0.5) and the graded grid restores
                # practical accuracy (measured 25% -> <1% vanilla error at
                # alpha ~= 0.36 with gamma ~= 2-3).
                u = np.linspace(0.0, 1.0, n_v)
                self.V_grid = self.V_max * u ** self._v_grid_power
            else:
                v_center = min(max(self.v0, 0.0), self.theta) if self.theta > 0 else self.v0
                v_center = min(max(v_center, 0.0), self.V_max)
                self.V_grid = concentrated_grid(0.0, self.V_max, v_center, n_v,
                                                concentration=max(0.5 * self.V_max, 1e-6))
            # per-node interior stencil coefficients for both directions
            self._xx = fd2_interior_coeffs(self.X_grid)   # (wm, w0, wp) each (n_x-2,)
            self._x1 = fd1_interior_coeffs(self.X_grid)
            self._vv = fd2_interior_coeffs(self.V_grid)
            self._v1 = fd1_interior_coeffs(self.V_grid)
            self.dx = None  # scalar spacing undefined on a concentrated grid
            self.dV = None

        self.S_grid = np.exp(self.X_grid)
        self.S_max = float(self.S_grid[-1])
        self.dt = float(T / max(n_t, 1))
        self._S_int = self.S_grid[1:-1]
        self._ones_int = np.ones(self.N_S - 2)

        # caches: _tri_V always (time-independent); _tri_S only when L constant.
        self._S_tri_cache: dict = {}
        self._V_tri_cache: dict = {}
        self._S_lu_cache: dict = {}
        self._V_lu_cache: dict = {}
        self._boundary_hook = None

    def _L(self, t):
        if self._constant_leverage:
            return self._ones_int
        return np.asarray(self.lev.leverage(self._S_int, t), dtype=float)

    def _forward_step_index(self, t: float) -> int:
        idx = int(np.searchsorted(self._time_grid, float(t), side="right") - 1)
        return min(max(idx, 0), self.r_fwd.size - 1)

    def _rq_at(self, t: float) -> tuple[float, float]:
        idx = self._forward_step_index(t)
        return float(self.r_fwd[idx]), float(self.q_fwd[idx])

    def forward_rates_at(self, t: float) -> tuple[float, float]:
        """The ``(r, q)`` the tridiagonal assembly uses for a step at ``t``.

        Public because collaborators that must stay consistent with the
        DISCRETIZED dynamics -- barrier corrections in particular -- have to
        sample this schedule rather than re-deriving one from the environment.
        """
        return self._rq_at(t)

    def _advance_tau(self, tau: float, step: float, final: bool) -> float:
        """Advance the backward time-to-maturity by one (half-)step.

        ``tau`` is the node's time-to-maturity, so the loop's last landing
        node IS the valuation date and its exact value is ``T``.  Summing
        ``dt = T / n_t`` cannot reproduce that: ``T / n_t`` is not a binary
        fraction for most ``n_t``, so the accumulated sum misses ``T`` by an
        ULP either way (``T=1.0, n_t=20`` overshoots by 2.2e-16).  Callers
        read calendar time back out as ``T - tau`` and hand it to the
        environment's rate curve, which fails closed on negative times, so an
        overshoot of one ULP raised ``ValidationError`` from the
        boundary-condition path of the 2D snowball/phoenix wrappers.

        The final node is therefore set to ``T`` exactly rather than
        accumulated, and every earlier node is clamped below ``T`` (a no-op
        in exact arithmetic — intermediate nodes sit a whole step short of
        maturity — kept as a guard so no consumer can ever see ``tau > T``).
        """
        if final:
            return float(self.T)
        return min(tau + step, float(self.T))

    def _forward_time_from_tau(self, tau: float) -> float:
        return min(max(float(self.T) - float(tau), 0.0), float(self.T))

    def _node_factor_at_time(
        self,
        t: float,
        node_factors: np.ndarray,
        fwd_values: np.ndarray,
    ) -> float:
        current = min(max(float(t), 0.0), float(self.T))
        if current <= self._time_grid[0] + 1e-14:
            return float(node_factors[0])
        if current >= self._time_grid[-1] - 1e-14:
            return float(node_factors[-1])
        idx = self._forward_step_index(current)
        return float(
            node_factors[idx]
            * np.exp(-float(fwd_values[idx]) * (current - self._time_grid[idx]))
        )

    def _factor_to_maturity(
        self,
        tau: float,
        node_factors: np.ndarray,
        fwd_values: np.ndarray,
    ) -> float:
        current = self._forward_time_from_tau(tau)
        node_factor = self._node_factor_at_time(current, node_factors, fwd_values)
        return float(node_factors[-1] / node_factor)

    def df_to_maturity(self, tau: float) -> float:
        return self._factor_to_maturity(tau, self.node_dfs, self.r_fwd)

    def carry_df_to_maturity(self, tau: float) -> float:
        return self._factor_to_maturity(tau, self.carry_node_dfs, self.q_fwd)

    # ---- operators ----
    def _A1(self, U, t):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3:
            return out
        r_step, q_step = self._rq_at(t)
        v_int = self.V_grid[1:-1]
        L2v = (self._L(t) ** 2)[:, None] * v_int[None, :]
        if self._uniform:
            U_xx = (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / (self.dx * self.dx)
            U_x = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2.0 * self.dx)
        else:
            wm2, w02, wp2 = self._xx
            wm1, w01, wp1 = self._x1
            U_xx = (U[:-2, 1:-1] * wm2[:, None] + U[1:-1, 1:-1] * w02[:, None]
                    + U[2:, 1:-1] * wp2[:, None])
            U_x = (U[:-2, 1:-1] * wm1[:, None] + U[1:-1, 1:-1] * w01[:, None]
                   + U[2:, 1:-1] * wp1[:, None])
        # WS-C1: the -rU reaction is carried implicitly in the S-direction (folded into A1
        # here and into the _tri_S diagonal), so the predictor no longer applies it.
        out[1:-1, 1:-1] = (0.5 * L2v * U_xx
                           + ((r_step - q_step) - 0.5 * L2v) * U_x
                           - r_step * U[1:-1, 1:-1])
        return out

    def _A2(self, U):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3:
            return out
        v_int = self.V_grid[1:-1]
        coef_d2 = 0.5 * self.sig_eff2 * v_int
        coef_d1 = self.kappa * (self.theta - v_int)
        if self._uniform:
            U_VV = (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / (self.dV * self.dV)
            U_V = (U[1:-1, 2:] - U[1:-1, :-2]) / (2.0 * self.dV)
        else:
            wm2, w02, wp2 = self._vv
            wm1, w01, wp1 = self._v1
            U_VV = U[1:-1, :-2] * wm2 + U[1:-1, 1:-1] * w02 + U[1:-1, 2:] * wp2
            U_V = U[1:-1, :-2] * wm1 + U[1:-1, 1:-1] * w01 + U[1:-1, 2:] * wp1
        out[1:-1, 1:-1] = coef_d2 * U_VV + coef_d1 * U_V
        if self._degenerate_v0:
            # v=0 row: only kappa*theta*U_v survives (diffusion vanishes); 2-point forward.
            dV0 = float(self.V_grid[1] - self.V_grid[0])
            out[1:-1, 0] = self.kappa * self.theta * (U[1:-1, 1] - U[1:-1, 0]) / dV0
        return out

    def _A0(self, U, t):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3 or abs(self.rho) <= 1e-12:
            return out
        v_int = self.V_grid[1:-1]
        Lv = self._L(t)[:, None] * v_int[None, :]
        if self._uniform:
            U_xv = (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4.0 * self.dx * self.dV)
        else:
            # non-uniform cross derivative = fd1_x then fd1_v (outer product of 1D stencils)
            wxm, wx0, wxp = self._x1
            wvm, wv0, wvp = self._v1
            Ux = U[:-2, :] * wxm[:, None] + U[1:-1, :] * wx0[:, None] + U[2:, :] * wxp[:, None]
            U_xv = Ux[:, :-2] * wvm + Ux[:, 1:-1] * wv0 + Ux[:, 2:] * wvp
        out[1:-1, 1:-1] = self.rho * self.sig_eff * Lv * U_xv
        return out

    def _ko_bnd(self, tau):
        """Dirichlet knock-out value at the truncated barrier boundary (rebate now vs discounted)."""
        return (
            self._barrier_rebate
            if self._barrier_pay_at_hit
            else self._barrier_rebate * self.df_to_maturity(tau)
        )

    def _bc(self, U, tau):
        df = self.df_to_maturity(tau)
        carry_df = self.carry_df_to_maturity(tau)
        if self._opt_is_call:
            U[0, :] = 0.0
            U[-1, :] = max(0.0, self.S_max * carry_df - self.K * df)
        else:
            U[0, :] = self.K * df
            U[-1, :] = 0.0
        if self._barrier_active:  # override the barrier-side x-boundary with the KO Dirichlet value
            if self._barrier_is_up:
                U[-1, :] = self._ko_bnd(tau)
            else:
                U[0, :] = self._ko_bnd(tau)
        if not self._degenerate_v0:
            U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]
        if self._boundary_hook is not None:
            hooked = self._boundary_hook(U, tau)
            if hooked is not None:
                U[:, :] = hooked

    def _terminal(self, is_call):
        S_mesh, _ = np.meshgrid(self.S_grid, self.V_grid, indexing="ij")
        self._opt_is_call = is_call
        U = np.maximum(S_mesh - self.K, 0.0) if is_call else np.maximum(self.K - S_mesh, 0.0)
        self._bc(U, 0.0)
        return U

    def _s_boundary_rhs(self, rhs, tau, v_index=0):
        df = self.df_to_maturity(tau)
        carry_df = self.carry_df_to_maturity(tau)
        if self._opt_is_call:
            rhs[0] = 0.0
            rhs[-1] = max(0.0, self.S_max * carry_df - self.K * df)
        else:
            rhs[0] = self.K * df
            rhs[-1] = 0.0
        if self._barrier_active:
            if self._barrier_is_up:
                rhs[-1] = self._ko_bnd(tau)
            else:
                rhs[0] = self._ko_bnd(tau)
        custom = self._custom_s_boundary_values(tau)
        if custom is not None:
            low, high = custom
            rhs[0] = low[int(v_index)]
            rhs[-1] = high[int(v_index)]
        return rhs

    def _custom_s_boundary_values(self, tau):
        if self._boundary_hook is None:
            return None
        U = np.zeros((self.N_S, self.N_V), dtype=float)
        hooked = self._boundary_hook(U, tau)
        if hooked is not None:
            U[:, :] = hooked
        return U[0, :].copy(), U[-1, :].copy()

    # ---- tridiagonal builders (cached where the operator is time-independent) ----
    def _tri_S(self, dt_step, theta_loc, t_mid):
        r_step, q_step = self._rq_at(t_mid)
        if self._constant_leverage:
            key = (float(dt_step), float(theta_loc), self._forward_step_index(t_mid))
            cached = self._S_tri_cache.get(key)
            if cached is not None:
                return cached
        L2 = self._L(t_mid) ** 2                                # (N_S-2,) interior
        V = np.maximum(self.V_grid, 1e-10)[:, None]             # (N_V, 1)
        a = np.zeros((self.N_V, self.N_S))
        b = np.ones((self.N_V, self.N_S))
        c = np.zeros((self.N_V, self.N_S))
        if self._uniform:
            c2 = 0.5 * (L2[None, :] * V) / (self.dx * self.dx)      # (N_V, N_S-2)
            c1 = ((r_step - q_step) - 0.5 * (L2[None, :] * V)) / (2.0 * self.dx)
            a[:, 1:-1] = -theta_loc * dt_step * (c2 - c1)
            # WS-C1: (I - theta*dt*A1) diagonal gains +theta*dt*r from the implicit -rU reaction.
            b[:, 1:-1] = 1.0 + theta_loc * dt_step * (2.0 * c2 + r_step)
            c[:, 1:-1] = -theta_loc * dt_step * (c2 + c1)
        else:
            # per-node operator L = d2*fd2 + d1*fd1; implicit (I - theta*dt*L) with the
            # WS-C1 +r reaction folded onto the diagonal. x-coeffs broadcast across V-slices.
            d2 = 0.5 * (L2[None, :] * V)                            # (N_V, N_S-2) diffusion coeff
            d1 = (r_step - q_step) - 0.5 * (L2[None, :] * V)        # convection coeff
            wm2, w02, wp2 = self._xx                                # (N_S-2,)
            wm1, w01, wp1 = self._x1
            sub_op = d2 * wm2[None, :] + d1 * wm1[None, :]
            diag_op = d2 * w02[None, :] + d1 * w01[None, :]
            sup_op = d2 * wp2[None, :] + d1 * wp1[None, :]
            a[:, 1:-1] = -theta_loc * dt_step * sub_op
            b[:, 1:-1] = 1.0 - theta_loc * dt_step * diag_op + theta_loc * dt_step * r_step
            c[:, 1:-1] = -theta_loc * dt_step * sup_op
        if self._constant_leverage:
            self._S_tri_cache[key] = (a, b, c)
        return a, b, c

    def _tri_V(self, dt_step, theta_loc):
        key = (float(dt_step), float(theta_loc))
        cached = self._V_tri_cache.get(key)
        if cached is not None:
            return cached
        N = self.N_V
        v = np.maximum(self.V_grid, 1e-10)
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
        if self._uniform:
            dV = self.dV
            coef_d2 = 0.5 * self.sig_eff2 * v / (dV * dV)
            coef_d1 = self.kappa * (self.theta - v) / (2.0 * dV)
            a[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] - coef_d1[1:-1])
            b[1:-1] = 1.0 + theta_loc * dt_step * (2.0 * coef_d2[1:-1])
            c[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] + coef_d1[1:-1])
        else:
            v_int = v[1:-1]
            d2 = 0.5 * self.sig_eff2 * v_int          # operator diffusion coeff (unscaled)
            d1 = self.kappa * (self.theta - v_int)    # operator convection coeff
            wm2, w02, wp2 = self._vv
            wm1, w01, wp1 = self._v1
            sub_op = d2 * wm2 + d1 * wm1
            diag_op = d2 * w02 + d1 * w01             # w02 < 0 (fd2 center weight)
            sup_op = d2 * wp2 + d1 * wp1
            a[1:-1] = -theta_loc * dt_step * sub_op
            b[1:-1] = 1.0 - theta_loc * dt_step * diag_op   # MINUS: I - theta*dt*L
            c[1:-1] = -theta_loc * dt_step * sup_op
        dV0 = float(self.V_grid[1] - self.V_grid[0])
        if self._degenerate_v0:
            # degenerate v=0 PDE row: (I - theta*dt * kappa*theta*U_v) with 2-point forward.
            conv = self.kappa * self.theta / dV0
            b[0] = 1.0 + theta_loc * dt_step * conv
            c[0] = -theta_loc * dt_step * conv
        else:
            b[0] = 1.0; c[0] = -1.0
        a[-1] = -1.0; b[-1] = 1.0
        self._V_tri_cache[key] = (a, b, c)
        return a, b, c

    def _ensure_S_lus(self, dt_step, theta_loc, t_mid):
        key = (float(dt_step), float(theta_loc), self._forward_step_index(t_mid))
        if key in self._S_lu_cache:
            return self._S_lu_cache[key]
        alpha, beta, gamma = self._tri_S(dt_step, theta_loc, t_mid)
        lus = []
        for j in range(self.N_V):
            A = sp.diags([alpha[j, 1:], beta[j], gamma[j, :-1]], offsets=[-1, 0, 1], format="csc")
            lus.append(spla.splu(A))
        self._S_lu_cache[key] = lus
        return lus

    def _ensure_V_lu(self, dt_step, theta_loc):
        key = (float(dt_step), float(theta_loc))
        if key in self._V_lu_cache:
            return self._V_lu_cache[key]
        alpha, beta, gamma = self._tri_V(dt_step, theta_loc)
        A = sp.diags([alpha[1:], beta, gamma[:-1]], offsets=[-1, 0, 1], format="csc")
        lu = spla.splu(A)
        self._V_lu_cache[key] = lu
        return lu

    # ---- direction solves ----
    def _solve_S(self, source, A1U, dt_step, theta_loc, tau, t_mid):
        if self.use_sparse:
            Y = np.empty_like(source)
            lus = self._ensure_S_lus(dt_step, theta_loc, t_mid)
            for j in range(self.N_V):
                rhs = self._s_boundary_rhs(
                    source[:, j] - theta_loc * dt_step * A1U[:, j], tau, v_index=j
                )
                Y[:, j] = lus[j].solve(rhs)
            self._bc(Y, tau)
            return Y
        a, b, c = self._tri_S(dt_step, theta_loc, t_mid)
        rhs = source - theta_loc * dt_step * A1U                    # (N_S, N_V), fresh array
        df = self.df_to_maturity(tau)
        carry_df = self.carry_df_to_maturity(tau)
        if self._opt_is_call:
            rhs[0, :] = 0.0
            rhs[-1, :] = max(0.0, self.S_max * carry_df - self.K * df)
        else:
            rhs[0, :] = self.K * df
            rhs[-1, :] = 0.0
        if self._barrier_active:
            if self._barrier_is_up:
                rhs[-1, :] = self._ko_bnd(tau)
            else:
                rhs[0, :] = self._ko_bnd(tau)
        custom = self._custom_s_boundary_values(tau)
        if custom is not None:
            low, high = custom
            rhs[0, :] = low
            rhs[-1, :] = high
        Y = solve_tridiag_batch(a, b, c, rhs.T).T                   # one system per V-slice
        self._bc(Y, tau)
        return Y

    def _solve_V(self, source, A2U, dt_step, theta_loc, tau):
        if self.use_sparse:
            U_out = np.empty_like(source)
            lu = self._ensure_V_lu(dt_step, theta_loc)
            for i in range(self.N_S):
                rhs = source[i, :] - theta_loc * dt_step * A2U[i, :]
                if not self._degenerate_v0:
                    rhs[0] = 0.0
                rhs[-1] = 0.0
                U_out[i, :] = lu.solve(rhs)
            self._bc(U_out, tau)
            return U_out
        a, b, c = self._tri_V(dt_step, theta_loc)
        rhs = source - theta_loc * dt_step * A2U                    # (N_S, N_V), fresh array
        if not self._degenerate_v0:
            rhs[:, 0] = 0.0
        rhs[:, -1] = 0.0
        U_out = solve_tridiag_batch(np.broadcast_to(a, (self.N_S, self.N_V)),
                                    np.broadcast_to(b, (self.N_S, self.N_V)),
                                    np.broadcast_to(c, (self.N_S, self.N_V)),
                                    rhs)                             # one system per S-row
        self._bc(U_out, tau)
        return U_out

    # ---- ADI steps ----
    def _douglas_step(self, U, dt_step, tau, theta_loc, t_mid):
        A1U, A2U = self._A1(U, t_mid), self._A2(U)
        A0U = self._A0(U, t_mid)
        # WS-C1: -rU is inside A1U now, so the predictor drops the explicit - r*U term.
        Y0 = U + dt_step * (A1U + A2U + A0U)
        Y1 = self._solve_S(Y0, A1U, dt_step, theta_loc, tau, t_mid)
        return self._solve_V(Y1, A2U, dt_step, theta_loc, tau)

    def _cs_step(self, U, dt_step, tau, theta_loc, t_mid):
        A1U, A2U = self._A1(U, t_mid), self._A2(U)
        A0U = self._A0(U, t_mid)
        Y0 = U + dt_step * (A1U + A2U + A0U)
        Y1 = self._solve_S(Y0, A1U, dt_step, theta_loc, tau, t_mid)
        Y2 = self._solve_V(Y1, A2U, dt_step, theta_loc, tau)
        # Craig-Sneyd corrector: restart from the explicit predictor Y0 (NOT Y2 — restarting
        # from the already-swept Y2 re-applies the implicit operators and degrades CS to a
        # first-order double-Douglas). With Y0, CS is genuinely second-order.
        if abs(self.rho) > 1e-12:
            Ycorr = Y0 + 0.5 * dt_step * (self._A0(Y2, t_mid) - A0U)
            self._bc(Ycorr, tau)
        else:
            Ycorr = Y0
        Z1 = self._solve_S(Ycorr, A1U, dt_step, theta_loc, tau, t_mid)
        return self._solve_V(Z1, A2U, dt_step, theta_loc, tau)

    def solve(
        self,
        is_call,
        scheme,
        theta,
        rannacher,
        step_hook=None,
        terminal_override=None,
        boundary_hook=None,
        damped_step_keys=None,
        damped_step_theta=1.0,
    ):
        """Backward ADI solve.

        step_hook(U, tau) -> U (optional): applied to the value surface after the terminal
        condition and after every time step, with tau = time-to-maturity of the new node.
        Used to inject a barrier knock-out condition without duplicating the ADI loop.
        terminal_override (optional ndarray, shape (n_x, n_v)): replaces the vanilla terminal
        payoff (e.g. a constant 1 for a survival / no-touch leg).
        boundary_hook(U, tau) -> U (optional): overrides boundary rows for non-vanilla
        terminal-value problems while keeping the ADI core reusable.
        damped_step_keys (optional set of int): integer step indices
        (round(tau/dt) of the step's landing node) that run as damped
        Douglas steps at theta = damped_step_theta (the wrappers pass
        params.event_theta; 1.0 = fully implicit) — a per-event Rannacher
        restart damping the modes a discrete event jump excites
        (Pooley-Vetzal-Forsyth; Giles-Carter). Craig-Sneyd also restarts as
        damped Douglas on these steps.
        """
        previous_hook = self._boundary_hook
        self._boundary_hook = boundary_hook
        try:
            if terminal_override is None:
                U = self._terminal(is_call)
            else:
                U = np.array(terminal_override, dtype=float)
                if boundary_hook is not None:
                    self._bc(U, 0.0)
            tau = 0.0
            if step_hook is not None:
                U = step_hook(U, tau)
            if rannacher and self.N_T >= 1:
                dt_half = 0.5 * self.dt
                steps_remaining = self.N_T - 1
                for half in range(2):
                    tau = self._advance_tau(
                        tau, dt_half, final=(steps_remaining == 0 and half == 1)
                    )
                    U = self._douglas_step(U, dt_half, tau, 1.0, self.T - tau + 0.5 * dt_half)
                    if step_hook is not None:
                        U = step_hook(U, tau)
            else:
                steps_remaining = self.N_T
            for step in range(steps_remaining):
                tau = self._advance_tau(
                    tau, self.dt, final=(step == steps_remaining - 1)
                )
                t_mid = self.T - tau + 0.5 * self.dt
                if (
                    damped_step_keys is not None
                    and int(round(tau / self.dt)) in damped_step_keys
                ):
                    # Per-event Rannacher restart: damped Douglas step at
                    # the caller's event theta (params.event_theta).
                    U = self._douglas_step(
                        U, self.dt, tau, float(damped_step_theta), t_mid
                    )
                elif scheme == ADIScheme.DOUGLAS:
                    U = self._douglas_step(U, self.dt, tau, theta, t_mid)
                else:  # CRAIG_SNEYD (MCS is rejected by the wrappers)
                    U = self._cs_step(U, self.dt, tau, theta, t_mid)
                if step_hook is not None:
                    U = step_hook(U, tau)
            return U
        finally:
            self._boundary_hook = previous_hook

    # ---- read-off ----
    def interpolate(self, U, x_val, v_val):
        X, V = self.X_grid, self.V_grid
        x_val = max(min(x_val, X[-1]), X[0])
        v_val = max(min(v_val, V[-1]), V[0])
        i0 = max(min(int(np.searchsorted(X, x_val, side="right") - 1), len(X) - 2), 0)
        j0 = max(min(int(np.searchsorted(V, v_val, side="right") - 1), len(V) - 2), 0)
        i1, j1 = i0 + 1, j0 + 1
        X0, X1, V0, V1 = X[i0], X[i1], V[j0], V[j1]
        if X1 == X0 or V1 == V0:
            return float(U[i0, j0])
        t = (x_val - X0) / (X1 - X0)
        u = (v_val - V0) / (V1 - V0)
        return float(
            (1 - t) * (1 - u) * U[i0, j0] + t * (1 - u) * U[i1, j0]
            + (1 - t) * u * U[i0, j1] + t * u * U[i1, j1]
        )

    def price_delta_gamma(self, U, s0):
        """Price and SPOT delta/gamma from the solved surface.

        Gamma comes from spatial derivatives of the PDE solution on the S-grid (not from
        re-bumping bilinearly-interpolated prices, which has meaningless curvature).
        """
        V = self.V_grid
        v_val = min(max(self.v0, V[0]), V[-1])
        j0 = max(min(int(np.searchsorted(V, v_val, side="right") - 1), len(V) - 2), 0)
        j1 = j0 + 1
        w = 0.0 if V[j1] == V[j0] else (v_val - V[j0]) / (V[j1] - V[j0])
        v_curve = (1.0 - w) * U[:, j0] + w * U[:, j1]   # price vs S at v0
        S = self.S_grid
        price = float(np.interp(s0, S, v_curve))
        dVdS = np.gradient(v_curve, S, edge_order=2)
        d2VdS2 = np.gradient(dVdS, S, edge_order=2)
        delta = float(np.interp(s0, S, dVdS))
        gamma = float(np.interp(s0, S, d2VdS2))
        return price, delta, gamma
