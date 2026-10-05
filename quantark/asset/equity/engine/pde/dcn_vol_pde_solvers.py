"""Surface-aware DCN PDE engines: Dupire local vol (1-D) and Heston (2-D).

Both are deterministic cross-checks of the surface-aware DCN MC engines,
completing the MC<->PDE triangle that Q1/Q2 already have for flat/curve
models:

- ``LocalVolDCNPDEEngine`` reuses the two-surface Crank-Nicolson DCN solver
  with a state-dependent sigma(S, t) sampled from a Dupire
  :class:`LocalVolSurface` per backward step.
- ``HestonDCNPDESolver`` evolves the two DCN value surfaces (V1 knocked-in,
  V0 never-knocked-in) on the shared (log-spot, variance) ADI core used by
  the phoenix/snowball Heston solvers, stepping the *exact* SSE daily DCN
  grid (no event-date snapping) with Rannacher restarts after every event
  date. Under a Feller-violating calibration the variance density piles up
  at v = 0, so the default variance boundary here is the degenerate-PDE
  row (in 't Hout-Foulon), not Neumann.

Backward event order at each observation date matches ``dcn_pde_solver``:
coupon injection -> KO overwrite -> KI projection; every daily grid node is
a discrete KI monitoring date, consistent with the MC engines.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from dataclasses import replace

from scipy.linalg import solve_banded

from quantark.asset.equity.engine.pde.core.components import resolve_stepper
from quantark.asset.equity.engine.pde.dcn_pde_solver import (
    DCNPDEResult,
    apply_dcn_events,
)
from quantark.asset.equity.engine.pde.grid import GridConfig, GridRequest
from quantark.asset.equity.engine.pde.grid.binder import GridLayerMixin
from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS, DCN_ACCURACY_PROFILES
from quantark.asset.equity.param.pde_removed_names import (
    REMOVED_DCN_NAMES,
    reject_removed_kwargs,
)
from quantark.priceenv.term_sampling import TermCoefficients
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.param import GridVolSurface
from quantark.priceenv import TermMarketContext
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.enum.engine_enums import ADIScheme, EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol


# --- transitional (spec 3C Task 19 -> Task 20): today's kernel DCN engine, LocalVolDCNPDEEngine's base ---

def _operator_coeffs(x: np.ndarray, r: float, q: float, sig):
    """Non-uniform central-difference coefficients for the log-spot
    generator ``L V = 0.5 sig^2 V_xx + (r - q - 0.5 sig^2) V_x - r V``.

    ``sig`` may be a scalar (flat/term vol) or an array over the full grid
    (state-dependent local vol); array input is sampled at interior nodes.
    """
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    sig = np.asarray(sig, dtype=float)
    sig_int = sig[1:-1] if sig.ndim else sig
    mu = r - q - 0.5 * sig_int * sig_int
    s2 = 0.5 * sig_int * sig_int
    lo = (2.0 * s2 - mu * dxp) / (dxm * (dxm + dxp))
    hi = (2.0 * s2 + mu * dxm) / (dxp * (dxm + dxp))
    mid = -(lo + hi) - r
    return lo, mid, hi


def _theta_step(
    v: np.ndarray,
    lo: np.ndarray,
    mid: np.ndarray,
    hi: np.ndarray,
    dt: float,
    theta: float,
) -> np.ndarray:
    """One theta-scheme step of ``V_t + L V = 0`` backward in time: solve
    ``(I - theta dt L) v_new = (I + (1-theta) dt L) v``. Boundary rows are
    identity during the solve (previous boundary values), then refreshed by
    one-sided linear extrapolation in x (V_xx = 0 at the far boundaries)."""
    n = v.size
    rhs = v.copy()
    interior = slice(1, n - 1)
    lv = lo * v[:-2] + mid * v[1:-1] + hi * v[2:]
    rhs[interior] = v[interior] + (1.0 - theta) * dt * lv
    ab = np.zeros((3, n))
    ab[1, 0] = ab[1, -1] = 1.0
    ab[1, interior] = 1.0 - theta * dt * mid
    ab[0, 2:] = -theta * dt * hi
    ab[2, :-2] = -theta * dt * lo
    out = solve_banded((1, 1), ab, rhs)
    out[0] = 2.0 * out[1] - out[2]
    out[-1] = 2.0 * out[-2] - out[-3]
    return out



class _KernelDCNPDEEngine(GridLayerMixin, BaseEngine):
    """Two-surface Crank-Nicolson (Rannacher-restarted) DCN engine - the pre-3C kernel, kept verbatim as
    ``LocalVolDCNPDEEngine``'s base until spec 3C Task 20 puts that engine on the core (``DCNPDEEngine`` is on the
    core from Task 19).

    Configured by ``PDEParams`` through the grid layer (spec §6.2) over the
    DCN route table: one fill step per observation interval (the schedule is
    the time resolution; an explicit time basis is rejected), ``accuracy``
    scales spatial resolution only (401 / 801 / 1601), and the damping counts
    apply in DCN's half-step form. Standard reproduces the pre-0.6 plan. The
    certified stepping loop and the nodal event semantics validated against
    the DCN MC engine are unchanged.
    """

    engine_type = EngineType.PDE
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    _route = "DCN"

    def __init__(self, params: Optional[PDEParams] = None, **removed):
        reject_removed_kwargs(type(self).__name__, removed, REMOVED_DCN_NAMES)
        if params is not None and not isinstance(params, PDEParams):
            # 0.5's first positional argument was num_space_nodes.
            raise ValidationError(
                f"{type(self).__name__}(params=...) takes a PDEParams, got "
                f"{type(params).__name__} {params!r}; the positional node count "
                f"was removed in 0.6.0: {REMOVED_DCN_NAMES['num_space_nodes']}"
            )
        super().__init__(params if params is not None else PDEParams())
        self._grid_binder = None
        points = self._grid_layer_binder().config.points  # resolve + check eagerly
        if points < 201:
            raise ValidationError(f"DCN needs grid points >= 201, got {points}")

    def _route_profiles(self):
        return DCN_ACCURACY_PROFILES

    def _route_overlay(self, grid: Optional[GridConfig]) -> Optional[GridConfig]:
        # max_points tracks an explicit points (max(2000, points), spec §4.2),
        # so explicit densities (e.g. 2401 in the MC gate tests) resolve.
        if grid is not None and grid.points is not None and grid.max_points is None:
            return replace(grid, max_points=max(2000, int(grid.points)))
        return grid

    def grid_request(self, product, market, tau) -> GridRequest:
        # The DCN PDE horizon is the final observation date from the grid
        # context, not the product's nominal maturity; the passed tau is
        # superseded so external validation compares on equal footing.
        ctx = build_dcn_grid_context(product)
        tau = float(ctx.times[-1])
        interior = tuple(
            float(t) for t in ctx.times[1:-1] if 0.0 < float(t) < tau
        )
        criticals = [
            market.spot,
            float(product.ki_barrier),
            float(product.ko_barrier),
            float(product.coupon_barrier),
        ]
        return GridRequest(
            tau=tau,
            bound_anchors=(market.spot, float(product.initial_price)),
            critical_prices=tuple(p for p in criticals if p and p > 0),
            hard_lower=None,
            hard_upper=None,
            event_times=interior,
            day_count=market.bus_days_in_year,
        )

    def price(self, product, pricing_env) -> float:
        return self.price_detailed(product, pricing_env).pv

    def _step_vol(self, s_grid, t, tc, i):
        """Vol used on backward step ``[t_i, t_{i+1}]``: scalar term vol here;
        subclasses may return a per-node array (state-dependent local vol)."""
        return float(tc.step_vols[i])

    def price_detailed(self, product, pricing_env, layout=None) -> DCNPDEResult:
        if not isinstance(product, DCNOption):
            raise PricingError("DCNPDEEngine only supports DCNOption")
        ctx = build_dcn_grid_context(product)
        times = ctx.times

        market = self.market_snapshot(product, pricing_env)
        tau_h = float(times[-1])
        if layout is None:
            layout = self._bound_layout_for_solve(
                self.grid_request(product, market, tau_h), market
            )
        else:
            self._external_layout_check(product, pricing_env, layout)
        if not np.array_equal(layout.time.t, times):
            raise PricingError(
                "grid layer time nodes diverge from the DCN observation grid "
                "(daily cadence must reproduce ctx.times exactly)"
            )
        x = layout.spatial.x
        s_grid = layout.spatial.s

        df = make_df_fn(pricing_env)
        tc = TermCoefficients.from_env(
            pricing_env, times, ref_strike=product.initial_price
        )
        s0 = product.initial_price
        notional, part = product.notional, product.participation

        obs_at_col = {int(c): j for j, c in enumerate(ctx.obs_cols)}
        ki_mask = s_grid <= product.ki_barrier
        ko_mask = s_grid >= product.ko_barrier
        cpn_mask = s_grid >= product.coupon_barrier

        def event_amounts(j: int, t_obs: float):
            cpn = None
            if ctx.obs_is_coupon[j]:
                cpn = (
                    part * product.coupon_rate * ctx.coupon_accruals[j]
                    * notional * (df(ctx.coupon_pay_times[j]) / df(t_obs))
                )
            ko = None
            if ctx.obs_is_ko[j]:
                ko = (
                    part * product.ko_coupon_rate * ctx.ko_accruals[j]
                    * notional * (df(ctx.ko_pay_times[j]) / df(t_obs))
                )
            return cpn, ko

        # terminal (t = last grid node == final observation date)
        t_mat = float(times[-1])
        d_settle = df(ctx.loss_pay_time) / df(t_mat)
        v1 = (
            -(notional / s0) * part * np.maximum(product.k_loss - s_grid, 0.0)
            * d_settle
        )
        v0 = np.zeros_like(v1)
        cpn_amt, ko_amt = event_amounts(len(ctx.obs_cols) - 1, t_mat)
        v0, v1 = apply_dcn_events(
            v0, v1, cpn_mask, ko_mask, ki_mask, cpn_amt, ko_amt
        )

        config = self._grid_layer_binder().config
        st = resolve_stepper(self.params, DAMPING_DEFAULTS["DCN"])
        theta, event_theta = float(st.theta), float(st.event_theta)
        rann = int(st.terminal_damping_steps)  # damp the terminal kink too
        rann_theta = 1.0  # terminal damping is always backward Euler
        for i in range(times.size - 2, -1, -1):  # step [t_i, t_{i+1}] backward
            dt = float(times[i + 1] - times[i])
            r_i = float(tc.fwd_rates[i])
            q_i = float(tc.fwd_carry[i])
            sig_i = self._step_vol(s_grid, float(times[i]), tc, i)
            lo, mid, hi = _operator_coeffs(x, r_i, q_i, sig_i)
            if rann > 0:
                v0 = _theta_step(v0, lo, mid, hi, dt / 2.0, rann_theta)
                v0 = _theta_step(v0, lo, mid, hi, dt / 2.0, rann_theta)
                v1 = _theta_step(v1, lo, mid, hi, dt / 2.0, rann_theta)
                v1 = _theta_step(v1, lo, mid, hi, dt / 2.0, rann_theta)
                rann -= 1
            else:
                v0 = _theta_step(v0, lo, mid, hi, dt, theta)
                v1 = _theta_step(v1, lo, mid, hi, dt, theta)
            if i > 0:
                t_i = float(times[i])
                j = obs_at_col.get(i)
                if j is not None:
                    cpn_amt, ko_amt = event_amounts(j, t_i)
                    v0, v1 = apply_dcn_events(
                        v0, v1, cpn_mask, ko_mask, ki_mask, cpn_amt, ko_amt
                    )
                    rann = int(st.event_damping_steps)
                    rann_theta = event_theta
                else:
                    # every daily node is a KI monitoring date (strictly
                    # after t=0): discrete KI projection, consistent with MC
                    v0 = np.where(ki_mask, v1, v0)

        observed_ki_at_valuation = (
            bool(product.knocked_in_at_valuation)
            or float(pricing_env.spot) <= product.ki_barrier
        )
        surface = v1 if observed_ki_at_valuation else v0
        pv_unsigned = float(np.interp(np.log(float(pricing_env.spot)), x, surface))
        pv = product.direction_sign * pv_unsigned
        return DCNPDEResult(
            pv=pv,
            direction_sign=product.direction_sign,
            num_space_nodes=int(config.points),
            num_time_steps=times.size - 1,
        )


class LocalVolDCNPDEEngine(_KernelDCNPDEEngine):
    """Two-surface DCN PDE under Dupire local vol.

    Mirrors ``LocalVolDCNMCEngine``: pass a prebuilt
    :class:`LocalVolSurface`, or leave it ``None`` and the engine builds
    Dupire local vol from the environment's :class:`GridVolSurface`. The
    per-step operator vol is sigma(S_grid, t_i), the same forward-time
    sampling convention as the MC path generator.
    """

    def __init__(
        self,
        local_vol_surface: Optional[LocalVolSurface] = None,
        params: Optional[PDEParams] = None,
        **removed,
    ):
        super().__init__(params=params, **removed)
        if local_vol_surface is not None and not isinstance(
            local_vol_surface, LocalVolSurface
        ):
            raise ValidationError(
                "local_vol_surface must be a LocalVolSurface or None"
            )
        self._prebuilt = local_vol_surface
        self._active_lv: Optional[LocalVolSurface] = None

    def _build_surface(self, env) -> LocalVolSurface:
        if self._prebuilt is not None:
            return self._prebuilt
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError(
                "LocalVolDCNPDEEngine needs a GridVolSurface or a prebuilt "
                "LocalVolSurface"
            )
        return build_dupire_local_vol(
            env.vol_surface,
            spot=env.spot,
            rate_curve=env.rate_curve,
            div_yield=env.get_div_yield,
        )

    def _step_vol(self, s_grid, t, tc, i):
        vol = np.asarray(self._active_lv.local_vol(s_grid, t), dtype=float)
        return np.maximum(vol, 1e-8)

    def price_detailed(self, product, pricing_env) -> DCNPDEResult:
        self._active_lv = self._build_surface(pricing_env)
        try:
            return super().price_detailed(product, pricing_env)
        finally:
            self._active_lv = None


class HestonDCNPDESolver(BaseEngine):
    """Two-surface DCN PDE under Heston on a 2-D (log-spot, variance) grid.

    ADI (Craig-Sneyd by default) on the shared :class:`HestonSLVADICore`,
    driven step-by-step over the exact DCN daily grid rather than the
    core's uniform ``solve`` loop, so KI monitoring dates and monthly event
    dates land exactly on time nodes. ``substeps_per_interval`` refines
    time between daily nodes for discretization ladders; events and KI
    projection are applied only at the daily nodes.
    """

    engine_type = EngineType.PDE
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL

    def __init__(
        self,
        model_params: HestonParams,
        n_x: int = 301,
        n_v: int = 101,
        substeps_per_interval: int = 1,
        scheme: ADIScheme | str = ADIScheme.CRAIG_SNEYD,
        theta: float = 0.5,
        rannacher_steps: int = 2,
        v0_boundary: str = "degenerate_pde",
        v_grid_power: float = 2.5,
    ):
        super().__init__()
        if not isinstance(model_params, HestonParams):
            raise ValidationError("model_params must be a HestonParams")
        try:
            scheme = (
                ADIScheme[scheme.upper()] if isinstance(scheme, str) else scheme
            )
        except KeyError:
            raise ValidationError(f"unknown ADI scheme: {scheme}")
        if scheme not in (ADIScheme.DOUGLAS, ADIScheme.CRAIG_SNEYD):
            raise ValidationError(
                "HestonDCNPDESolver supports DOUGLAS and CRAIG_SNEYD only"
            )
        if int(n_x) < 51 or int(n_v) < 21:
            raise ValidationError("need n_x >= 51 and n_v >= 21")
        if (
            isinstance(substeps_per_interval, bool)
            or int(substeps_per_interval) < 1
        ):
            raise ValidationError(
                "substeps_per_interval must be a positive integer"
            )
        if v0_boundary not in ("neumann", "degenerate_pde"):
            raise ValidationError(
                "v0_boundary must be 'neumann' or 'degenerate_pde'"
            )
        self.model_params = model_params
        self.n_x = int(n_x)
        self.n_v = int(n_v)
        self.substeps = int(substeps_per_interval)
        self.scheme = scheme
        self.theta = float(theta)
        self.rannacher_steps = int(rannacher_steps)
        self.v0_boundary = str(v0_boundary)
        self.v_grid_power = float(v_grid_power)

    def price(self, product, pricing_env) -> float:
        return self.price_detailed(product, pricing_env).pv

    def _fine_grid(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Refine each daily interval into equal substeps; return the fine
        grid and, per fine node, the daily-grid column it coincides with
        (-1 for interior substep nodes)."""
        if self.substeps == 1:
            return times.astype(float), np.arange(times.size)
        pieces = [np.array([times[0]])]
        cols = [np.array([0])]
        for i in range(times.size - 1):
            seg = np.linspace(times[i], times[i + 1], self.substeps + 1)[1:]
            pieces.append(seg)
            seg_cols = np.full(self.substeps, -1, dtype=int)
            seg_cols[-1] = i + 1
            cols.append(seg_cols)
        return np.concatenate(pieces), np.concatenate(cols)

    def _step_surface(self, core, U, dt, tau_new, t_mid, implicit_euler):
        if implicit_euler:
            return core._douglas_step(U, dt, tau_new, 1.0, t_mid)
        if self.scheme == ADIScheme.DOUGLAS:
            return core._douglas_step(U, dt, tau_new, self.theta, t_mid)
        return core._cs_step(U, dt, tau_new, self.theta, t_mid)

    def price_detailed(self, product, pricing_env) -> DCNPDEResult:
        if not isinstance(product, DCNOption):
            raise PricingError("HestonDCNPDESolver only supports DCNOption")
        ctx = build_dcn_grid_context(product)
        times = np.asarray(ctx.times, dtype=float)
        t_fine, day_col = self._fine_grid(times)
        T = float(times[-1])
        df = make_df_fn(pricing_env)
        market = TermMarketContext.from_env(
            pricing_env, t_fine, ref_strike=product.initial_price
        )
        core = HestonSLVADICore(
            float(pricing_env.spot),
            float(product.k_loss),
            T,
            float(pricing_env.get_rate(T)),
            float(pricing_env.get_div_yield(T)),
            self.model_params,
            self.n_x,
            self.n_v,
            t_fine.size - 1,
            market_context=market,
            leverage=None,
            eta=1.0,
            grid_style="concentrated",
            v0_boundary=self.v0_boundary,
            v_grid_power=self.v_grid_power,
            barrier_concentrate=float(product.ki_barrier),
            critical_spots=[
                float(product.coupon_barrier),
                float(product.ko_barrier),
                float(product.ki_barrier),
                float(product.k_loss),
            ],
        )
        core._boundary_hook = None
        s_grid = core.S_grid
        s0 = product.initial_price
        notional, part = product.notional, product.participation

        obs_at_col = {int(c): j for j, c in enumerate(ctx.obs_cols)}
        ki_mask = (s_grid <= product.ki_barrier)[:, None]
        ko_mask = (s_grid >= product.ko_barrier)[:, None]
        cpn_mask = (s_grid >= product.coupon_barrier)[:, None]

        def event_amounts(j: int, t_obs: float):
            cpn = None
            if ctx.obs_is_coupon[j]:
                cpn = (
                    part * product.coupon_rate * ctx.coupon_accruals[j]
                    * notional * (df(ctx.coupon_pay_times[j]) / df(t_obs))
                )
            ko = None
            if ctx.obs_is_ko[j]:
                ko = (
                    part * product.ko_coupon_rate * ctx.ko_accruals[j]
                    * notional * (df(ctx.ko_pay_times[j]) / df(t_obs))
                )
            return cpn, ko

        loss_bound = -(notional / s0) * part * float(product.k_loss)

        def v1_boundary_hook(U, tau):
            t = min(max(T - float(tau), 0.0), T)
            d_settle = df(ctx.loss_pay_time) / df(t)
            # S_min is exp(-many sigma): the loss leg is effectively the
            # full strike; the far high-S knocked-in surface is worthless.
            U[0, :] = loss_bound * d_settle
            U[-1, :] = 0.0
            return U

        def v0_boundary_hook(U, tau):
            # Low boundary sits deep inside the KI region, where the daily
            # projection overwrites V0 with V1 anyway; keep them equal.
            t = min(max(T - float(tau), 0.0), T)
            d_settle = df(ctx.loss_pay_time) / df(t)
            U[0, :] = loss_bound * d_settle
            U[-1, :] = 0.0
            return U

        # terminal (t = T = final observation date)
        d_settle = df(ctx.loss_pay_time) / df(T)
        v1 = np.repeat(
            (
                -(notional / s0) * part
                * np.maximum(product.k_loss - s_grid, 0.0) * d_settle
            )[:, None],
            self.n_v,
            axis=1,
        )
        v0 = np.zeros_like(v1)
        cpn_amt, ko_amt = event_amounts(len(ctx.obs_cols) - 1, T)
        v0, v1 = apply_dcn_events(
            v0, v1, cpn_mask, ko_mask, ki_mask, cpn_amt, ko_amt
        )

        rann = self.rannacher_steps  # damp the terminal kink too
        for i in range(t_fine.size - 2, -1, -1):  # step [t_i, t_{i+1}]
            dt = float(t_fine[i + 1] - t_fine[i])
            t_lo = float(t_fine[i])
            surfaces = []
            for U, hook in ((v1, v1_boundary_hook), (v0, v0_boundary_hook)):
                core._boundary_hook = hook
                if rann > 0:
                    for half in (0, 1):
                        t_new = t_lo + (1 - half) * dt / 2.0
                        tau_new = T - t_new
                        t_mid = t_new + dt / 4.0
                        U = self._step_surface(
                            core, U, dt / 2.0, tau_new, t_mid, True
                        )
                else:
                    U = self._step_surface(
                        core, U, dt, T - t_lo, t_lo + dt / 2.0, False
                    )
                surfaces.append(U)
            core._boundary_hook = None
            v1, v0 = surfaces
            if rann > 0:
                rann -= 1
            core._S_tri_cache.clear()  # keys include the step index
            k = int(day_col[i])
            if k > 0:
                j = obs_at_col.get(k)
                if j is not None:
                    cpn_amt, ko_amt = event_amounts(j, t_lo)
                    v0, v1 = apply_dcn_events(
                        v0, v1, cpn_mask, ko_mask, ki_mask, cpn_amt, ko_amt
                    )
                    rann = self.rannacher_steps
                else:
                    # every daily node is a KI monitoring date
                    v0 = np.where(ki_mask, v1, v0)

        observed_ki_at_valuation = (
            bool(product.knocked_in_at_valuation)
            or float(pricing_env.spot) <= product.ki_barrier
        )
        surface = v1 if observed_ki_at_valuation else v0
        pv_unsigned = core.interpolate(
            surface,
            float(np.log(float(pricing_env.spot))),
            float(self.model_params.v0),
        )
        pv = product.direction_sign * pv_unsigned
        return DCNPDEResult(
            pv=pv,
            direction_sign=product.direction_sign,
            num_space_nodes=self.n_x * self.n_v,
            num_time_steps=t_fine.size - 1,
        )
