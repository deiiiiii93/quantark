"""Surface-aware DCN PDE engines: Dupire local vol (1-D) and Heston (2-D).

Both are deterministic cross-checks of the surface-aware DCN MC engines,
completing the MC<->PDE triangle that Q1/Q2 already have for flat/curve
models:

- ``LocalVolDCNPDEEngine`` is the core DCN (spec 3C §5.4) with one Dupire
  local-vol row per step at the interval midpoint on the solve's S grid.
- ``HestonDCNPDESolver`` evolves the two DCN value surfaces (V1 knocked-in,
  V0 never-knocked-in) on the shared (log-spot, variance) ADI core used by
  the phoenix/snowball Heston solvers, stepping the *exact* SSE daily DCN
  grid (no event-date snapping) with Rannacher restarts after every event
  date. Under a Feller-violating calibration the variance density piles up
  at v = 0, so the default variance boundary here is the degenerate-PDE
  row (in 't Hout-Foulon), not Neumann.

Backward event order at each observation date matches ``dcn_pde_solver``:
coupon injection -> KO overwrite -> KI projection; every daily grid node is
a discrete KI monitoring date, consistent with the MC engines. The Heston
events are the 1-D route's: dual-cell projections on an unpinned grid by
default (T14, ``docs/superpowers/specs/2026-10-13-dcn-nx-ladder-evidence``),
the legacy whole-node masks on a level-pinned grid under
``event_projection="nodal"``.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.dcn_valuation import (
    observe_valuation_row,
    resolve_dcn_lifecycle_state,
)
from quantark.asset.equity.engine.pde.core.adapters.local_vol import LocalVolDCNAdapter
from quantark.asset.equity.engine.pde.dcn_pde_solver import (
    DCNPDEEngine,
    DCNPDEResult,
    apply_dcn_events,
)
from quantark.asset.equity.engine.pde.grid.events import (
    project_event_values,
    project_piecewise_event,
)
from quantark.asset.equity.engine.pde.local_vol_surface import LocalVolSurfaceMixin
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.priceenv import TermMarketContext
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.enum.engine_enums import ADIScheme, EngineType, EventProjectionMode
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface


class LocalVolDCNPDEEngine(LocalVolSurfaceMixin, DCNPDEEngine):
    """Two-surface DCN PDE under Dupire local vol on the core: one local-vol row per step at the interval midpoint
    on the solve's S grid (spec 3C §5.4). Pass a prebuilt LocalVolSurface, or None to build Dupire local vol from the
    environment's GridVolSurface."""

    _adapter_class = LocalVolDCNAdapter

    def __init__(self, local_vol_surface: Optional[LocalVolSurface] = None, params: Optional[PDEParams] = None,
                 **removed):
        super().__init__(params=params, **removed)
        if local_vol_surface is not None and not isinstance(local_vol_surface, LocalVolSurface):
            raise ValidationError("local_vol_surface must be a LocalVolSurface or None")
        self._init_local_vol(local_vol_surface)

    def price(self, product, pricing_env, *, lifecycle_state=None):
        return self._with_surface(pricing_env, lambda: DCNPDEEngine.price(
            self, product, pricing_env, lifecycle_state=lifecycle_state))

    def calculate_greeks(self, product, pricing_env, *, lifecycle_state=None):
        return self._with_surface(pricing_env, lambda: DCNPDEEngine.calculate_greeks(
            self, product, pricing_env, lifecycle_state=lifecycle_state))

    def price_detailed(self, product, pricing_env, *, lifecycle_state=None):
        return self._with_surface(pricing_env, lambda: DCNPDEEngine.price_detailed(
            self, product, pricing_env, lifecycle_state=lifecycle_state))

    def _session_outputs(self, product, pricing_env, **kwargs):
        return self._with_surface(pricing_env, lambda: DCNPDEEngine._session_outputs(self, product, pricing_env, **kwargs))

    def calculate_spot_greeks_curve(self, product, pricing_env, spot_levels):
        return self._with_surface(pricing_env, lambda: DCNPDEEngine.calculate_spot_greeks_curve(
            self, product, pricing_env, spot_levels))


class HestonDCNPDESolver(BaseEngine):
    """Two-surface DCN PDE under Heston on a 2-D (log-spot, variance) grid.

    ADI (Craig-Sneyd by default) on the shared :class:`HestonSLVADICore`,
    driven step-by-step over the exact DCN daily grid rather than the
    core's uniform ``solve`` loop, so KI monitoring dates and monthly event
    dates land exactly on time nodes. ``substeps_per_interval`` refines
    time between daily nodes for discretization ladders; events and KI
    projection are applied only at the daily nodes.

    ``event_projection``: ``None`` (default) or ``CELL_AVERAGE`` applies each
    date's coupon / KO / KI as the dual-cell projection of the 1-D DCN route
    (``core/adapters/dcn.py``), column by column, on the plain concentrated
    grid; ``"nodal"`` keeps the pre-T14 discretization bit for bit (whole-node
    masks, every level pinned to a node), whose trigger sits up to half a
    cell off each level and makes the price erratic in ``n_x``.
    """

    engine_type = EngineType.PDE
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True  # DCNLifecycleState (W14)

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
        event_projection: EventProjectionMode | str | None = None,
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
        if isinstance(event_projection, str):
            try:
                event_projection = EventProjectionMode(event_projection.lower())
            except ValueError:
                raise ValidationError(
                    f"unknown event_projection: {event_projection!r}"
                )
        if event_projection is not None and not isinstance(
            event_projection, EventProjectionMode
        ):
            raise ValidationError(
                "event_projection must be None, an EventProjectionMode or its "
                f"string value, got {type(event_projection).__name__}"
            )
        self.model_params = model_params
        self.event_projection = (
            EventProjectionMode.CELL_AVERAGE
            if event_projection is None
            else event_projection
        )
        self.n_x = int(n_x)
        self.n_v = int(n_v)
        self.substeps = int(substeps_per_interval)
        self.scheme = scheme
        self.theta = float(theta)
        self.rannacher_steps = int(rannacher_steps)
        self.v0_boundary = str(v0_boundary)
        self.v_grid_power = float(v_grid_power)

    def price(self, product, pricing_env, *, lifecycle_state=None) -> float:
        return self.price_detailed(product, pricing_env, lifecycle_state=lifecycle_state).pv

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

    @property
    def _nodal(self) -> bool:
        return self.event_projection is EventProjectionMode.NODAL

    def _apply_events(self, v0, v1, cpn, ko, x_grid, masks, product):
        """One date's events (backward order coupon, knock-out, knock-in);
        ``cpn`` / ``ko`` None: that event is not active (both None: KI only).
        Cell-average: ``DCNEvents.apply`` of the 1-D route, levels in log
        space, acting on axis 0 of the (n_x, n_v) surfaces."""
        if self._nodal:
            ki_mask, ko_mask, cpn_mask = masks
            return apply_dcn_events(v0, v1, cpn_mask, ko_mask, ki_mask, cpn, ko)
        if ko is not None:
            v1 = project_event_values(
                x_grid, math.log(product.ko_barrier), v1,
                np.full_like(v1, ko), True,
            )
        thresholds = sorted(
            {product.ki_barrier}
            | ({product.coupon_barrier} if cpn is not None else set())
            | ({product.ko_barrier} if ko is not None else set())
        )
        branches = []
        for r in range(len(thresholds) + 1):
            lo = 0.0 if r == 0 else thresholds[r - 1]
            if r == 0:
                branches.append(v1)                    # below the knock-in level
            elif ko is not None and lo >= product.ko_barrier:
                branches.append(np.full_like(v0, ko))  # knock-out over coupon
            elif cpn is not None and lo >= product.coupon_barrier:
                branches.append(v0 + cpn)
            else:
                branches.append(v0)
        v0 = project_piecewise_event(
            x_grid, [math.log(b) for b in thresholds], branches
        )
        return v0, v1

    def price_detailed(self, product, pricing_env, *, lifecycle_state=None) -> DCNPDEResult:
        """Price ``product``; a ``DCNLifecycleState`` adds its pending
        receivables, and a knocked-out / matured one prices them alone (W14)."""
        if not isinstance(product, DCNOption):
            raise PricingError("HestonDCNPDESolver only supports DCNOption")
        state = resolve_dcn_lifecycle_state(product, pricing_env, lifecycle_state)
        if state is not None and state.terminal:
            return DCNPDEResult(pv=state.pending_pv, direction_sign=product.direction_sign,
                                num_space_nodes=0, num_time_steps=0)
        result = self._price_contingent(product, pricing_env)
        if state is None:
            return result
        return dataclasses.replace(result, pv=result.pv + state.pending_pv)

    def _price_contingent(self, product, pricing_env) -> DCNPDEResult:
        ctx = build_dcn_grid_context(product)
        df = make_df_fn(pricing_env)
        # the monthly row observed today, pointwise at the spot: the final
        # observation date (W10) or a knock-out today (W10b) settle here
        row = observe_valuation_row(product, pricing_env, ctx, df)
        if row is not None and row.ends_contract:
            return DCNPDEResult(
                pv=row.pv,
                direction_sign=product.direction_sign,
                num_space_nodes=0,
                num_time_steps=0,
            )
        times = np.asarray(ctx.times, dtype=float)
        t_fine, day_col = self._fine_grid(times)
        T = float(times[-1])
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
            pin_critical_levels=self._nodal,
        )
        core._boundary_hook = None
        s_grid = core.S_grid
        x_grid = core.X_grid
        s0 = product.initial_price
        notional, part = product.notional, product.participation

        obs_at_col = {int(c): j for j, c in enumerate(ctx.obs_cols)}
        masks = (
            (s_grid <= product.ki_barrier)[:, None],
            (s_grid >= product.ko_barrier)[:, None],
            (s_grid >= product.coupon_barrier)[:, None],
        ) if self._nodal else None

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
        v0, v1 = self._apply_events(
            v0, v1, cpn_amt, ko_amt, x_grid, masks, product
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
                    v0, v1 = self._apply_events(
                        v0, v1, cpn_amt, ko_amt, x_grid, masks, product
                    )
                    rann = self.rannacher_steps
                elif self._nodal:
                    # every daily node is a KI monitoring date
                    v0 = np.where(masks[0], v1, v0)
                else:
                    v0 = self._apply_events(
                        v0, v1, None, None, x_grid, masks, product
                    )[0]

        # an unmonitored valuation instant (a closed day, or the post-close
        # view) is no knock-in test: only the carried flag reads V1
        observed_ki_at_valuation = bool(product.knocked_in_at_valuation) or (
            ctx.valuation_monitored
            and float(pricing_env.spot) <= product.ki_barrier
        )
        surface = v1 if observed_ki_at_valuation else v0
        pv_unsigned = core.interpolate(
            surface,
            float(np.log(float(pricing_env.spot))),
            float(self.model_params.v0),
        )
        pv = product.direction_sign * pv_unsigned
        if row is not None and row.coupon_paid:
            # W10b: a coupon today, added to the smooth continuation (the
            # march never applies the t = 0 row: k > 0 above)
            pv = pv + product.direction_sign * row.coupon_pv
        return DCNPDEResult(
            pv=pv,
            direction_sign=product.direction_sign,
            num_space_nodes=self.n_x * self.n_v,
            num_time_steps=t_fine.size - 1,
        )
