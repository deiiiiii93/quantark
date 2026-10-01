"""DCN two-surface PDE engine (spec WP1.4) — independent cross-check of MC.

V1 = knocked-in surface (no coupons ever; KO still live; terminal = loss leg
discounted from settlement_date). V0 = never-knocked-in surface (coupons +
KO; projected onto V1 below the KI barrier at every daily step). Backward
event order at each observation date: coupon injection -> KO overwrite ->
KI projection (the reverse of the MC forward event priority KI -> KO ->
coupon, so the two engines are path-equivalent).

Deterministic by construction: no Monte Carlo imports (the DCN grid context
lives in the neutral product-side module ``dcn_grid``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.linalg import solve_banded

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.pde.core.components import resolve_stepper
from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.pde.grid import GridConfig, GridRequest
from quantark.asset.equity.engine.pde.grid.binder import GridLayerMixin
from quantark.asset.equity.engine.pde.grid.config import DCN_ACCURACY_PROFILES
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.param.pde_removed_names import (
    REMOVED_DCN_NAMES,
    reject_removed_kwargs,
)
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.priceenv.term_sampling import TermCoefficients, make_df_fn
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError


@dataclass(frozen=True)
class DCNPDEResult:
    pv: float
    direction_sign: float
    num_space_nodes: int
    num_time_steps: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _sinh_grid(
    x_min: float, x_max: float, x_star: float, n: int, strength: float
) -> np.ndarray:
    """Tavella-Randall grid on [x_min, x_max] concentrated at x_star.

    ``strength`` is in log-space units: smaller values concentrate nodes
    more tightly around ``x_star``.
    """
    u = np.linspace(0.0, 1.0, n)
    a = float(strength)
    c1 = np.arcsinh((x_min - x_star) / a)
    c2 = np.arcsinh((x_max - x_star) / a)
    x = x_star + a * np.sinh(c1 + u * (c2 - c1))
    x[0], x[-1] = x_min, x_max
    return x


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


def apply_dcn_events(
    v0: np.ndarray,
    v1: np.ndarray,
    cpn_mask: np.ndarray,
    ko_mask: np.ndarray,
    ki_mask: np.ndarray,
    coupon_amount: Optional[float],
    ko_amount: Optional[float],
):
    """Backward event operator at one observation date (spec WP1.4 order):
    (1) coupon injection on V0, (2) KO overwrite on both surfaces, (3) KI
    projection ``V0 := V1`` below the KI barrier. ``None`` amount = that
    event type is not active at this date."""
    if coupon_amount is not None:
        v0 = np.where(cpn_mask, v0 + coupon_amount, v0)
    if ko_amount is not None:
        v0 = np.where(ko_mask, ko_amount, v0)
        v1 = np.where(ko_mask, ko_amount, v1)
    v0 = np.where(ki_mask, v1, v0)
    return v0, v1


class DCNPDEEngine(GridLayerMixin, BaseEngine):
    """Two-surface Crank-Nicolson (Rannacher-restarted) DCN engine.

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
