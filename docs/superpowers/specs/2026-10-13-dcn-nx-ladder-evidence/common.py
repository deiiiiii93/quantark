"""Shared fixtures for the T14 DCN n_x ladder evidence (2026-10-13).

Contract and model are d04's (docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/d04_dcn_refinement_ladder.py):
DCN_A (test/dcn_fixtures.py), flat_env(**FLAT), Heston v0 .0249 kappa 9.92 theta .045 sigma 1.583 rho -.025.

``ProbeHestonDCN`` is a runtime probe, NOT library code: its ``_price_contingent`` is a verbatim copy of
``HestonDCNPDESolver._price_contingent`` (worktree HEAD 7577d5cf) with the two event applications routed through
``self._events`` / ``self._ki_only`` so the event treatment can be switched:

- ``"nodal"``         production: raw ``<=`` / ``>=`` masks on S = exp(X) (``apply_dcn_events`` and the inline
                      daily ``np.where``). Checked bit-for-bit against the production solver in e01.
- ``"inclusive"``     nodal masks that also own nodes within ``Tolerance.RELATIVE`` of each barrier (the 1-ULP
                      exp(log(b)) round-trip noise no longer decides ownership; the product's comparisons are
                      inclusive).
- ``"cell_average"``  the dual-cell projection the 1D DCN route uses by default since 0.10.0
                      (``core/adapters/dcn.py: DCNEvents.apply``), applied to the (n_x, n_v) surfaces column-wise.

Run from the worktree root:
    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python <script>
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn  # noqa: E402
from quantark.asset.equity.engine.dcn_valuation import observe_valuation_row  # noqa: E402
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEResult, apply_dcn_events  # noqa: E402
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver  # noqa: E402
from quantark.asset.equity.engine.pde.grid.events import (  # noqa: E402
    project_event_values, project_piecewise_event)
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context  # noqa: E402
from quantark.priceenv import TermMarketContext  # noqa: E402
from quantark.priceenv.term_sampling import make_df_fn  # noqa: E402
from quantark.util.numerical import Tolerance  # noqa: E402
from quantark.volmodels import adi_core  # noqa: E402
from quantark.volmodels.adi_core import HestonSLVADICore  # noqa: E402
from quantark.volmodels.heston import HestonParams  # noqa: E402

FELLER = HestonParams(v0=0.0249, kappa=9.92, theta=0.045, sigma=1.583, rho=-0.025)
TIME_NODES = 486                      # DCN_A daily grid (ctx.times.size)
RSS_CAP_GB = 6.0


def product():
    return make_dcn(DCN_A)


def env():
    return flat_env(**FLAT)


def rss_estimate_gb(n_x, n_v, sub=1):
    """The task's budget rule: 16 B x n_x x n_v x time nodes x 1.5."""
    nodes = (TIME_NODES - 1) * sub + 1
    return 16.0 * n_x * n_v * nodes * 1.5 / 1e9


def core_for(n_x, n_v, prod=None, en=None, pin=True):
    """The production core's grids for (n_x, n_v) (n_t = 1: the X / V grids do not depend on the time grid)."""
    prod = prod or product()
    en = en or env()
    T = float(build_dcn_grid_context(prod).times[-1])
    saved = adi_core._pin_grid_targets
    if not pin:
        adi_core._pin_grid_targets = lambda grid, targets: grid
    try:
        return HestonSLVADICore(
            float(en.spot), float(prod.k_loss), T, float(en.get_rate(T)), float(en.get_div_yield(T)), FELLER,
            n_x, n_v, 1, market_context=None, leverage=None, eta=1.0, grid_style="concentrated",
            v0_boundary="degenerate_pde", v_grid_power=2.5, barrier_concentrate=float(prod.ki_barrier),
            critical_spots=[float(prod.coupon_barrier), float(prod.ko_barrier), float(prod.ki_barrier),
                            float(prod.k_loss)])
    finally:
        adi_core._pin_grid_targets = saved


def barriers(prod=None):
    prod = prod or product()
    return {"KI": float(prod.ki_barrier), "CPN": float(prod.coupon_barrier), "KO": float(prod.ko_barrier),
            "K_loss": float(prod.k_loss)}


class ProbeHestonDCN(HestonDCNPDESolver):
    """HestonDCNPDESolver with a switchable event treatment (see the module docstring)."""

    def __init__(self, *args, events="nodal", pin=True, **kw):
        super().__init__(*args, **kw)
        if events not in ("nodal", "inclusive", "cell_average"):
            raise ValueError(events)
        self.events = events
        self.pin = pin                    # False: the plain sinh grid (no node moved onto the levels)

    def price_detailed(self, product, pricing_env, *, lifecycle_state=None):
        saved = adi_core._pin_grid_targets
        if not self.pin:
            adi_core._pin_grid_targets = lambda grid, targets: grid
        try:
            return super().price_detailed(product, pricing_env, lifecycle_state=lifecycle_state)
        finally:
            adi_core._pin_grid_targets = saved

    # -- event treatments -------------------------------------------------------------------------------------
    def _masks(self, s_grid, product):
        ki = s_grid <= product.ki_barrier
        ko = s_grid >= product.ko_barrier
        cp = s_grid >= product.coupon_barrier
        if self.events == "inclusive":
            def near(b):
                return np.isclose(s_grid, b, rtol=Tolerance.RELATIVE, atol=0.0)
            ki, ko, cp = ki | near(product.ki_barrier), ko | near(product.ko_barrier), cp | near(product.coupon_barrier)
        return ki[:, None], ko[:, None], cp[:, None]

    def _events(self, v0, v1, cpn, ko, x_grid, masks, product):
        if self.events != "cell_average":
            ki_mask, ko_mask, cpn_mask = masks
            return apply_dcn_events(v0, v1, cpn_mask, ko_mask, ki_mask, cpn, ko)
        # verbatim DCNEvents.apply (core/adapters/dcn.py), barriers in log space
        if ko is not None:
            v1 = project_event_values(x_grid, math.log(product.ko_barrier), v1, np.full_like(v1, ko), True)
        thresholds = sorted({product.ki_barrier} | ({product.coupon_barrier} if cpn is not None else set())
                            | ({product.ko_barrier} if ko is not None else set()))
        branches = []
        for r in range(len(thresholds) + 1):
            lo = 0.0 if r == 0 else thresholds[r - 1]
            if r == 0:
                branches.append(v1)
            elif ko is not None and lo >= product.ko_barrier:
                branches.append(np.full_like(v0, ko))
            elif cpn is not None and lo >= product.coupon_barrier:
                branches.append(v0 + cpn)
            else:
                branches.append(v0)
        return project_piecewise_event(x_grid, [math.log(t) for t in thresholds], branches), v1

    def _ki_only(self, v0, v1, x_grid, masks, product):
        if self.events != "cell_average":
            return np.where(masks[0], v1, v0)
        return self._events(v0, v1, None, None, x_grid, masks, product)[0]

    # -- verbatim copy of HestonDCNPDESolver._price_contingent with the two event calls switched ------------------
    def _price_contingent(self, product, pricing_env) -> DCNPDEResult:
        ctx = build_dcn_grid_context(product)
        df = make_df_fn(pricing_env)
        row = observe_valuation_row(product, pricing_env, ctx, df)
        if row is not None and row.ends_contract:
            return DCNPDEResult(pv=row.pv, direction_sign=product.direction_sign, num_space_nodes=0,
                                num_time_steps=0)
        times = np.asarray(ctx.times, dtype=float)
        t_fine, day_col = self._fine_grid(times)
        T = float(times[-1])
        market = TermMarketContext.from_env(pricing_env, t_fine, ref_strike=product.initial_price)
        core = HestonSLVADICore(
            float(pricing_env.spot), float(product.k_loss), T, float(pricing_env.get_rate(T)),
            float(pricing_env.get_div_yield(T)), self.model_params, self.n_x, self.n_v, t_fine.size - 1,
            market_context=market, leverage=None, eta=1.0, grid_style="concentrated",
            v0_boundary=self.v0_boundary, v_grid_power=self.v_grid_power,
            barrier_concentrate=float(product.ki_barrier),
            critical_spots=[float(product.coupon_barrier), float(product.ko_barrier), float(product.ki_barrier),
                            float(product.k_loss)],
        )
        core._boundary_hook = None
        s_grid = core.S_grid
        x_grid = core.X_grid
        s0 = product.initial_price
        notional, part = product.notional, product.participation
        obs_at_col = {int(c): j for j, c in enumerate(ctx.obs_cols)}
        masks = self._masks(s_grid, product)

        def event_amounts(j, t_obs):
            cpn = None
            if ctx.obs_is_coupon[j]:
                cpn = (part * product.coupon_rate * ctx.coupon_accruals[j] * notional
                       * (df(ctx.coupon_pay_times[j]) / df(t_obs)))
            ko = None
            if ctx.obs_is_ko[j]:
                ko = (part * product.ko_coupon_rate * ctx.ko_accruals[j] * notional
                      * (df(ctx.ko_pay_times[j]) / df(t_obs)))
            return cpn, ko

        loss_bound = -(notional / s0) * part * float(product.k_loss)

        def v1_boundary_hook(U, tau):
            t = min(max(T - float(tau), 0.0), T)
            d_settle = df(ctx.loss_pay_time) / df(t)
            U[0, :] = loss_bound * d_settle
            U[-1, :] = 0.0
            return U

        def v0_boundary_hook(U, tau):
            t = min(max(T - float(tau), 0.0), T)
            d_settle = df(ctx.loss_pay_time) / df(t)
            U[0, :] = loss_bound * d_settle
            U[-1, :] = 0.0
            return U

        d_settle = df(ctx.loss_pay_time) / df(T)
        v1 = np.repeat((-(notional / s0) * part * np.maximum(product.k_loss - s_grid, 0.0) * d_settle)[:, None],
                       self.n_v, axis=1)
        v0 = np.zeros_like(v1)
        cpn_amt, ko_amt = event_amounts(len(ctx.obs_cols) - 1, T)
        v0, v1 = self._events(v0, v1, cpn_amt, ko_amt, x_grid, masks, product)

        rann = self.rannacher_steps
        for i in range(t_fine.size - 2, -1, -1):
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
                        U = self._step_surface(core, U, dt / 2.0, tau_new, t_mid, True)
                else:
                    U = self._step_surface(core, U, dt, T - t_lo, t_lo + dt / 2.0, False)
                surfaces.append(U)
            core._boundary_hook = None
            v1, v0 = surfaces
            if rann > 0:
                rann -= 1
            core._S_tri_cache.clear()
            k = int(day_col[i])
            if k > 0:
                j = obs_at_col.get(k)
                if j is not None:
                    cpn_amt, ko_amt = event_amounts(j, t_lo)
                    v0, v1 = self._events(v0, v1, cpn_amt, ko_amt, x_grid, masks, product)
                    rann = self.rannacher_steps
                else:
                    v0 = self._ki_only(v0, v1, x_grid, masks, product)

        observed_ki_at_valuation = bool(product.knocked_in_at_valuation) or (
            ctx.valuation_monitored and float(pricing_env.spot) <= product.ki_barrier)
        surface = v1 if observed_ki_at_valuation else v0
        pv_unsigned = core.interpolate(surface, float(np.log(float(pricing_env.spot))), float(self.model_params.v0))
        pv = product.direction_sign * pv_unsigned
        if row is not None and row.coupon_paid:
            pv = pv + product.direction_sign * row.coupon_pv
        return DCNPDEResult(pv=pv, direction_sign=product.direction_sign, num_space_nodes=self.n_x * self.n_v,
                            num_time_steps=t_fine.size - 1)


def price(n_x, n_v, sub=1, events="production", params=FELLER, prod=None, en=None, pin=True, **solver_kw):
    prod = prod or product()
    en = en or env()
    if events == "production":
        return HestonDCNPDESolver(params, n_x=n_x, n_v=n_v, substeps_per_interval=sub, **solver_kw).price(prod, en)
    return ProbeHestonDCN(params, n_x=n_x, n_v=n_v, substeps_per_interval=sub, events=events, pin=pin,
                          **solver_kw).price(prod, en)


def show_ladder(label, rows, out=print):
    """rows: [(n_x, value)]; prints value, successive difference d and observed order p."""
    out(f"  {label}")
    prev = prev_d = None
    for n_x, v in rows:
        d = None if prev is None else v - prev
        p = (math.log2(abs(prev_d) / abs(d)) if prev_d not in (None, 0.0) and d not in (None, 0.0) else None)
        out(f"    n_x {n_x:>5}  {v:14.4f}  {'' if d is None else f'd {d:+11.4f}'}  "
            f"{'' if p is None else f'p {p:6.2f}'}")
        prev_d, prev = d, v
