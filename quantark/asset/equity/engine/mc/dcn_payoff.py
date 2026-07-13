"""Vectorized DCN path-payoff kernel (spec WP1.2, forward order KI->KO->coupon).

Shared by DCNMCEngine and the LV/Heston DCN variants. Paths are spot levels
on the DCN daily trading grid (column 0 = valuation date). All returned leg
PVs are discounted to t=0 and UNSIGNED (BUYER perspective); the engine
applies direction_sign. Daily KI is exact discrete monitoring (the grid IS
the observation set) — no barrier continuity correction, per spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from quantark.asset.equity.product.option.dcn_grid import (  # noqa: F401
    DCNGridContext,
    build_dcn_grid_context,
)
from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class DCNPathCashflows:
    fixed_coupon_pv: np.ndarray            # (n_paths,)
    fixed_coupon_pv_by_period: np.ndarray  # (n_paths, n_obs)
    ko_pv: np.ndarray                      # (n_paths,)
    loss_pv: np.ndarray                    # (n_paths,)
    ko_obs_row: np.ndarray                 # (n_paths,) int, -1 = never KO
    knocked_in: np.ndarray                 # (n_paths,) bool
    coupon_paid: np.ndarray                # (n_paths, n_obs) bool

    @property
    def total_pv(self) -> np.ndarray:
        return self.fixed_coupon_pv + self.ko_pv + self.loss_pv


def compute_dcn_cashflows(
    paths: np.ndarray,
    product: DCNOption,
    ctx: DCNGridContext,
    df: Callable[[float], float],
) -> DCNPathCashflows:
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] != ctx.times.size:
        raise ValidationError(
            f"paths must be (n_paths, {ctx.times.size}), got {paths.shape}"
        )
    n_paths, n_obs = paths.shape[0], ctx.obs_cols.size
    part, notional, s0 = (
        product.participation, product.notional, product.initial_price
    )
    b_ki, b_ko, b_c = product.ki_barrier, product.ko_barrier, product.coupon_barrier

    dfs_c = np.array([df(t) for t in ctx.coupon_pay_times])
    dfs_k = np.array([df(t) for t in ctx.ko_pay_times])
    df_loss = float(df(ctx.loss_pay_time))

    alive = np.ones(n_paths, dtype=bool)
    knocked_in = np.full(n_paths, bool(product.knocked_in_at_valuation))
    ko_obs_row = np.full(n_paths, -1, dtype=int)
    coupon_paid = np.zeros((n_paths, n_obs), dtype=bool)
    coupon_pv_by = np.zeros((n_paths, n_obs))
    ko_pv = np.zeros(n_paths)

    # KI monitoring strictly after valuation: slices start at prev_col + 1
    prev_col = 0
    for j in range(n_obs):
        c = int(ctx.obs_cols[j])
        # 1) daily KI over (prev_obs, this obs], BEFORE monthly processing.
        #    Only ALIVE paths monitor: the contract terminates at KO, so a
        #    breach after a KO observation is not a contractual KI event
        #    (same-date KI-before-KO ordering is preserved because a path
        #    KO'ing at THIS observation is still alive during this update).
        seg = paths[:, prev_col + 1:c + 1]
        if seg.shape[1] > 0:
            knocked_in |= alive & (seg <= b_ki).any(axis=1)
        s_obs = paths[:, c]
        # 2) KO priority (alive paths, KO-eligible obs)
        if ctx.obs_is_ko[j]:
            ko_now = alive & (s_obs >= b_ko)
            ko_pv[ko_now] = (
                part * product.ko_coupon_rate * ctx.ko_accruals[j]
                * notional * dfs_k[j]
            )
            ko_obs_row[ko_now] = j
            alive &= ~ko_now
        # 3) fixed coupon (not KO'd today, not knocked in, coupon-eligible)
        if ctx.obs_is_coupon[j]:
            pay = alive & ~knocked_in & (s_obs >= b_c)
            coupon_paid[pay, j] = True
            coupon_pv_by[pay, j] = (
                part * product.coupon_rate * ctx.coupon_accruals[j]
                * notional * dfs_c[j]
            )
        prev_col = c

    # 4) maturity loss leg: survivors (never KO) that are knocked in
    s_t = paths[:, int(ctx.obs_cols[-1])]
    loss_pv = np.zeros(n_paths)
    hit = alive & knocked_in
    loss_pv[hit] = (
        -(notional / s0) * part * np.maximum(product.k_loss - s_t[hit], 0.0)
        * df_loss
    )
    return DCNPathCashflows(
        fixed_coupon_pv=coupon_pv_by.sum(axis=1),
        fixed_coupon_pv_by_period=coupon_pv_by,
        ko_pv=ko_pv,
        loss_pv=loss_pv,
        ko_obs_row=ko_obs_row,
        knocked_in=knocked_in,
        coupon_paid=coupon_paid,
    )
