"""A DCN monthly observation at the valuation instant (W10 / W10b, user rulings 2026-10-10).

Spec: ``docs/superpowers/specs/2026-10-10-dcn-final-observation-proposal.md``. The row observed at the valuation
instant is decided on today's spot (the close, Q1), pointwise at barrier ties (Q2), in the MC path kernel's order
(``engine/mc/dcn_payoff.py``): knock-in = carried OR spot <= KI; knock-out (spot >= KO, in either knock-in state) pays
the knock-out coupon and ends the contract; the fixed coupon pays when not knocked out, not knocked in and spot >= the
coupon barrier; on the FINAL row the loss leg pays when knocked in and not knocked out. Every flow is discounted from
the valuation date to its own payment date (DF = 1 when paid today). Leg amounts are the kernel's arithmetic, so the MC
route equals the kernel's one-column row bit for bit.

Neutral module: no Monte Carlo import, so the PDE routes call it too. Decided-but-unpaid coupons from earlier rows are
not carried (W14) and a knocked-out DCN is assumed alive (W14). Payment dates are the grid context's, i.e.
``DCNOption.payment_dates()``: a ``settlement_convention`` times today's row as it times every later one (W15).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.asset.equity.engine.settlement_support import at_exercise_instant
from quantark.asset.equity.product.option.dcn_grid import DCNGridContext, build_dcn_grid_context
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

__all__ = [
    "DCNValuationRow",
    "at_final_observation",
    "observe_valuation_row",
    "settle_final_observation",
    "settled_row_delta",
]


@dataclass(frozen=True)
class DCNValuationRow:
    """The monthly row observed at the valuation instant.

    Leg PVs are UNSIGNED (buyer perspective, as the MC kernel's) and discounted to the valuation date; ``loss_pv`` is
    non-zero only on the final row. ``pv`` applies ``direction_sign`` leg by leg in the MC result's order.
    """

    is_final: bool
    knocked_in: bool
    knocked_out: bool
    coupon_paid: bool
    coupon_pv: float
    ko_pv: float
    loss_pv: float
    direction_sign: float
    loss_df: float

    @property
    def ends_contract(self) -> bool:
        """Nothing is left to price after today: the final row, or a knock-out today."""
        return self.is_final or self.knocked_out

    @property
    def pv(self) -> float:
        sign = self.direction_sign
        return float(sign * self.coupon_pv) + float(sign * self.ko_pv) + float(sign * self.loss_pv)


def at_final_observation(product) -> bool:
    """Whether the DCN is valued on its final observation date (``get_maturity`` 0.0)."""
    return at_exercise_instant(product.get_maturity())


def observe_valuation_row(
    product, pricing_env, ctx: Optional[DCNGridContext] = None, df=None,
) -> Optional[DCNValuationRow]:
    """The monthly row observed at the valuation instant, or ``None`` when the valuation date is not one."""
    ctx = build_dcn_grid_context(product) if ctx is None else ctx
    if ctx.obs_cols.size == 0 or int(ctx.obs_cols[0]) != 0:
        return None
    df = make_df_fn(pricing_env) if df is None else df
    spot = float(pricing_env.spot)
    part, notional, s0 = product.participation, product.notional, product.initial_price
    is_final = ctx.times.size == 1
    # the contract's inclusive inequalities, exactly as the kernel and the PDE masks state them (Q2: pointwise)
    knocked_in = bool(product.knocked_in_at_valuation) or spot <= product.ki_barrier
    knocked_out = bool(ctx.obs_is_ko[0]) and spot >= product.ko_barrier
    coupon_paid = (bool(ctx.obs_is_coupon[0]) and not knocked_out and not knocked_in
                   and spot >= product.coupon_barrier)
    ko_pv = coupon_pv = loss_pv = 0.0
    if knocked_out:
        dfs_k = np.array([df(t) for t in ctx.ko_pay_times])
        ko_pv = float(part * product.ko_coupon_rate * ctx.ko_accruals[0] * notional * dfs_k[0])
    if coupon_paid:
        dfs_c = np.array([df(t) for t in ctx.coupon_pay_times])
        coupon_pv = float(part * product.coupon_rate * ctx.coupon_accruals[0] * notional * dfs_c[0])
    loss_df = float(df(ctx.loss_pay_time))
    if is_final and knocked_in and not knocked_out:
        loss_pv = float(-(notional / s0) * part * np.maximum(product.k_loss - spot, 0.0) * loss_df)
    return DCNValuationRow(
        is_final=is_final, knocked_in=knocked_in, knocked_out=knocked_out, coupon_paid=coupon_paid,
        coupon_pv=coupon_pv, ko_pv=ko_pv, loss_pv=loss_pv, direction_sign=float(product.direction_sign),
        loss_df=loss_df,
    )


def settle_final_observation(product, pricing_env, ctx: Optional[DCNGridContext] = None) -> DCNValuationRow:
    """The DCN on its final observation date (T = 0): the settled final row."""
    ctx = build_dcn_grid_context(product) if ctx is None else ctx
    if ctx.times.size != 1:
        raise ValidationError("settle_final_observation needs a DCN valued on its final observation date")
    return observe_valuation_row(product, pricing_env, ctx)


def settled_row_delta(product, pricing_env, row: DCNValuationRow) -> float:
    """Q6: the slope of the settled final row at the spot (signed); 0 off the loss leg and at a tie, where the row
    jumps or kinks (the KO barrier; the coupon barrier when not knocked in; the KI barrier when not carried; K_loss on
    the loss leg)."""
    if not row.is_final:
        raise ValidationError("settled_row_delta is the final row's slope")
    spot = float(pricing_env.spot)
    ties = [product.ko_barrier]
    if not row.knocked_in:
        ties.append(product.coupon_barrier)
    if not product.knocked_in_at_valuation:
        ties.append(product.ki_barrier)
    on_loss_leg = row.knocked_in and not row.knocked_out
    if on_loss_leg:
        ties.append(product.k_loss)
    if any(is_close(spot, float(level)) for level in ties):
        return 0.0
    if not on_loss_leg or spot >= product.k_loss:
        return 0.0
    slope = (product.notional / product.initial_price) * product.participation * row.loss_df
    return float(row.direction_sign * slope)
