"""Statistics of a product knocked out at valuation: a terminal outcome, no march (spec 3B §10 step 3).

Under the cell-average projection every column is decided at the spot when the valuation date's
knock-out fires, so a march would contribute nothing. This builds the outcome without a layout and hands
it to the shared readout tail: the knock-out's cash, the coupon paid with it, an empty maturity leg.
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.util.enum import CouponPayType
from quantark.util.numerical import is_close

from .readout import assemble_autocallable

__all__ = ["knocked_out_statistics"]


def knocked_out_statistics(solver, product, env, request):
    """The statistics of a knock-out that fires at valuation, from ``request``'s streams
    and pv (``external_pv``, else the immediate knock-out payoff)."""
    tau = float(product.get_maturity(env))
    ko_records = solver._filter_observations_by_tau(product.resolve_ko_observations(env), tau)
    n_ko = len(ko_records)
    rec0_pos = next(k for k, rec in enumerate(ko_records) if is_close(float(rec.observation_time), 0.0))
    rec0 = ko_records[rec0_pos]

    streams = request.streams
    if streams is None:
        want_ki = True
        want_coupon = True
    else:
        from quantark.cashleg.event_distribution import EventType

        want_ki = bool(streams & {EventType.KI, EventType.MATURITY_WITH_KI})
        want_coupon = EventType.COUPON in streams

    # paid from the valuation instant (t = 0 on every layout)
    df_delay0 = solver._cashflow_value_at_time(
        pricing_env=env, cashflow=1.0, current_time=0.0, settlement_time=rec0.settlement_time,
    )
    ed_unit = np.array([df_delay0 if i == rec0_pos else 0.0 for i in range(n_ko)], dtype=float)

    already_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
    if already_knocked_in:
        ki = (1.0, 1.0, np.array([0.0], dtype=float), np.array([1.0], dtype=float),
              np.array([0.0], dtype=float))
    elif want_ki:
        df_T = env.get_discount_factor(float(tau))
        ki_probability = float(0.0 / df_T) if df_T > 0.0 else 0.0
        ki = (ki_probability, ki_probability, np.array([], dtype=float), np.array([], dtype=float),
              np.array([], dtype=float))
    else:
        ki = (0.0, 0.0, np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float))

    pv = request.external_pv if request.external_pv is not None else solver._get_immediate_ko_payoff(product, env)
    extra_fields, coupon_rows = {}, None
    if isinstance(product, PhoenixOption):
        extra_fields, coupon_rows = _knocked_out_coupon(solver, product, env, ko_records, rec0_pos, df_delay0,
                                                        want_coupon)
    return assemble_autocallable(solver, product, env, pv=pv, ko_records=ko_records, ed_unit=ed_unit,
                                 ki=ki, extra_fields=extra_fields, maturity=0.0, coupon_rows=coupon_rows)


def _knocked_out_coupon(solver, product, env, ko_records, rec0_pos, df_delay0, want_coupon):
    """The coupon decided with the knock-out (spec 3B §5.3): paid with the redemption, as the value's
    ``_get_immediate_ko_payoff`` pays it, memory arrears included; every later coupon is dead."""
    n_ko = len(ko_records)
    coupon = solver._immediate_ko_coupon(product, env)
    cash = np.zeros(n_ko, dtype=float)
    cash[rec0_pos] = coupon * df_delay0
    coupon_probability = np.zeros(n_ko, dtype=float)
    coupon_probability[rec0_pos] = 1.0 if product.is_coupon_triggered(env.spot, 0) else 0.0
    det = [float(r.observation_time) for r in ko_records]
    pay = [float(r.settlement_time if r.settlement_time is not None else r.observation_time) for r in ko_records]
    det_dates, pay_dates = [r.observation_date for r in ko_records], [r.settlement_date for r in ko_records]
    expiry = product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
    if expiry:
        # the n + 1 payment buckets of the marched readout; the terminal bucket is empty
        terminal = resolve_terminal_timing(product, env)
        coupon_rows = (det + [float(terminal.determination_time)], pay + [float(terminal.payment_time)],
                       np.append(cash, 0.0), det_dates + [terminal.determination_date],
                       pay_dates + [terminal.payment_date])
    else:
        coupon_rows = (det, pay, cash, det_dates, pay_dates)
    if want_coupon:
        extra = {"coupon_probability": coupon_probability, "expected_discounted_coupon_cashflow": cash}
    else:       # the report is pruned; the ledger keeps the coupon rows (decision 3)
        extra = {"coupon_probability": np.array([]), "expected_discounted_coupon_cashflow": np.array([])}
    if expiry:
        extra["coupon_payment_is_path_dependent"] = True
    return extra, coupon_rows
