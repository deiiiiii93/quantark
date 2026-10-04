"""Statistics of a product knocked out at valuation: a terminal outcome, no march (spec 3B §5.5).

The family's valuation-date decision (decision 15) fired a knock-out, so every column is decided at the
spot and a march would contribute nothing, under every projection (decision 7). This builds the outcome
from the decision without a layout: one knock-out of probability 1 at the decision's record (a pre or a
post record on a KO-reset), its cash, the coupon paid with it, empty maturity legs.
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.event_stats import payment_aware_cashflow_fields
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.util.enum import CouponPayType
from quantark.util.numerical import is_close

from .readout import assemble_autocallable

__all__ = ["knocked_out_statistics"]


def _position(records, record):
    """The decision's record in a resolved schedule (the one observed at the valuation instant)."""
    t = float(record.observation_time)
    return next(k for k, rec in enumerate(records) if is_close(float(rec.observation_time), t))


def _wanted(request):
    """Whether the request's streams report the coupon fields."""
    if request.streams is None:
        return True
    from quantark.cashleg.event_distribution import EventType

    return EventType.COUPON in request.streams


def _ki_fields(product):
    """A knocked-out note settles nothing knocked in: ``ki_probability`` (= ki_survive) is 0 whatever the
    streams; a carried knock-in is ``ki_ever`` 1 at time 0, else no knock-in happened."""
    if bool(getattr(product, "_otc_lifecycle_knocked_in", False)):
        return (0.0, 1.0, np.array([0.0], dtype=float), np.array([1.0], dtype=float), np.array([0.0], dtype=float))
    empty = np.array([], dtype=float)
    return (0.0, 0.0, empty, empty, empty)


def knocked_out_statistics(solver, product, env, request, decision, pv=None):
    """The statistics of the knock-out ``decision`` fired at valuation, from ``request``'s streams; pv is
    ``pv`` (the caller's knocked-out value), else ``external_pv``, else ``solver._knocked_out_value``."""
    if pv is None:
        pv = (request.external_pv if request.external_pv is not None
              else solver._knocked_out_value(product, env, decision))
    want_coupon = _wanted(request)
    if isinstance(product, KnockOutResetSnowballOption):
        return _ko_reset_statistics(solver, product, env, decision, float(pv))
    tau = float(product.get_maturity(env))
    ko_records = solver._filter_observations_by_tau(product.resolve_ko_observations(env), tau)
    n_ko = len(ko_records)
    rec0_pos = _position(ko_records, decision.ko_record)
    rec0 = ko_records[rec0_pos]
    # paid from the valuation instant (t = 0 on every layout)
    df_delay0 = solver._cashflow_value_at_time(
        pricing_env=env, cashflow=1.0, current_time=0.0, settlement_time=rec0.settlement_time,
    )
    ed_unit = np.array([df_delay0 if i == rec0_pos else 0.0 for i in range(n_ko)], dtype=float)
    extra_fields, coupon_rows = {}, None
    if isinstance(product, PhoenixOption):
        extra_fields, coupon_rows = _knocked_out_coupon(solver, product, env, ko_records, rec0_pos, df_delay0,
                                                        want_coupon)
    return assemble_autocallable(solver, product, env, pv=pv, ko_records=ko_records, ed_unit=ed_unit,
                                 ki=_ki_fields(product), extra_fields=extra_fields, maturity=0.0,
                                 coupon_rows=coupon_rows)


def _knocked_out_coupon(solver, product, env, ko_records, rec0_pos, df_delay0, want_coupon):
    """The coupon decided with the knock-out (spec 3B §5.3): paid with the redemption, as the value's
    ``_get_immediate_ko_payoff`` pays it, memory arrears included; every later coupon is dead."""
    n_ko = len(ko_records)
    coupon = solver._immediate_ko_coupon(product, env)
    cash = np.zeros(n_ko, dtype=float)
    cash[rec0_pos] = coupon * df_delay0
    coupon_probability = np.zeros(n_ko, dtype=float)
    # the decision's trigger: the knocked-out period's own coupon barrier (decision 13, F10)
    coupon_probability[rec0_pos] = 1.0 if solver._valuation_decision(product, env).coupon_index is not None else 0.0
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


def _ko_reset_statistics(solver, product, env, decision, pv):
    """``KOResetEventStats`` with the decision's record in its schedule's fields (``decision.ko_regime``),
    paid as ``_knocked_out_value`` pays it, and both maturity buckets 0 at their times."""
    tau = float(product.get_maturity(env))
    pre = solver._filter_observations_by_tau(solver._get_cached_pre_ko_records(env, product), tau)
    post = solver._filter_observations_by_tau(solver._get_cached_post_ko_records(env, product), tau)
    regime = pre if decision.ko_regime == "pre" else post
    pos = _position(regime, decision.ko_record)
    settle = solver._ko_settlement_time(product, env, decision.ko_record)
    df = float(env.get_discount_factor(settle)) if settle is not None and settle > 0.0 else 1.0

    def schedule(records, fired):
        times = np.array([float(r.observation_time) for r in records], dtype=float)
        pay = np.array([float(r.settlement_time if r.settlement_time is not None else r.observation_time)
                        for r in records], dtype=float)
        prob, cash = np.zeros(len(records)), np.zeros(len(records))
        if fired:
            prob[pos], cash[pos] = 1.0, float(decision.ko_record.payoff) * df
            if settle is not None:
                pay[pos] = float(settle)
        return times, pay, prob, cash

    pre_t, pre_pay, pre_prob, pre_cash = schedule(pre, decision.ko_regime == "pre")
    post_t, post_pay, post_prob, post_cash = schedule(post, decision.ko_regime == "post")
    terminal = resolve_terminal_timing(product, env)
    delay = float(terminal.payment_time) - float(terminal.determination_time)
    pre_maturity, post_maturity = float(product.get_pre_maturity_time(env)), tau
    ledger = payment_aware_cashflow_fields(
        env,
        determination_times=[*pre_t, *post_t, pre_maturity, post_maturity],
        payment_times=[*pre_pay, *post_pay, pre_maturity + delay, post_maturity + delay],
        expected_discounted_cashflows=[*pre_cash, *post_cash, 0.0, 0.0],
        determination_dates=[*(r.observation_date for r in pre), *(r.observation_date for r in post),
                             None, terminal.determination_date],
        payment_dates=[*(r.settlement_date for r in pre), *(r.settlement_date for r in post),
                       None, terminal.payment_date],
    )
    ki_probability, ki_ever, ki_times, ki_event, ki_survival = _ki_fields(product)
    return solver._make_event_stats(
        pv=pv,
        ko_times=pre_t,
        ko_probability=pre_prob,
        survival_probability=np.maximum(0.0, 1.0 - np.cumsum(pre_prob)),
        expected_discounted_ko_cashflow=pre_cash,
        ki_probability=ki_probability,
        expected_discounted_maturity_cashflow=0.0,
        reconciliation_error=float(pv - float(np.sum(pre_cash)) - float(np.sum(post_cash))),
        ki_times=ki_times,
        ki_event_probability=ki_event,
        ki_survival_probability=ki_survival,
        ki_ever_probability=ki_ever,
        ki_survive_knocked_in_probability=ki_probability,
        pre_ko_times=pre_t,
        pre_ko_probability=pre_prob,
        post_ko_times=post_t,
        post_ko_probability=post_prob,
        pre_ko_probability_total=float(np.sum(pre_prob)),
        post_ko_probability_total=float(np.sum(post_prob)),
        expected_discounted_post_ko_cashflow=float(np.sum(post_cash)),
        pre_ko_payment_times=pre_pay,
        post_ko_payment_times=post_pay,
        expected_discounted_pre_maturity_cashflow=0.0,
        expected_discounted_post_maturity_cashflow=0.0,
        pre_maturity_probability=0.0,
        post_maturity_probability=0.0,
        pre_maturity_time=pre_maturity,
        post_maturity_time=post_maturity,
        pre_maturity_payment_time=pre_maturity + delay,
        post_maturity_payment_time=post_maturity + delay,
        **ledger,
    )
