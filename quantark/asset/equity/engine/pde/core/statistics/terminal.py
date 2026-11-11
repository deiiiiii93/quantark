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

from . import event_statistics as es
from .readout import assemble_autocallable

__all__ = ["knocked_out_statistics", "knocked_out_event_statistics"]


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


def _ki_fields(product, env, solver, request):
    """A knocked-out note settles nothing knocked in: ``ki_probability`` (= ki_survive) is 0 whatever the
    streams; a carried knock-in is ``ki_ever`` 1 at time 0, else no knock-in happened."""
    from .snowball import _want
    if not _want(request.streams)[0]:
        empty = np.array([], dtype=float)
        return (0.0, 0.0, empty, empty, empty)
    if bool(getattr(product, "_otc_lifecycle_knocked_in", False)):
        return (0.0, 1.0, np.array([0.0], dtype=float), np.array([1.0], dtype=float), np.array([0.0], dtype=float))
    empty = np.array([], dtype=float)
    from ..adapters.base import continuous_knock_in
    if solver.adapter.statistics_model == "flat_vol_1d" and continuous_knock_in(product):
        from quantark.asset.equity.engine.continuous_ki_support import continuous_ki_ever
        return (0.0, continuous_ki_ever(product, env, solver.params), empty, empty, empty)
    if solver.adapter.statistics_model == "local_vol_1d" and continuous_knock_in(product):
        from ...vol_model_statistics import local_vol_ki_ever
        return (0.0, local_vol_ki_ever(solver, product, env), empty, empty, empty)
    return (0.0, 0.0, empty, empty, empty)


def knocked_out_statistics(solver, product, env, request, decision, pv=None):
    """The statistics of the knock-out ``decision`` fired at valuation, from ``request``'s streams; pv is
    ``pv`` (the caller's knocked-out value), else ``external_pv``, else ``solver._knocked_out_value``."""
    if pv is None:
        pv = (request.external_pv if request.external_pv is not None
              else solver._knocked_out_value(product, env, decision))
    want_coupon = _wanted(request)
    if isinstance(product, KnockOutResetSnowballOption):
        return _ko_reset_statistics(solver, product, env, decision, float(pv), request)
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
                                 ki=_ki_fields(product, env, solver, request), extra_fields=extra_fields, maturity=0.0,
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


def _ko_reset_statistics(solver, product, env, decision, pv, request):
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
    ki_probability, ki_ever, ki_times, ki_event, ki_survival = _ki_fields(product, env, solver, request)
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


# --- EventStatistics (unified event statistics T12) ------------------------------------------------------


def knocked_out_event_statistics(solver, product, env, request, decision, pv=None):
    """``EventStatistics`` of the knock-out ``decision`` fired at valuation (DESIGN §4.5, §4.9c): one knock-out
    row of probability 1 at the decision's record (paid as ``_knocked_out_value`` pays it), the coupon decided
    with it on the coupon table (Phoenix), the maturity and knock-in settlement rows computed zeros, and the
    knock-in path statistics by the valuation-date rule -- a carried knock-in is P(knock in) = 1 at time 0,
    otherwise no knock-in happened (no forward calculation). ``pv`` as ``knocked_out_statistics``."""
    if pv is None:
        pv = (request.external_pv if request.external_pv is not None
              else solver._knocked_out_value(product, env, decision))
    carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
    rule = es.knock_in_rule(product, knocked_out_now=True, knocked_in_now=carried)
    terminal = resolve_terminal_timing(product, env)
    unavailable = {}
    if isinstance(product, KnockOutResetSnowballOption):
        observations, maturity_timing, ki_timing = _ko_reset_knocked_out_rows(solver, product, env, decision, terminal)
        coupons = path_dependent_cash = None
        family = "ko_reset"
    else:
        tau = float(product.get_maturity(env))
        records = solver._filter_observations_by_tau(product.resolve_ko_observations(env), tau)
        pos = _position(records, decision.ko_record)
        rec0 = records[pos]
        # paid from the valuation instant (t = 0 on every layout)
        df_delay0 = solver._cashflow_value_at_time(pricing_env=env, cashflow=1.0, current_time=0.0,
                                                   settlement_time=rec0.settlement_time)
        unit = np.array([df_delay0 if i == pos else 0.0 for i in range(len(records))], dtype=float)
        coupons = path_dependent_cash = None
        if isinstance(product, PhoenixOption):
            observations = es.observation_table(env, records, unit, is_coupon_obs=np.ones(len(records), dtype=bool))
            coupons, path_dependent_cash = _knocked_out_coupon_table(solver, product, env, records, pos, df_delay0,
                                                                     observations, terminal, request, unavailable)
            family = "phoenix"
        else:
            observations = es.observation_table(env, records, unit)
            family = "snowball"
        timing = dict(time=float(terminal.determination_time), payment_time=float(terminal.payment_time),
                      date=terminal.determination_date, payment_date=terminal.payment_date)
        maturity_timing = ki_timing = timing
    maturity = es.outcome_row(env, probability=0.0, discounted=0.0, **maturity_timing)
    ki_settlement = es.outcome_row(env, probability=0.0, discounted=0.0, **ki_timing)
    loss = es.expected_loss(product, env, probability=0.0, discounted_cash=0.0, payment_time=ki_settlement.payment_time)
    knock_in, ki_table = es.knock_in_fields(rule, want=es.want_knock_in_path(request.streams),
                                            probability_no_ko=0.0, expected_loss=loss, unavailable=unavailable)
    lifetime = float(np.sum(observations.probability * observations.times))
    return es.assemble(family=family, pv=float(pv), observations=observations, coupons=coupons, maturity=maturity,
                       ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table, expected_lifetime=lifetime,
                       provenance=solver._native_provenance(), unavailable=unavailable,
                       path_dependent_cash=path_dependent_cash)


def _knocked_out_coupon_table(solver, product, env, records, pos, df_delay0, observations, terminal, request,
                              unavailable):
    """The coupon decided with the knock-out (spec 3B §5.3): paid with the redemption, as the value's
    ``_get_immediate_ko_payoff`` pays it, memory arrears included; every later coupon is dead. Under EXPIRY the
    row is path-dependent by convention and the ledger holds the coupon at the knock-out's settlement."""
    from quantark.asset.equity.engine.event_stats import CashLedger, CouponTable

    n = len(records)
    coupon = float(solver._immediate_ko_coupon(product, env))
    cash = np.zeros(n, dtype=float)
    cash[pos] = coupon * df_delay0
    to_pay = np.zeros(n, dtype=float)
    to_pay[pos] = coupon
    probability = np.zeros(n, dtype=float)
    # the decision's trigger: the knocked-out period's own coupon barrier (decision 13, F10)
    probability[pos] = 1.0 if solver._valuation_decision(product, env).coupon_index is not None else 0.0
    if not es.want_coupon_probability(request.streams):
        unavailable["coupons.probability"] = es.NOT_REQUESTED
    survival = np.concatenate([[1.0], observations.survival[:-1]]) if n else np.zeros(0)
    expiry = product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
    shown = probability if "coupons.probability" not in unavailable else None
    if not expiry:
        payment = observations.payment_times
        coupons = CouponTable(probability=shown, survival=survival, payment_times=payment,
                              payment_dates=observations.payment_dates, discounted_cash=cash, cash_to_pay=to_pay,
                              payment_is_path_dependent=np.zeros(n, dtype=bool))
        return coupons, None
    terminal_pay = float(terminal.payment_time)
    coupons = CouponTable(probability=shown, survival=survival, payment_times=np.full(n, terminal_pay),
                          payment_dates=None if terminal.payment_date is None else (terminal.payment_date,) * n,
                          discounted_cash=cash, cash_to_pay=to_pay, payment_is_path_dependent=np.full(n, True))
    # the n + 1 payment buckets of the marched readout; the terminal bucket is empty
    determination = np.concatenate([observations.times, [float(terminal.determination_time)]])
    payment = np.concatenate([observations.payment_times, [terminal_pay]])
    determination_dates = (None if observations.dates is None or terminal.determination_date is None
                           else observations.dates + (terminal.determination_date,))
    payment_dates = (None if observations.payment_dates is None or terminal.payment_date is None
                     else observations.payment_dates + (terminal.payment_date,))
    ledger = CashLedger(determination_times=determination, payment_times=payment,
                        determination_dates=determination_dates, payment_dates=payment_dates,
                        discounted_cash=np.append(cash, 0.0), cash_to_pay=np.append(to_pay, 0.0))
    return coupons, ledger


def _ko_reset_knocked_out_rows(solver, product, env, decision, terminal):
    """``(observations, maturity timing, knock-in settlement timing)`` of a KO-reset knocked out at valuation:
    both schedules on one table (the pre record first at a tie), probability 1 on the decision's record of its
    schedule, paid as ``_knocked_out_value`` pays it."""
    tau = float(product.get_maturity(env))
    pre = solver._filter_observations_by_tau(solver._get_cached_pre_ko_records(env, product), tau)
    post = solver._filter_observations_by_tau(solver._get_cached_post_ko_records(env, product), tau)
    fired = pre if decision.ko_regime == "pre" else post
    pos = _position(fired, decision.ko_record) + (0 if decision.ko_regime == "pre" else len(pre))
    records = list(pre) + list(post)
    settle = records[pos].settlement_time
    df = float(env.get_discount_factor(settle)) if settle is not None and settle > 0.0 else 1.0
    unit = np.array([df if i == pos else 0.0 for i in range(len(records))], dtype=float)
    times = np.array([float(r.observation_time) for r in records], dtype=float)
    tags = np.r_[np.zeros(len(pre)), np.ones(len(post))]
    order = np.lexsort((tags, times))
    observations = es.observation_table(env, records, unit, schedule=["pre"] * len(pre) + ["post"] * len(post),
                                        order=order)
    delay = float(terminal.payment_time) - float(terminal.determination_time)
    pre_maturity = float(product.get_pre_maturity_time(env))
    maturity_timing = dict(time=pre_maturity, payment_time=pre_maturity + delay)
    ki_timing = dict(time=tau, payment_time=tau + delay, date=terminal.determination_date,
                     payment_date=terminal.payment_date)
    return observations, maturity_timing, ki_timing
