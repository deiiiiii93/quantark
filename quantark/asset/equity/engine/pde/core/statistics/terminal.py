"""Statistics of a product knocked out at valuation: a terminal outcome, no march (spec 3B §5.5).

The family's valuation-date decision (decision 15) fired a knock-out, so every column is decided at the
spot and a march would contribute nothing, under every projection (decision 7). This builds the outcome
from the decision without a layout: one knock-out of probability 1 at the decision's record (a pre or a
post record on a KO-reset), its cash, the coupon paid with it, empty maturity legs.
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.settlement_support import first_schedule_settlement, resolve_terminal_timing
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.util.enum import CouponPayType
from quantark.util.numerical import is_close

from . import event_statistics as es

__all__ = ["knocked_out_event_statistics"]


def _position(records, record):
    """The decision's record in a resolved schedule (the one observed at the valuation instant)."""
    t = float(record.observation_time)
    return next(k for k, rec in enumerate(records) if is_close(float(rec.observation_time), t))


# --- EventStatistics (unified event statistics T12) ------------------------------------------------------


def knocked_out_event_statistics(solver, product, env, request, decision, pv=None):
    """``EventStatistics`` of the knock-out ``decision`` fired at valuation (DESIGN §4.5, §4.9c): one knock-out
    row of probability 1 at the decision's record (paid as ``_knocked_out_value`` pays it), the coupon decided
    with it on the coupon table (Phoenix), the maturity and knock-in settlement rows computed zeros, and the
    knock-in path statistics by the valuation-date rule -- a carried knock-in is P(knock in) = 1 at time 0,
    otherwise no knock-in happened (no forward calculation). ``pv``: the caller's (``price()``'s), else the request's external pv, else the decided value."""
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
                                            probability_no_ko=0.0, expected_loss=loss, unavailable=unavailable,
                                            valuation_dates=es.at_valuation_dates(product, env))
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
    # the note that ends with schedule 1: paid on the explicit settlement date when booked (spec D33), else the
    # terminal delay after schedule 1's maturity
    settlement = first_schedule_settlement(product, env, terminal)
    maturity_timing = dict(time=settlement.determination_time, payment_time=settlement.payment_time)
    if settlement.settlement_date_governs:
        maturity_timing.update(payment_date=settlement.payment_date)
    ki_timing = dict(time=tau, payment_time=tau + delay, date=terminal.determination_date,
                     payment_date=terminal.payment_date)
    return observations, maturity_timing, ki_timing
