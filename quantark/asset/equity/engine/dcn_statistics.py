"""``EventStatistics`` of a DCN from directly computed outcomes (unified event statistics spec §Contracts, D12, AC17,
AC18; T15).

Engine-neutral (no Monte Carlo import): the DCN grid engines (``pde/dcn_statistics.py``) hand this module the outcome
probabilities their forward pass integrated, and it lays them out exactly as the DCN MC statistics do
(``mc/_dcn_event_statistics.py``, T11):

- the observation table: one row per monthly observation (a row inside ``ko_lock_months`` is a coupon-only row with
  KO probability 0), KO cash = the knock-out coupon only, paid at the row's knock-out payment;
- the coupon table on the same rows: P(the row's fixed coupon is paid), paid at ``ctx.coupon_pay_times``;
- the maturity row (never knocked out, never knocked in: nothing is paid) and the KI settlement row (knocked in, never
  knocked out: the loss leg), both decided at the final observation and paid at the loss-leg settlement;
- knock-in as the DCN monitors it (daily closes of an alive note, never after a knock-out) and the per-date knock-in
  table on the daily KI dates;
- the receivables of a lifecycle state as ledger rows, never events (a terminated note reports them alone, AC17).

Times and discounting are those of DCN ``price()`` (spec D12): the contract's ACT/365F grid read on the environment's
curve (``make_df_fn``); dates are the schedule's. Cash carries ``direction_sign``; probabilities, times and the
expected loss are unsigned. Every leg amount is the kernel's arithmetic (``dcn_payoff.compute_dcn_cashflows``,
``dcn_valuation.observe_valuation_row``): the discounted cash of a deterministic-payment row is ``sign x amount x P x
DF(payment)`` and its cash to pay ``sign x amount x P``; the KI settlement's cash is the loss leg's own expectation
(spot dependent), its cash to pay that over the settlement DF.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from quantark.asset.equity.engine.event_stats import (
    CashLedger,
    CouponTable,
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    ObservationTable,
    OutcomeRow,
    ReceivablesTable,
    StatisticsProvenance,
    build_cash_ledger,
)
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.numerical import Tolerance, is_zero

__all__ = [
    "NO_KNOCK_IN",
    "ZERO_TO_ROUND_OFF",
    "DCNRows",
    "decided_row_statistics",
    "dcn_statistics",
    "terminated_dcn_statistics",
]

#: expected knock-in time when no path knocks in before a knock-out (DESIGN §4.9)
NO_KNOCK_IN = "undefined: P(knock in) = 0"
#: expected knock-in time when the first knock-in mass is zero to round-off
ZERO_TO_ROUND_OFF = "undefined: P(knock in) is zero to round-off"


class DCNRows:
    """The DCN's fixed rows for one (product, valuation): the monthly axis and its dates, the leg amounts, the
    settlement DFs and the daily knock-in axis (``ctx.times[first_ki_col:]``, the valuation close only when it is a
    schedule date)."""

    def __init__(self, product, pricing_env, ctx) -> None:
        self.product, self.env, self.ctx = product, pricing_env, ctx
        self.sign = float(product.direction_sign)
        self.df = make_df_fn(pricing_env)
        schedule = product.schedule
        self.valuation_date = schedule.valuation_date
        self.times = np.asarray(ctx.times[ctx.obs_cols], dtype=float)
        rows = schedule.monthly
        pay = product.payment_dates()
        self.dates = tuple(row.observation_date for row in rows)
        self.ko_payment_dates = tuple(pay.ko_payment_dates)
        self.coupon_payment_dates = tuple(pay.coupon_payment_dates)
        self.loss_payment_date = pay.loss_payment_date
        part, notional = product.participation, product.notional
        self.ko_amount = part * product.ko_coupon_rate * np.asarray(ctx.ko_accruals, dtype=float) * notional
        self.coupon_amount = part * product.coupon_rate * np.asarray(ctx.coupon_accruals, dtype=float) * notional
        self.ko_pay_times = np.asarray(ctx.ko_pay_times, dtype=float)
        self.coupon_pay_times = np.asarray(ctx.coupon_pay_times, dtype=float)
        self.df_ko = np.array([float(self.df(t)) for t in self.ko_pay_times], dtype=float)
        self.df_coupon = np.array([float(self.df(t)) for t in self.coupon_pay_times], dtype=float)
        self.df_loss = float(self.df(ctx.loss_pay_time))
        self.first_ki_col = 0 if ctx.valuation_monitored else 1
        self.ki_times = np.asarray(ctx.times[self.first_ki_col:], dtype=float)
        ki_dates = tuple(schedule.daily_ki_dates)
        self.ki_dates = ki_dates if len(ki_dates) == self.ki_times.size else None
        self.carried = bool(product.knocked_in_at_valuation)
        self.knocked_in_now = (not self.carried and bool(ctx.valuation_monitored)
                               and float(pricing_env.spot) <= float(product.ki_barrier))


def _empty_ki_table() -> KnockInTable:
    return KnockInTable.empty()


def dcn_statistics(rows: DCNRows, *, pv: float, ko_probability, coupon_probability, maturity_probability: float,
                   ki_settlement_probability: float, ki_settlement_cash: float, ki_row_probability,
                   clean_ko_probability, knocked_out_now: bool, receivables: ReceivablesTable,
                   provenance: StatisticsProvenance) -> EventStatistics:
    """The statistics of directly computed DCN outcomes.

    Args:
        rows: the DCN's fixed rows.
        pv: the value ``price()`` returned (receivables included, spec D23).
        ko_probability, coupon_probability: per monthly row, P(KO there) and P(the row's fixed coupon is paid).
        maturity_probability, ki_settlement_probability: the two survivor outcomes.
        ki_settlement_cash: E[DF x loss leg x 1{settles knocked in}], buyer view (<= 0), unsigned by direction.
        ki_row_probability: P(first knock-in at each daily KI date before any KO), on ``rows.ki_times``; ignored when
            the knock-in is decided at valuation (carried or breached now) or the note knocks out at valuation.
        clean_ko_probability: per monthly row, P(KO there with no knock-in before it) (the KI table's survival).
        knocked_out_now: the note knocks out at the valuation instant.
        receivables: the lifecycle state's receivables (ledger rows, never events).
        provenance: the producing model.
    """
    ctx, sign = rows.ctx, rows.sign
    n = rows.times.size
    ko_p = np.asarray(ko_probability, dtype=float)
    cpn_p = np.asarray(coupon_probability, dtype=float)
    survival = 1.0 - np.cumsum(ko_p)
    observations = ObservationTable(
        times=rows.times, payment_times=rows.ko_pay_times, dates=rows.dates, payment_dates=rows.ko_payment_dates,
        is_ko_obs=ctx.obs_is_ko, is_coupon_obs=ctx.obs_is_coupon, probability=ko_p, survival=survival,
        discounted_cash=sign * rows.ko_amount * ko_p * rows.df_ko, cash_to_pay=sign * rows.ko_amount * ko_p,
        payment_is_path_dependent=np.zeros(n, dtype=bool),
    )
    coupons = CouponTable(
        probability=cpn_p, survival=np.concatenate([[1.0], survival[:-1]]),
        payment_times=rows.coupon_pay_times, payment_dates=rows.coupon_payment_dates,
        discounted_cash=sign * rows.coupon_amount * cpn_p * rows.df_coupon,
        cash_to_pay=sign * rows.coupon_amount * cpn_p, payment_is_path_dependent=np.zeros(n, dtype=bool),
    )
    timing = dict(time=float(rows.times[-1]), payment_time=float(ctx.loss_pay_time), date=rows.dates[-1],
                  payment_date=rows.loss_payment_date)
    maturity = OutcomeRow(probability=float(maturity_probability), discounted_cash=0.0, cash_to_pay=0.0, **timing)
    kis_cash = sign * float(ki_settlement_cash)
    ki_settlement = OutcomeRow(probability=float(ki_settlement_probability), discounted_cash=kis_cash,
                               cash_to_pay=kis_cash / rows.df_loss, **timing)
    unavailable: Dict[str, str] = {}
    if rows.carried or (rows.knocked_in_now and not knocked_out_now):
        # the knock-in is decided at valuation: one row at 0 dated the valuation date (DESIGN §4.5)
        ki_table = KnockInTable(times=[0.0], payment_times=[0.0], dates=(rows.valuation_date,),
                                payment_dates=(rows.valuation_date,), probability=[1.0], survival=[0.0],
                                interpolated=[False])
        probability, expected_time = 1.0, 0.0
    elif knocked_out_now:
        ki_table = _empty_ki_table()
        probability, expected_time = 0.0, None
        unavailable["knock_in.expected_time"] = NO_KNOCK_IN
    else:
        ki_p = np.asarray(ki_row_probability, dtype=float)
        clean = np.asarray(clean_ko_probability, dtype=float)
        order = np.argsort(rows.times, kind="stable")
        cumulative_ko = np.concatenate([[0.0], np.cumsum(clean[order])])
        # a knock-out at a row's own time is out by then (it wins the tie against a discrete knock-in)
        ko_by_row = cumulative_ko[np.searchsorted(rows.times[order], rows.ki_times + Tolerance.ZERO, side="right")]
        ki_table = KnockInTable(times=rows.ki_times, payment_times=rows.ki_times, dates=rows.ki_dates,
                                payment_dates=rows.ki_dates, probability=ki_p,
                                survival=1.0 - np.cumsum(ki_p) - ko_by_row,
                                interpolated=np.zeros(rows.ki_times.size, dtype=bool))
        probability = float(np.sum(ki_p))
        expected_time = None
        if probability == 0.0:
            unavailable["knock_in.expected_time"] = NO_KNOCK_IN
        elif is_zero(probability):
            unavailable["knock_in.expected_time"] = ZERO_TO_ROUND_OFF
        else:
            expected_time = float(np.sum(rows.ki_times * ki_p)) / probability
    knock_in = KnockInStatistics(probability=probability, probability_no_ko=ki_settlement.probability,
                                 expected_time=expected_time, expected_loss=-float(ki_settlement_cash))
    lifetime = float(np.sum(ko_p * rows.times)) + (maturity.probability + ki_settlement.probability) * float(
        rows.times[-1])
    ledger = build_cash_ledger(observations=observations, coupons=coupons, maturity=maturity,
                               ki_settlement=ki_settlement, receivables=receivables)
    return EventStatistics(
        product_family="dcn", pv=float(pv), observations=observations, coupons=coupons, maturity=maturity,
        ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table, receivables=receivables, ledger=ledger,
        expected_lifetime=lifetime, provenance=provenance, unavailable=unavailable,
    )


def decided_row_statistics(product, pricing_env, ctx, row, *, pv: float, receivables: ReceivablesTable,
                           provenance: StatisticsProvenance) -> EventStatistics:
    """A valuation-date row that ends the contract (``dcn_valuation.observe_valuation_row``: the final observation,
    W10, or a knock-out today, W10b), decided at the spot: every outcome is the row's 0/1 decision, exactly as the
    DCN MC's one-path settled row (``mc/dcn_mc_engine.settled_dcn_result``)."""
    rows = DCNRows(product, pricing_env, ctx)
    n = rows.times.size
    ko = np.zeros(n)
    coupon = np.zeros(n)
    if row.knocked_out:
        ko[0] = 1.0
    if row.coupon_paid:
        coupon[0] = 1.0
    survives = not row.knocked_out
    maturity = 1.0 if (survives and row.is_final and not row.knocked_in) else 0.0
    settles_ki = 1.0 if (survives and row.is_final and row.knocked_in) else 0.0
    # the final row's loss leg at the spot (buyer view, ``row.loss_pv``)
    ki_cash = float(row.loss_pv) if settles_ki else 0.0
    ki_rows = np.zeros(rows.ki_times.size)
    clean = ko.copy() if not row.knocked_in else np.zeros(n)
    return dcn_statistics(rows, pv=pv, ko_probability=ko, coupon_probability=coupon, maturity_probability=maturity,
                          ki_settlement_probability=settles_ki, ki_settlement_cash=ki_cash, ki_row_probability=ki_rows,
                          clean_ko_probability=clean, knocked_out_now=row.knocked_out, receivables=receivables,
                          provenance=provenance)


def terminated_dcn_statistics(product, state, *, pv: float, provenance: StatisticsProvenance,
                              standard_errors_reason: Optional[str] = None) -> Optional[EventStatistics]:
    """A knocked-out or matured lifecycle state (spec AC17, DESIGN 4.9c): its receivables only, or ``None`` when
    nothing is left unpaid.

    Every termination probability is 0 and the probability identity is not checked (the type recognises the
    terminated note from these rows); ``pv`` (the receivables' value) reconciles with the receivables. The knock-in
    path statistics follow the valuation-date rule of a note knocked out at valuation: P(knock in) = 1 with one KI
    row at t = 0 when the note was already knocked in, else 0 with an empty table. ``standard_errors_reason``: the
    ``unavailable["standard_errors"]`` reason of an engine that reports standard errors (MC); a grid engine has none.
    """
    receivables = state.receivables_table(product.schedule.valuation_date)
    if receivables.size == 0:
        return None
    valuation = product.schedule.valuation_date
    empty_rows = dict(times=[], payment_times=[], is_ko_obs=[], is_coupon_obs=[], probability=[], survival=[],
                      discounted_cash=[], cash_to_pay=[], payment_is_path_dependent=[])
    observations = ObservationTable(**empty_rows)
    coupons = CouponTable(probability=[], survival=[], payment_times=[], discounted_cash=[], cash_to_pay=[],
                          payment_is_path_dependent=[])
    nothing = dict(time=0.0, payment_time=0.0, probability=0.0, discounted_cash=0.0, cash_to_pay=0.0)
    carried = bool(product.knocked_in_at_valuation)
    unavailable: Dict[str, str] = {}
    if standard_errors_reason is not None:
        unavailable["standard_errors"] = standard_errors_reason
    if carried:
        ki_table = KnockInTable(times=[0.0], payment_times=[0.0], dates=(valuation,), payment_dates=(valuation,),
                                probability=[1.0], survival=[0.0], interpolated=[False])
    else:
        ki_table = KnockInTable.empty()
        unavailable["knock_in.expected_time"] = NO_KNOCK_IN
    knock_in = KnockInStatistics(probability=1.0 if carried else 0.0, probability_no_ko=0.0,
                                 expected_time=0.0 if carried else None, expected_loss=0.0)
    ledger = CashLedger(
        determination_times=receivables.decision_times, payment_times=receivables.payment_times,
        determination_dates=receivables.decision_dates, payment_dates=receivables.payment_dates,
        discounted_cash=receivables.discounted_cash, cash_to_pay=receivables.cash_to_pay,
    )
    return EventStatistics(
        product_family="dcn", pv=float(pv), observations=observations, coupons=coupons,
        maturity=OutcomeRow(**nothing), ki_settlement=OutcomeRow(**nothing), knock_in=knock_in, ki_table=ki_table,
        receivables=receivables, ledger=ledger, expected_lifetime=0.0, provenance=provenance,
        unavailable=unavailable,
    )
