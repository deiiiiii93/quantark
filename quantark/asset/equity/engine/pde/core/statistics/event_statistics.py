"""``EventStatistics`` of the PDE 1D statistics groups (unified event statistics spec §Contracts, T12).

The statistics group marches one column per termination outcome beside the value (``columns.py``); this module
turns the columns read at the spot into the one result type:

* every outcome directly from its own column (DESIGN §4.7): knock-out rows (the knock-out redemption only, a
  coupon paid with it sits in the coupon table, AC11), the maturity row (never knocked out, never knocked in)
  and the knock-in settlement row (knocked in, never knocked out), each probability from its unit column and
  each cash from its cash column;
* cash to pay (DESIGN §4.6): ``discounted / DF(payment)`` on rows whose payment time is deterministic; the
  producer's own undiscounted column on path-dependent rows (memory EXPIRY coupons);
* the knock-in path statistics by the valuation-date rules (DESIGN §4.5): knocked out at valuation, knocked in
  at valuation (carried, or decided now), no knock-in barrier. Elsewhere P(knock in), the expected first
  knock-in time and the per-date knock-in table come from the forward adjoint pass (spec T13, ``adjoint.py``),
  run only when the request asks for the knock-in stream; a route the pass cannot transpose reports its reason,
  never a stand-in;
* the expected knocked-in loss (DESIGN §4.5): ``E[DF(payment) (principal - knocked-in redemption) 1{settles
  knocked in}] = principal DF(payment) P(KI settlement) - KI settlement cash``, both marched columns.

``pv`` is ``price()``'s value (spec D23); ``reconciliation_error`` and ``probability_residual`` are the type's.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from quantark.asset.equity.engine.knock_in_days import at_valuation_dates  # noqa: F401 (re-export)
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

__all__ = [
    "NOT_REQUESTED",
    "NO_KNOCK_IN",
    "PENDING_FORWARD_PASS",
    "KnockInRule",
    "all_dates",
    "assemble",
    "expected_loss",
    "knock_in_fields",
    "knock_in_rule",
    "observation_table",
    "outcome_row",
    "to_pay",
    "want_coupon_probability",
    "want_knock_in_path",
]

#: the reason of the knock-in path statistics a PDE 1D statistics march does not produce (spec T13 adds the pass)
PENDING_FORWARD_PASS = (
    "undefined: this statistics group has no forward adjoint pass for P(knock in), its expected time and the "
    "per-date knock-in table"
)
#: a pruned value (DESIGN §4.9b)
NOT_REQUESTED = "not requested"
#: expected knock-in time when no path knocks in before a knock-out (DESIGN §4.9)
NO_KNOCK_IN = "undefined: P(knock in) = 0"


class KnockInRule:
    """The valuation-date rule that decides the knock-in path statistics without a forward pass (DESIGN §4.5)."""

    KNOCKED_IN_NOW = "knocked_in_now"      # carried, or a knock-in decided at valuation: P = 1, one row at 0
    NO_KNOCK_IN = "no_knock_in"            # no barrier, or knocked out at valuation without a carried knock-in
    FORWARD = "forward"                    # the forward adjoint pass (T13)


def want_knock_in_path(streams) -> bool:
    """Whether a stream selection asks for the knock-in path statistics (the KI stream, DESIGN §4.9b)."""
    if streams is None:
        return True
    from quantark.cashleg.event_distribution import EventType

    return EventType.KI in streams


def want_coupon_probability(streams) -> bool:
    """Whether a stream selection asks for the coupon table's probability column (the COUPON stream)."""
    if streams is None:
        return True
    from quantark.cashleg.event_distribution import EventType

    return EventType.COUPON in streams


def knock_in_rule(product, *, knocked_out_now: bool, knocked_in_now: bool) -> str:
    """The rule for a note knocked out (``knocked_out_now``) or knocked in (``knocked_in_now``: carried, or a
    level breached at valuation) at the valuation instant; a knock-out at valuation keeps only a carried
    knock-in (the caller passes ``knocked_in_now`` = carried there)."""
    if knocked_in_now:
        return KnockInRule.KNOCKED_IN_NOW
    if knocked_out_now or not product.has_ki_barrier:
        return KnockInRule.NO_KNOCK_IN
    return KnockInRule.FORWARD


def all_dates(values) -> Optional[tuple]:
    values = tuple(values)
    return values if values and all(value is not None for value in values) else None


def to_pay(env, discounted, payment_times) -> np.ndarray:
    """Cash to pay of rows with a deterministic payment time: ``discounted / DF(payment)`` (DESIGN §4.6)."""
    discounted = np.asarray(discounted, dtype=float)
    df = np.array([float(env.get_discount_factor(float(t))) for t in np.atleast_1d(payment_times)], dtype=float)
    return discounted / df


def observation_table(env, records, unit: np.ndarray, *, is_coupon_obs: Optional[np.ndarray] = None,
                      schedule: Optional[Sequence[str]] = None, order: Optional[np.ndarray] = None,
                      survival: Optional[np.ndarray] = None) -> ObservationTable:
    """The knock-out table of ``records`` from their discounted knock-out units (one paid at the record's
    settlement): probability ``unit / DF(settlement)``, cash ``unit x payoff`` (the knock-out redemption)."""
    records = list(records)
    n = len(records)
    order = np.arange(n) if order is None else np.asarray(order, dtype=int)
    times = np.array([float(r.observation_time) for r in records], dtype=float)
    payment = np.array([float(r.settlement_time if r.settlement_time is not None else r.observation_time)
                        for r in records], dtype=float)
    payoff = np.array([float(r.payoff) if r.payoff is not None else 0.0 for r in records], dtype=float)
    unit = np.asarray(unit, dtype=float)
    df = np.array([float(env.get_discount_factor(t)) for t in payment], dtype=float)
    probability = unit / df
    discounted = unit * payoff
    cash_to_pay = payoff * probability
    is_ko = np.array([r.barrier is not None for r in records], dtype=bool)
    probability = np.where(is_ko, probability, 0.0)
    discounted = np.where(is_ko, discounted, 0.0)
    cash_to_pay = np.where(is_ko, cash_to_pay, 0.0)
    times, payment = times[order], payment[order]
    probability, discounted, cash_to_pay, is_ko = probability[order], discounted[order], cash_to_pay[order], is_ko[order]
    if survival is None:
        survival = 1.0 - np.cumsum(probability)
    dates = all_dates(records[i].observation_date for i in order)
    payment_dates = all_dates(records[i].settlement_date for i in order)
    return ObservationTable(
        times=times, payment_times=payment, dates=dates, payment_dates=payment_dates, is_ko_obs=is_ko,
        is_coupon_obs=np.zeros(n, dtype=bool) if is_coupon_obs is None else np.asarray(is_coupon_obs, dtype=bool),
        schedule=None if schedule is None else tuple(schedule[i] for i in order),
        probability=probability, survival=np.asarray(survival, dtype=float), discounted_cash=discounted,
        cash_to_pay=cash_to_pay, payment_is_path_dependent=np.zeros(n, dtype=bool),
    )


def outcome_row(env, *, time: float, payment_time: float, date=None, payment_date=None, probability: float,
                discounted: float) -> OutcomeRow:
    """A scalar outcome paid at a deterministic time: cash to pay ``discounted / DF(payment)``."""
    return OutcomeRow(time=float(time), payment_time=float(payment_time), date=date, payment_date=payment_date,
                      probability=float(probability), discounted_cash=float(discounted),
                      cash_to_pay=float(to_pay(env, [discounted], [payment_time])[0]))


def knock_in_fields(rule: str, *, want: bool, probability_no_ko: float, expected_loss: float,
                    unavailable: Dict[str, str], forward: Optional[Callable[[], tuple]] = None,
                    valuation_dates: Optional[tuple] = None):
    """``(KnockInStatistics, ki_table)`` by the valuation-date ``rule``; the reasons go into ``unavailable``.

    ``forward``: the forward adjoint pass (``adjoint.knock_in_from_forward_pass`` bound to the march), run only for
    the ``FORWARD`` rule of a request that asks for the knock-in stream (D17: a pruned request never runs it); it
    returns ``(probability, expected_time, ki_table, expected_time_reason)`` or raises ``AdjointUnavailable``.
    ``valuation_dates``: the at-valuation row's dates (``knock_in_days.at_valuation_dates``; ``None`` on a numeric
    note)."""
    if rule == KnockInRule.KNOCKED_IN_NOW:
        table = KnockInTable(times=np.zeros(1), payment_times=np.zeros(1), dates=valuation_dates,
                             payment_dates=valuation_dates, probability=np.ones(1),
                             survival=np.zeros(1), interpolated=np.zeros(1, dtype=bool))
        stats = KnockInStatistics(probability=1.0, probability_no_ko=probability_no_ko, expected_time=0.0,
                                  expected_loss=expected_loss)
        return stats, table
    if rule == KnockInRule.NO_KNOCK_IN:
        unavailable["knock_in.expected_time"] = NO_KNOCK_IN
        stats = KnockInStatistics(probability=0.0, probability_no_ko=probability_no_ko, expected_time=None,
                                  expected_loss=expected_loss)
        return stats, KnockInTable.empty()
    if want and forward is not None:
        from .adjoint import AdjointUnavailable

        try:
            probability, expected_time, table, time_reason = forward()
        except AdjointUnavailable as refused:
            reason = refused.reason
        else:
            if time_reason is not None:
                unavailable["knock_in.expected_time"] = time_reason
            stats = KnockInStatistics(probability=probability, probability_no_ko=probability_no_ko,
                                      expected_time=expected_time, expected_loss=expected_loss)
            return stats, table
    else:
        reason = PENDING_FORWARD_PASS if want else NOT_REQUESTED
    for key in ("knock_in.probability", "knock_in.expected_time", "ki_table"):
        unavailable[key] = reason
    stats = KnockInStatistics(probability=None, probability_no_ko=probability_no_ko, expected_time=None,
                              expected_loss=expected_loss)
    return stats, None


def expected_loss(product, env, *, probability: float, discounted_cash: float, payment_time: float) -> float:
    """``principal DF(payment) P(KI settlement) - KI settlement cash`` (DESIGN §4.5): the discounted knocked-in
    downside ``principal - redemption`` over the paths that settle knocked in, ``principal`` 0 when excluded."""
    principal = (float(product.initial_price) * float(product.contract_multiplier)
                 if product.payoff_config.include_principal else 0.0)
    return principal * float(env.get_discount_factor(float(payment_time))) * float(probability) - float(discounted_cash)


def assemble(*, family: str, pv: float, observations: ObservationTable, coupons: Optional[CouponTable],
             maturity: OutcomeRow, ki_settlement: OutcomeRow, knock_in: KnockInStatistics,
             ki_table: Optional[KnockInTable], expected_lifetime: float, provenance: StatisticsProvenance,
             unavailable: Dict[str, str], path_dependent_cash: Optional[CashLedger] = None) -> EventStatistics:
    """The result, its ledger built from the directly computed tables (``build_cash_ledger``)."""
    receivables = ReceivablesTable.empty()
    ledger = build_cash_ledger(observations=observations, coupons=coupons, maturity=maturity,
                               ki_settlement=ki_settlement, receivables=receivables,
                               path_dependent_cash=path_dependent_cash)
    return EventStatistics(
        product_family=family, pv=float(pv), observations=observations, coupons=coupons, maturity=maturity,
        ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table, receivables=receivables, ledger=ledger,
        expected_lifetime=float(expected_lifetime), provenance=provenance, unavailable=unavailable,
    )
