"""``EventStatistics`` of the DCN MC engines (unified event statistics spec §Contracts, D12, AC17, AC18; T11).

``DCNMCEngine.price_detailed`` / ``calculate_event_stats`` / ``price_with_events`` (and every vol-model DCN subclass,
which only replaces the path generation) record one batch of outcome sums per priced batch of the price's own paths
(``dcn_mc_engine._dcn_batch_record``). This module turns them into the one result type:

- the observation table: one row per monthly observation (the KO and coupon observations are one grid; a row inside
  ``ko_lock_months`` is a coupon-only row with KO probability 0), KO cash = the knock-out coupon only, paid at the
  row's knock-out payment;
- the coupon table on the same rows: P(the row's fixed coupon is paid) -- the contract pays no fixed coupon on a
  knock-out and none once knocked in, so neither counts -- and its cash, paid at the row's coupon payment
  (``ctx.coupon_pay_times``, distinct from the knock-out payment);
- the maturity row (never knocked out, never knocked in: nothing is paid) and the KI settlement row (knocked in,
  never knocked out: the loss leg), both decided at the final observation and paid at the loss-leg settlement;
- knock-in as the DCN always monitored it: daily closes of an alive note, so never after a knock-out; the per-date
  knock-in table on the daily KI dates (the product's own schedule, exact discrete monitoring);
- the receivables of a lifecycle state (cash decided at or before the valuation, paid after it) as ledger rows, never
  events; a terminated note with unpaid cash reports them alone (spec AC17);
- standard errors from independent batch means (spec AC21).

Times and discounting are those of DCN ``price()``: the contract's grid on the environment's axis from the schedule's
valuation date, read on the environment's curve; dates are the schedule's exact dates. Cash carries ``price()``'s
sign (``direction_sign``: a seller reports negative cash); probabilities, times and the expected loss are unsigned.
``pv`` is the value ``price()`` returned (spec D23); ``reconciliation_error`` and ``probability_residual`` are the
type's own computation, never absorbed.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from quantark.asset.equity.engine.event_stats import (
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
from quantark.asset.equity.engine.dcn_statistics import ki_row_dates
from quantark.asset.equity.engine.knock_in_days import at_valuation_dates
from quantark.asset.equity.engine.mc._mc_event_statistics import (
    FEW_KNOCK_IN_PATHS,
    NO_KNOCK_IN,
    KnockInLayout,
    knock_in_rows,
    ratio_standard_error,
    standard_errors_or_reason,
)
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.numerical import Tolerance

__all__ = ["DETERMINISTIC", "TERMINATED", "dcn_event_statistics", "terminated_dcn_statistics"]

#: ``unavailable["standard_errors"]`` of a route that simulates nothing (spec AC21: batch means only)
DETERMINISTIC = "undefined: nothing simulated (the final observation is decided at the valuation)"
TERMINATED = "undefined: nothing simulated (a terminated note reports its receivables only)"


def _provenance(engine_name: str) -> StatisticsProvenance:
    return StatisticsProvenance(statistics_model=engine_name, price_model=engine_name)


def _aggregate(records: Sequence[Mapping]) -> Dict[str, object]:
    """Sum of the batch records, key by key (every recorded value is an additive path sum)."""
    total: Dict[str, object] = {}
    for record in records:
        for key, value in record.items():
            if key == "kind":
                total[key] = value
            elif key in total:
                total[key] = total[key] + (np.asarray(value, dtype=float) if np.ndim(value) else float(value))
            else:
                total[key] = np.array(value, dtype=float) if np.ndim(value) else float(value)
    return total


class _DCNAssembly:
    """The DCN's fixed rows (axis, dates, amounts) for one (product, valuation), and the map from a batch's sums to
    every path-average field (linear in the sums, so the batch means give the standard errors)."""

    def __init__(self, product, pricing_env, ctx, receivables: ReceivablesTable) -> None:
        self.product, self.env, self.ctx = product, pricing_env, ctx
        self.sign = float(product.direction_sign)
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
        # the kernel's leg amounts (``dcn_payoff.compute_dcn_cashflows``), undiscounted: zero off eligible rows
        self.ko_amount = part * product.ko_coupon_rate * np.asarray(ctx.ko_accruals, dtype=float) * notional
        self.coupon_amount = part * product.coupon_rate * np.asarray(ctx.coupon_accruals, dtype=float) * notional
        self.df_loss = float(make_df_fn(pricing_env)(ctx.loss_pay_time))
        self.receivables = receivables
        self.receivable_cash = float(np.sum(receivables.discounted_cash))
        self.receivable_cash_to_pay = float(np.sum(receivables.cash_to_pay))
        # knock-in rows: the daily KI dates on the grid (column 0 only when the valuation close is one)
        self.first_ki_col = 0 if ctx.valuation_monitored else 1
        self.carried = bool(product.knocked_in_at_valuation)
        self.knocked_in_now = (not self.carried and bool(ctx.valuation_monitored)
                               and float(pricing_env.spot) <= float(product.ki_barrier))
        if self.carried:
            self.layout = KnockInLayout("carried", np.zeros(0), None)
        else:
            row_times = np.asarray(ctx.times[self.first_ki_col:], dtype=float)
            self.layout = KnockInLayout("discrete", row_times, ki_row_dates(ctx, self.first_ki_col))

    def _ki_time_sum(self, sums: Mapping) -> float:
        """Sum of the first knock-in times (carried or at the valuation close: 0) on the DCN axis."""
        return float(np.dot(np.asarray(sums["ki_first_count"], dtype=float), np.asarray(self.ctx.times, dtype=float)))

    def fields(self, sums: Mapping) -> Dict[str, object]:
        n = float(sums["n"])
        s = self.sign
        ko_count = np.asarray(sums["ko_count"], dtype=float)
        coupon_count = np.asarray(sums["coupon_count"], dtype=float)
        probability = ko_count / n
        survival = 1.0 - np.cumsum(probability)
        fields: Dict[str, object] = {
            "observations.probability": probability,
            "observations.survival": survival,
            "observations.discounted_cash": s * np.asarray(sums["ko_disc_cash"], dtype=float) / n,
            "observations.cash_to_pay": s * self.ko_amount * ko_count / n,
            "coupons.probability": coupon_count / n,
            "coupons.survival": np.concatenate([[1.0], survival[:-1]]),
            "coupons.discounted_cash": s * np.asarray(sums["fixed_disc_cash_by_period"], dtype=float) / n,
            "coupons.cash_to_pay": s * self.coupon_amount * coupon_count / n,
            "maturity.probability": float(sums["survive_no_ki_count"]) / n,
            "maturity.discounted_cash": 0.0,
            "maturity.cash_to_pay": 0.0,
            "ki_settlement.probability": float(sums["survive_ki_count"]) / n,
            "ki_settlement.discounted_cash": s * float(sums["loss_disc_cash"]) / n,
        }
        # the loss leg settles on one date: its cash to pay is the discounted cash over that date's DF
        fields["ki_settlement.cash_to_pay"] = fields["ki_settlement.discounted_cash"] / self.df_loss
        fields["knock_in.probability"] = float(sums["ki_count"]) / n
        fields["knock_in.probability_no_ko"] = fields["ki_settlement.probability"]
        # buyer-view loss leg <= 0: the knocked-in downside's size, whoever holds it (DESIGN 4.5, 4.10)
        fields["knock_in.expected_loss"] = -float(sums["loss_disc_cash"]) / n
        knocked_out_now = bool(self.times.size) and self.times[0] <= Tolerance.ZERO and ko_count[0] == n
        times, dates, ki_probability, ki_survival, interpolated = knock_in_rows(
            self.layout, n=n,
            first_counts=np.asarray(sums["ki_first_count"], dtype=float)[self.first_ki_col:],
            knocked_in_now=self.knocked_in_now, knocked_out_now=knocked_out_now,
            knock_in_count=float(sums["ki_count"]),
            clean_ko_times=self.times, clean_ko_counts=np.asarray(sums["ko_clean_count"], dtype=float),
            valuation_dates=at_valuation_dates(self.product, self.env),
        )
        fields["ki_table.probability"] = ki_probability
        fields["ki_table.survival"] = ki_survival
        fields["_ki_rows"] = (times, dates, interpolated)
        fields["expected_lifetime"] = float(sums["life_sum"]) / n
        fields["total_discounted_cash"] = (
            float(np.sum(fields["observations.discounted_cash"])) + float(np.sum(fields["coupons.discounted_cash"]))
            + fields["maturity.discounted_cash"] + fields["ki_settlement.discounted_cash"] + self.receivable_cash
        )
        fields["total_cash_to_pay"] = (
            float(np.sum(fields["observations.cash_to_pay"])) + float(np.sum(fields["coupons.cash_to_pay"]))
            + fields["maturity.cash_to_pay"] + fields["ki_settlement.cash_to_pay"] + self.receivable_cash_to_pay
        )
        return fields

    def build(self, *, pv: float, records: Sequence[Mapping], engine_name: str,
              se_reason: Optional[str]) -> EventStatistics:
        totals = _aggregate(records)
        fields = self.fields(totals)
        ctx, n = self.ctx, self.times.size
        observations = ObservationTable(
            times=self.times, payment_times=np.asarray(ctx.ko_pay_times, dtype=float), dates=self.dates,
            payment_dates=self.ko_payment_dates, is_ko_obs=ctx.obs_is_ko, is_coupon_obs=ctx.obs_is_coupon,
            probability=fields["observations.probability"], survival=fields["observations.survival"],
            discounted_cash=fields["observations.discounted_cash"], cash_to_pay=fields["observations.cash_to_pay"],
            payment_is_path_dependent=np.zeros(n, dtype=bool),
        )
        coupons = CouponTable(
            probability=fields["coupons.probability"], survival=fields["coupons.survival"],
            payment_times=np.asarray(ctx.coupon_pay_times, dtype=float), payment_dates=self.coupon_payment_dates,
            discounted_cash=fields["coupons.discounted_cash"], cash_to_pay=fields["coupons.cash_to_pay"],
            payment_is_path_dependent=np.zeros(n, dtype=bool),
        )
        timing = dict(time=float(self.times[-1]), payment_time=float(ctx.loss_pay_time), date=self.dates[-1],
                      payment_date=self.loss_payment_date)
        maturity = OutcomeRow(probability=fields["maturity.probability"],
                              discounted_cash=fields["maturity.discounted_cash"],
                              cash_to_pay=fields["maturity.cash_to_pay"], **timing)
        ki_settlement = OutcomeRow(probability=fields["ki_settlement.probability"],
                                   discounted_cash=fields["ki_settlement.discounted_cash"],
                                   cash_to_pay=fields["ki_settlement.cash_to_pay"], **timing)
        unavailable: Dict[str, str] = {}
        knocked_in = float(totals["ki_count"])
        expected_time = self._ki_time_sum(totals) / knocked_in if knocked_in > 0.0 else None
        if expected_time is None:
            unavailable["knock_in.expected_time"] = NO_KNOCK_IN
        knock_in = KnockInStatistics(
            probability=fields["knock_in.probability"], probability_no_ko=ki_settlement.probability,
            expected_time=expected_time, expected_loss=fields["knock_in.expected_loss"],
        )
        times, dates, interpolated = fields["_ki_rows"]
        ki_table = KnockInTable(
            times=times, payment_times=times, dates=dates, payment_dates=dates,
            probability=fields["ki_table.probability"], survival=fields["ki_table.survival"],
            interpolated=interpolated,
        )
        errors = None
        if se_reason is None:
            estimate = {key: value for key, value in fields.items() if not key.startswith("_")}
            errors, se_reason = standard_errors_or_reason(
                records, independent=True,
                fields_of=lambda record: {k: v for k, v in self.fields(record).items() if not k.startswith("_")},
                estimate=estimate,
            )
        if se_reason is not None:
            unavailable["standard_errors"] = se_reason
        elif expected_time is not None and knocked_in >= 2.0:
            errors["knock_in.expected_time"] = ratio_standard_error(
                [self._ki_time_sum(record) for record in records], [float(record["ki_count"]) for record in records])
        else:
            unavailable["standard_errors.knock_in.expected_time"] = FEW_KNOCK_IN_PATHS
        ledger = build_cash_ledger(observations=observations, coupons=coupons, maturity=maturity,
                                   ki_settlement=ki_settlement, receivables=self.receivables)
        return EventStatistics(
            product_family="dcn", pv=float(pv), observations=observations, coupons=coupons, maturity=maturity,
            ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table, receivables=self.receivables,
            ledger=ledger, expected_lifetime=fields["expected_lifetime"], provenance=_provenance(engine_name),
            unavailable=unavailable, standard_errors=errors,
        )


def dcn_event_statistics(product, pricing_env, ctx, *, pv: float, records: Sequence[Mapping], engine_name: str,
                         state=None, deterministic: bool = False) -> EventStatistics:
    """The DCN statistics of one priced run.

    Args:
        product, pricing_env, ctx: the priced DCN, its environment and its grid context.
        pv: the value ``price()`` returned (the receivables included).
        records: the run's batch records, in batch order (``dcn_mc_engine._dcn_batch_record``).
        engine_name: the engine class name (provenance).
        state: the run's validated ``DCNStateValuation`` (receivables), or ``None``.
        deterministic: the run simulated nothing (the final observation decided at the valuation): no batch-mean
            standard error exists.
    """
    receivables = ReceivablesTable.empty() if state is None else state.receivables_table(
        product.schedule.valuation_date)
    return _DCNAssembly(product, pricing_env, ctx, receivables).build(
        pv=pv, records=records, engine_name=engine_name, se_reason=DETERMINISTIC if deterministic else None)


def terminated_dcn_statistics(product, state, *, pv: float, engine_name: str) -> Optional[EventStatistics]:
    """A knocked-out or matured lifecycle state (spec AC17, DESIGN 4.9c): its receivables only, or ``None`` when
    nothing is left unpaid -- the engine-neutral ``dcn_statistics.terminated_dcn_statistics`` (shared with the DCN PDE
    engines), with the MC's standard-error reason (nothing simulated)."""
    from quantark.asset.equity.engine.dcn_statistics import terminated_dcn_statistics as terminated

    return terminated(product, state, pv=pv, provenance=_provenance(engine_name), standard_errors_reason=TERMINATED)
