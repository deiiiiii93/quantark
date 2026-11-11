"""``EventStatistics`` of the Phoenix MC engines (unified event statistics spec §Contracts, T10).

``PhoenixMCEngine.calculate_event_stats`` (and every vol-model Phoenix subclass) runs ``price()`` under the
statistics recorder; every priced batch deposits one record of outcome sums (``_event_stats_recorder``). This
module turns those records into the one result type, sharing the Snowball assembly's knock-in / standard-error
tail (``_snowball_event_statistics._Assembly``):

- the observation table: one row per Phoenix observation (the KO and coupon observations are one grid; a row
  without a knock-out barrier is a coupon-only row), KO cash = the knock-out redemption only (DESIGN §4.2);
- the coupon table on the same rows (DESIGN §4.3): P(alive just before the row and the coupon barrier met), the
  coupon cash the payoff loop pays there -- the current coupon and any memory arrears it releases, including
  arrears released by a knock-out on a missed coupon and ``initial_coupon_arrears`` owed at the pricing date --
  discounted where the price discounts it. INSTANT: paid at the row's settlement (the knock-out's own
  settlement for a coupon paid with a knock-out). EXPIRY: rolled up and paid when the note ends, so every coupon
  row is path-dependent and the cash ledger buckets it at the actual ending event (each knock-out row's
  settlement, else the terminal payment), undiscounted cash summed per path (DESIGN §4.6);
- the maturity row (never KO, never KI) and the KI settlement row (KI, no KO): the redemption only, the rolled-up
  coupons being the coupon table's;
- knock-in before any knock-out only (DESIGN §4.5), the per-date knock-in table (discrete: the product's KI rows;
  continuous: trading days by step overlap, spec D18), the expected knocked-in loss;
- standard errors from independent batch means (spec AC21).

``pv`` is the value ``price()`` returned (D23); ``reconciliation_error`` and ``probability_residual`` are the type's
own computation, never absorbed.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from quantark.asset.equity.engine.event_stats import (
    CashLedger,
    CouponTable,
    EventStatistics,
    ObservationTable,
    OutcomeRow,
    ReceivablesTable,
    build_cash_ledger,
)
from quantark.asset.equity.engine.mc._snowball_event_statistics import (
    _all_dates,
    _Assembly,
    _knocked_out_now,
)
from quantark.util.enum import CouponPayType, ObservationType

__all__ = ["phoenix_event_statistics"]


class _PhoenixAssembly(_Assembly):
    family = "phoenix"

    def __init__(self, engine, product, pricing_env, totals: Mapping) -> None:
        super().__init__(engine, product, pricing_env)
        profile = product.get_ko_observation_profile(pricing_env)
        self.times = np.asarray(profile["observation_times"], dtype=float)
        barriers = np.asarray(
            [np.nan if barrier is None else float(barrier) for barrier in profile["barriers"]], dtype=float
        )
        self.ko_payoffs = np.asarray(profile["payoffs"], dtype=float)
        n = self.times.size
        # a row with no knock-out level is a coupon-only observation (the kernel never knocks out there)
        self.is_ko_obs = np.isfinite(barriers)
        self.is_coupon_obs = np.ones(n, dtype=bool)
        records = product.resolve_ko_observations(pricing_env)
        timings = engine._payment_timings
        self.terminal = timings.terminal
        # a knock-out redemption settles on its record's settlement whatever the coupon convention
        self.ko_payment_times = np.asarray(timings.observation_payment_times, dtype=float)
        self.dates = _all_dates(record.observation_date for record in records)
        self.ko_payment_dates = _all_dates(record.settlement_date for record in records)
        self.instant = product.coupon_config.coupon_pay_type == CouponPayType.INSTANT
        if self.instant:
            self.coupon_payment_times = self.ko_payment_times
            self.coupon_payment_dates = self.ko_payment_dates
        else:
            # EXPIRY: no single payment time; the row carries the latest one (the terminal payment) as its
            # placeholder and the ledger holds the cash at the actual ending events
            self.coupon_payment_times = np.full(n, float(self.terminal.payment_time))
            self.coupon_payment_dates = (
                None if self.terminal.payment_date is None else (self.terminal.payment_date,) * n
            )
        continuous = product.has_ki_barrier and (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )
        discrete_dates = None
        if product.has_ki_barrier and not continuous:
            discrete_dates = _all_dates(
                record.observation_date for record in product.resolve_ki_observations(pricing_env)
            )
        self.layout = self.knock_in_layout(
            totals["ki_row_times"], continuous,
            bool(getattr(product, "_otc_lifecycle_knocked_in", False)) and product.has_ki_barrier,
            float(product.get_maturity(pricing_env)), discrete_dates,
        )

    def fields(self, sums: Mapping) -> Dict[str, object]:
        n = float(sums["n"])
        ko_count = np.asarray(sums["ko_count"], dtype=float)
        probability = ko_count / n
        survival = 1.0 - np.cumsum(probability)
        fields: Dict[str, object] = {
            "observations.probability": probability,
            "observations.survival": survival,
            "observations.discounted_cash": np.asarray(sums["ko_disc_cash"], dtype=float) / n,
            "observations.cash_to_pay": np.asarray(sums["ko_cash"], dtype=float) / n,
            "coupons.probability": np.asarray(sums["coupon_count"], dtype=float) / n,
            "coupons.survival": np.concatenate([[1.0], survival[:-1]]) if survival.size else survival,
            "coupons.discounted_cash": np.asarray(sums["coupon_disc_cash"], dtype=float) / n,
            "coupons.cash_to_pay": np.asarray(sums["coupon_cash"], dtype=float) / n,
            "maturity.probability": float(sums["v0_count"]) / n,
            "maturity.discounted_cash": float(sums["v0_disc_cash"]) / n,
            "maturity.cash_to_pay": float(sums["v0_cash"]) / n,
            "ki_settlement.probability": float(sums["v1_count"]) / n,
            "ki_settlement.discounted_cash": float(sums["v1_disc_cash"]) / n,
            "ki_settlement.cash_to_pay": float(sums["v1_cash"]) / n,
        }
        fields["knock_in.probability_no_ko"] = fields["ki_settlement.probability"]
        fields.update(self.knock_in_fields(
            sums, self.layout, knocked_out_now=_knocked_out_now(self.times, ko_count, n),
            clean_ko_times=self.times, clean_ko_counts=np.asarray(sums["ko_clean_count"], dtype=float),
        ))
        terminal_mass = fields["maturity.probability"] + fields["ki_settlement.probability"]
        fields["expected_lifetime"] = float(np.sum(probability * self.times)) + terminal_mass * float(
            self.terminal.determination_time)
        fields["total_discounted_cash"] = (
            float(np.sum(fields["observations.discounted_cash"])) + float(np.sum(fields["coupons.discounted_cash"]))
            + fields["maturity.discounted_cash"] + fields["ki_settlement.discounted_cash"]
        )
        fields["total_cash_to_pay"] = (
            float(np.sum(fields["observations.cash_to_pay"])) + float(np.sum(fields["coupons.cash_to_pay"]))
            + fields["maturity.cash_to_pay"] + fields["ki_settlement.cash_to_pay"]
        )
        return fields

    def _expiry_coupon_ledger(self, sums: Mapping) -> Optional[CashLedger]:
        """EXPIRY coupons at their actual payment events: each knock-out row's settlement, then the terminal one
        (one bucket per ending event, zeros kept; the recorded sums are per path end)."""
        if self.instant:
            return None
        n = float(sums["n"])
        terminal = self.terminal
        determination = np.concatenate([self.times, [float(terminal.determination_time)]])
        payment = np.concatenate([self.ko_payment_times, [float(terminal.payment_time)]])
        determination_dates = (
            None if self.dates is None or terminal.determination_date is None
            else self.dates + (terminal.determination_date,)
        )
        payment_dates = (
            None if self.ko_payment_dates is None or terminal.payment_date is None
            else self.ko_payment_dates + (terminal.payment_date,)
        )
        return CashLedger(
            determination_times=determination, payment_times=payment, determination_dates=determination_dates,
            payment_dates=payment_dates,
            discounted_cash=np.asarray(sums["coupon_end_disc_cash"], dtype=float) / n,
            cash_to_pay=np.asarray(sums["coupon_end_cash"], dtype=float) / n,
        )

    def build(self, pv: float, totals: Mapping, records) -> EventStatistics:
        fields = self.fields(totals)
        n = self.times.size
        observations = ObservationTable(
            times=self.times, payment_times=self.ko_payment_times, dates=self.dates,
            payment_dates=self.ko_payment_dates, is_ko_obs=self.is_ko_obs, is_coupon_obs=self.is_coupon_obs,
            probability=fields["observations.probability"], survival=fields["observations.survival"],
            discounted_cash=fields["observations.discounted_cash"], cash_to_pay=fields["observations.cash_to_pay"],
            payment_is_path_dependent=np.zeros(n, dtype=bool),
        )
        coupons = CouponTable(
            probability=fields["coupons.probability"], survival=fields["coupons.survival"],
            payment_times=self.coupon_payment_times, payment_dates=self.coupon_payment_dates,
            discounted_cash=fields["coupons.discounted_cash"], cash_to_pay=fields["coupons.cash_to_pay"],
            payment_is_path_dependent=np.full(n, not self.instant),
        )
        terminal = self.terminal
        timing = dict(time=float(terminal.determination_time), payment_time=float(terminal.payment_time),
                      date=terminal.determination_date, payment_date=terminal.payment_date)
        maturity = OutcomeRow(probability=fields["maturity.probability"],
                              discounted_cash=fields["maturity.discounted_cash"],
                              cash_to_pay=fields["maturity.cash_to_pay"], **timing)
        ki_settlement = OutcomeRow(probability=fields["ki_settlement.probability"],
                                   discounted_cash=fields["ki_settlement.discounted_cash"],
                                   cash_to_pay=fields["ki_settlement.cash_to_pay"], **timing)
        ledger = build_cash_ledger(observations=observations, coupons=coupons, maturity=maturity,
                                   ki_settlement=ki_settlement, receivables=ReceivablesTable.empty(),
                                   path_dependent_cash=self._expiry_coupon_ledger(totals))
        return self.finish(pv=pv, totals=totals, records=records, fields=fields, tables=dict(
            observations=observations, coupons=coupons, maturity=maturity, ki_settlement=ki_settlement,
            ledger=ledger))


def phoenix_event_statistics(engine, product, pricing_env, pv: float, totals: Mapping, records) -> EventStatistics:
    """The Phoenix statistics of one recorded ``price()`` (``totals``: the aggregate of ``records``)."""
    return _PhoenixAssembly(engine, product, pricing_env, totals).build(pv, totals, records)
