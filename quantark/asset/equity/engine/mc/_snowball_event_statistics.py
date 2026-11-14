"""``EventStatistics`` of the Snowball and KO-reset MC engines (unified event statistics spec §Contracts, T9).

``SnowballMCEngine.calculate_event_stats`` runs ``price()`` under the statistics recorder; every priced batch
deposits one record of outcome sums (``_event_stats_recorder``). This module turns those records into the one
result type:

- every outcome directly from the paths (DESIGN §4.7): KO rows, the maturity row (never KO, never KI), the KI
  settlement row (KI, no KO), each with its discounted cash and its undiscounted cash to pay;
- knock-in before any knock-out only (DESIGN §4.5): P(knock in), the conditional mean first-knock-in time, the
  expected knocked-in loss and the per-date knock-in table (discrete: the product's KI rows, exact; continuous:
  one row per trading day by step overlap, spec D18);
- KO-reset ABSOLUTE: both schedules on one table tagged ``pre``/``post``, survival with the F5 pre-maturity drop;
- KO-reset REBASED (spec D13): the post-KI offset table and the absolute termination distribution on the full
  deterministic candidate axis (every reachable knock-in date + each offset, plus maturity), zeros kept;
- standard errors from independent batch means (spec AC21).

The ``pv`` is the value ``price()`` returned (D23); ``reconciliation_error`` and ``probability_residual`` are the
type's own computation, never absorbed.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.knock_in_days import at_valuation_dates
from quantark.asset.equity.engine.event_stats import (
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    ObservationTable,
    OutcomeRow,
    PostKIOffsetTable,
    ReceivablesTable,
    StatisticsProvenance,
    TerminationDistribution,
    build_cash_ledger,
    build_rebased_cash_ledger,
)
from quantark.asset.equity.engine.mc._mc_event_statistics import (
    FEW_KNOCK_IN_PATHS,
    NO_KNOCK_IN,
    STEP_OVERLAP,
    KnockInLayout,
    knock_in_day_axis,
    knock_in_rows,
    ratio_standard_error,
    standard_errors_or_reason,
)
from quantark.asset.equity.settlement import CashflowKind
from quantark.util.enum import CouponPayType, ObservationType, PostKOScheduleMode
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import PricingError
from quantark.util.numerical import Tolerance

__all__ = ["snowball_event_statistics", "ko_reset_event_statistics"]

_EVENT_RANK = {"ko_pre": 0, "ko_post": 1, "maturity": 2, "ki_settlement": 3}
_LEDGER_EVENT = {"post": "ko_post", "maturity": "maturity", "ki_settlement": "ki_settlement"}


def _all_dates(values) -> Optional[tuple]:
    values = tuple(values)
    return values if all(value is not None for value in values) else None


def _provenance(engine) -> StatisticsProvenance:
    name = type(engine).__name__
    return StatisticsProvenance(statistics_model=name, price_model=name)


def _independent_batches(engine) -> bool:
    """Batch means are independent draws on PSEUDO (distinct seeds) and RQMC (distinct scrambles); the
    deterministic Sobol batches of a QUASI Dask run are consecutive blocks of one sequence."""
    return engine.method != MonteCarloMethod.QUASI


def _knocked_out_now(times: np.ndarray, counts: np.ndarray, n: float) -> bool:
    """A knock-out decided at the valuation instant: an observation at t = 0 that every path takes."""
    at_zero = np.asarray(times, dtype=float) <= Tolerance.ZERO
    return bool(np.any(at_zero & (np.asarray(counts, dtype=float) == n)))


class _Assembly:
    """Shared tail: knock-in statistics, standard errors, ``unavailable`` and the result (Snowball, KO-reset and,
    with its coupon table in ``tables["coupons"]``, Phoenix: ``_phoenix_event_statistics``)."""

    family: str

    def __init__(self, engine, product, pricing_env) -> None:
        self.engine = engine
        self.product = product
        self.env = pricing_env

    # subclasses: ``fields(sums) -> {dotted path: value}`` (every path average, linear in the sums)

    def knock_in_layout(self, ki_row_times: np.ndarray, continuous: bool, carried: bool, horizon: float,
                        discrete_dates: Optional[tuple]) -> KnockInLayout:
        if carried:
            return KnockInLayout("carried", np.zeros(0), None)
        if not self.product.has_ki_barrier:
            return KnockInLayout("none", np.zeros(0), None)
        if continuous:
            step_ends = np.asarray(ki_row_times, dtype=float)
            window_end = float(step_ends[-1]) if step_ends.size else float(horizon)
            days, dates = knock_in_day_axis(self.product, self.env, window_end)
            return KnockInLayout("continuous", days, dates, step_ends=step_ends)
        rows = np.asarray(ki_row_times, dtype=float)
        dates = discrete_dates if discrete_dates is not None and len(discrete_dates) == rows.size else None
        return KnockInLayout("discrete", rows, dates)

    def knock_in_fields(self, sums: Mapping, layout: KnockInLayout, *, knocked_out_now: bool,
                        clean_ko_times: np.ndarray, clean_ko_counts: np.ndarray) -> Dict[str, object]:
        n = float(sums["n"])
        knocked_in = 0.0 if layout.kind == "none" else float(sums["ki_before_ko_count"])
        times, dates, probability, survival, interpolated = knock_in_rows(
            layout, n=n, first_counts=np.asarray(sums.get("ki_first_count", np.zeros(0)), dtype=float),
            knocked_in_now=bool(sums.get("ki_breached_at_valuation", False)),
            knocked_out_now=knocked_out_now, knock_in_count=knocked_in,
            clean_ko_times=clean_ko_times, clean_ko_counts=clean_ko_counts,
            valuation_dates=at_valuation_dates(self.product, self.env),
        )
        return {
            "knock_in.probability": knocked_in / n,
            "knock_in.expected_loss": float(sums["v1_disc_loss"]) / n,
            "ki_table.probability": probability,
            "ki_table.survival": survival,
            "_ki_rows": (times, dates, interpolated),
        }

    def finish(self, *, pv: float, totals: Mapping, records, fields: Dict[str, object], tables: dict) -> EventStatistics:
        unavailable: Dict[str, str] = {}
        knocked_in = float(totals["ki_before_ko_count"]) if fields["knock_in.probability"] > 0.0 else 0.0
        expected_time = float(totals["ki_time_sum"]) / knocked_in if knocked_in > 0.0 else None
        if expected_time is None:
            unavailable["knock_in.expected_time"] = NO_KNOCK_IN
        times, dates, interpolated = fields["_ki_rows"]
        ki_table = KnockInTable(
            times=times, payment_times=times, dates=dates, payment_dates=dates,
            probability=fields["ki_table.probability"], survival=fields["ki_table.survival"],
            interpolated=interpolated, interpolation_scheme=STEP_OVERLAP if np.any(interpolated) else None,
        )
        ki_settlement = tables["ki_settlement"]
        knock_in = KnockInStatistics(
            probability=fields["knock_in.probability"], probability_no_ko=ki_settlement.probability,
            expected_time=expected_time, expected_loss=fields["knock_in.expected_loss"],
        )
        estimate = {key: value for key, value in fields.items() if not key.startswith("_")}
        errors, reason = standard_errors_or_reason(
            records, independent=_independent_batches(self.engine),
            fields_of=lambda record: {k: v for k, v in self.fields(record).items() if not k.startswith("_")},
            estimate=estimate,
        )
        if reason is not None:
            unavailable["standard_errors"] = reason
        else:
            # the "fewer than two knock-in paths" gate (spec AC21): ``ki_before_ko_count`` is a path count on native
            # records and the summed knock-in probability mass on conditional / control-corrected records, so there
            # it compares the effective number of knock-in paths
            if expected_time is not None and knocked_in >= 2.0:
                errors["knock_in.expected_time"] = ratio_standard_error(
                    [float(r["ki_time_sum"]) for r in records], [float(r["ki_before_ko_count"]) for r in records])
            else:
                unavailable["standard_errors.knock_in.expected_time"] = FEW_KNOCK_IN_PATHS
        return EventStatistics(
            product_family=self.family, pv=float(pv), observations=tables["observations"],
            coupons=tables.get("coupons"),
            maturity=tables["maturity"], ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table,
            post_ki_offsets=tables.get("post_ki_offsets"), termination=tables.get("termination"),
            receivables=ReceivablesTable.empty(), ledger=tables["ledger"],
            expected_lifetime=fields["expected_lifetime"], provenance=_provenance(self.engine),
            unavailable=unavailable, standard_errors=errors,
        )


# --- Snowball -----------------------------------------------------------------------------------------------


class _SnowballAssembly(_Assembly):
    family = "snowball"

    def __init__(self, engine, product, pricing_env, totals: Mapping) -> None:
        super().__init__(engine, product, pricing_env)
        profile = product.get_ko_observation_profile(pricing_env)
        self.ko_times = np.asarray(profile["observation_times"], dtype=float)
        records = product.resolve_ko_observations(pricing_env)
        timings = engine._payment_timings
        self.terminal = timings.terminal
        instant = product.accrual_config.coupon_pay_type == CouponPayType.INSTANT
        # a knock-out pays at its record's settlement under INSTANT, with the terminal payment under EXPIRY
        self.ko_payment_times = (
            np.asarray(timings.observation_payment_times, dtype=float) if instant
            else np.full(self.ko_times.size, float(self.terminal.payment_time))
        )
        self.ko_dates = _all_dates(record.observation_date for record in records)
        self.ko_payment_dates = _all_dates(
            (record.settlement_date if instant else self.terminal.payment_date) for record in records
        )
        continuous = product.has_ki_barrier and (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )
        discrete_dates = None
        if product.has_ki_barrier and not continuous:
            discrete_dates = _all_dates(record.observation_date for record in product.resolve_ki_observations(pricing_env))
        self.layout = self.knock_in_layout(
            totals["ki_row_times"], continuous,
            bool(getattr(product, "_otc_lifecycle_knocked_in", False)),
            float(product.get_maturity(pricing_env)), discrete_dates,
        )

    def fields(self, sums: Mapping) -> Dict[str, object]:
        n = float(sums["n"])
        ko_count = np.asarray(sums["ko_count"], dtype=float)
        probability = ko_count / n
        fields = {
            "observations.probability": probability,
            "observations.survival": 1.0 - np.cumsum(probability),
            "observations.discounted_cash": np.asarray(sums["ko_disc_cash"], dtype=float) / n,
            "observations.cash_to_pay": np.asarray(sums["ko_cash"], dtype=float) / n,
            "maturity.probability": float(sums["v0_count"]) / n,
            "maturity.discounted_cash": float(sums["v0_disc_cash"]) / n,
            "maturity.cash_to_pay": float(sums["v0_cash"]) / n,
            "ki_settlement.probability": float(sums["v1_count"]) / n,
            "ki_settlement.discounted_cash": float(sums["v1_disc_cash"]) / n,
            "ki_settlement.cash_to_pay": float(sums["v1_cash"]) / n,
        }
        fields["knock_in.probability_no_ko"] = fields["ki_settlement.probability"]
        fields.update(self.knock_in_fields(
            sums, self.layout, knocked_out_now=_knocked_out_now(self.ko_times, ko_count, n),
            clean_ko_times=self.ko_times, clean_ko_counts=np.asarray(sums["ko_clean_count"], dtype=float),
        ))
        terminal_mass = fields["maturity.probability"] + fields["ki_settlement.probability"]
        fields["expected_lifetime"] = float(np.sum(probability * self.ko_times)) + terminal_mass * float(
            self.terminal.determination_time)
        fields["total_discounted_cash"] = float(np.sum(fields["observations.discounted_cash"])) + fields[
            "maturity.discounted_cash"] + fields["ki_settlement.discounted_cash"]
        fields["total_cash_to_pay"] = float(np.sum(fields["observations.cash_to_pay"])) + fields[
            "maturity.cash_to_pay"] + fields["ki_settlement.cash_to_pay"]
        return fields

    def build(self, pv: float, totals: Mapping, records) -> EventStatistics:
        fields = self.fields(totals)
        n = self.ko_times.size
        observations = ObservationTable(
            times=self.ko_times, payment_times=self.ko_payment_times, dates=self.ko_dates,
            payment_dates=self.ko_payment_dates, is_ko_obs=np.ones(n, dtype=bool),
            is_coupon_obs=np.zeros(n, dtype=bool), probability=fields["observations.probability"],
            survival=fields["observations.survival"], discounted_cash=fields["observations.discounted_cash"],
            cash_to_pay=fields["observations.cash_to_pay"], payment_is_path_dependent=np.zeros(n, dtype=bool),
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
        ledger = build_cash_ledger(observations=observations, coupons=None, maturity=maturity,
                                   ki_settlement=ki_settlement, receivables=ReceivablesTable.empty())
        return self.finish(pv=pv, totals=totals, records=records, fields=fields, tables=dict(
            observations=observations, maturity=maturity, ki_settlement=ki_settlement, ledger=ledger))


def snowball_event_statistics(engine, product, pricing_env, pv: float, totals: Mapping, records) -> EventStatistics:
    """The Snowball statistics of one recorded ``price()`` (``totals``: the aggregate of ``records``)."""
    return _SnowballAssembly(engine, product, pricing_env, totals).build(pv, totals, records)


# --- KO-reset -----------------------------------------------------------------------------------------------


class _KOResetAssembly(_Assembly):
    family = "ko_reset"

    def __init__(self, engine, product, pricing_env, totals: Mapping) -> None:
        super().__init__(engine, product, pricing_env)
        maturity = float(product.get_max_maturity_time(pricing_env))
        grid = engine._build_time_grid_ko_reset(product, pricing_env, maturity)
        self.grid = grid
        self.rebased = grid["post_profile"]["mode"] != PostKOScheduleMode.ABSOLUTE
        self.carried = bool(grid["carried_knock_in"])
        pre_profile, post_profile = grid["pre_profile"], grid["post_profile"]
        self.pre_times = np.asarray(grid["pre_times"], dtype=float)
        self.post_times = np.asarray(grid["post_times"], dtype=float)
        self.pre_maturity = float(product.get_pre_maturity_time(pricing_env))
        _, self.pre_settlement = engine._compute_ko_schedule_payoffs(
            product, self.pre_times, np.asarray(pre_profile["rates"], dtype=float), pre_profile["records"],
            pricing_env, self.pre_maturity,
        )
        self.pre_dates = _all_dates(record.observation_date for record in pre_profile["records"])
        self.pre_payment_dates = self._schedule_payment_dates(self.pre_times, pre_profile["records"],
                                                              self.pre_maturity)
        self.v0_timing = engine._resolve_payment_timing(
            product, pricing_env, self.pre_maturity, CashflowKind.TERMINAL, cashflow_id="ko_reset_v0_terminal")
        if not self.rebased:
            self.post_maturity = float(product.get_post_maturity_time(pricing_env))
            _, self.post_settlement = engine._compute_ko_schedule_payoffs(
                product, self.post_times, np.asarray(post_profile["rates"], dtype=float), post_profile["records"],
                pricing_env, self.post_maturity,
            )
            self.post_dates = _all_dates(record.observation_date for record in post_profile["records"])
            self.post_payment_dates = self._schedule_payment_dates(self.post_times, post_profile["records"],
                                                                   self.post_maturity)
            self.v1_timing = engine._resolve_payment_timing(
                product, pricing_env, self.post_maturity, CashflowKind.TERMINAL,
                cashflow_id="ko_reset_v1_terminal")
            self._absolute_axis()
        else:
            self._rebased_axis()
        ki_records = grid["ki_records"]
        discrete_dates = _all_dates(record.observation_date for record in ki_records) if ki_records else None
        self.layout = self.knock_in_layout(
            totals["ki_row_times"], bool(grid["ki_continuous"]), self.carried, self.pre_maturity, discrete_dates)

    def _schedule_payment_dates(self, times: np.ndarray, records, maturity: float) -> Optional[tuple]:
        """Settlement dates of a schedule's knock-outs, as ``_compute_ko_schedule_payoffs`` settles them."""
        engine, product, env = self.engine, self.product, self.env
        if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:
            timings = [engine._resolve_payment_timing(product, env, float(t), CashflowKind.REDEMPTION,
                                                      record=records[i], cashflow_id=f"ko_reset[{i}]")
                       for i, t in enumerate(times)]
        else:
            terminal = engine._resolve_payment_timing(product, env, float(maturity), CashflowKind.TERMINAL,
                                                      cashflow_id="ko_reset_terminal")
            timings = [terminal] * len(times)
        return _all_dates(timing.payment_date for timing in timings)

    # ABSOLUTE: both schedules on one table, the pre record first at a tie

    def _absolute_axis(self) -> None:
        n_pre, n_post = self.pre_times.size, self.post_times.size
        times = np.concatenate([self.pre_times, self.post_times])
        tags = np.concatenate([np.zeros(n_pre), np.ones(n_post)])
        self.order = np.lexsort((tags, times))
        self.obs_times = times[self.order]
        self.obs_tags = tuple("pre" if tags[i] == 0 else "post" for i in self.order)
        self.obs_payment = np.concatenate([self.pre_settlement, self.post_settlement])[self.order]
        dates = None if self.pre_dates is None or self.post_dates is None else self.pre_dates + self.post_dates
        self.obs_dates = None if dates is None else tuple(dates[i] for i in self.order)
        pay_dates = (None if self.pre_payment_dates is None or self.post_payment_dates is None
                     else self.pre_payment_dates + self.post_payment_dates)
        self.obs_payment_dates = None if pay_dates is None else tuple(pay_dates[i] for i in self.order)
        # F5: the pre-maturity mass leaves the survival at its own determination when that precedes the last
        # knock-out row (the post schedule runs on)
        self.f5 = bool(self.obs_times.size) and self.pre_maturity < float(self.obs_times[-1]) - Tolerance.ZERO

    # REBASED: the deterministic candidate axis (spec D13)

    def _rebased_axis(self) -> None:
        engine, product, env, grid = self.engine, self.product, self.env, self.grid
        post_profile = grid["post_profile"]
        post_records = post_profile["records"]
        n_post = self.post_times.size
        post_last = int(np.argmax(self.post_times)) if n_post else -1
        ki_times = np.asarray(grid["ki_times"], dtype=float)
        by_ki = grid["post_times_by_ki"] or []
        instant = product.accrual_config.coupon_pay_type == CouponPayType.INSTANT
        dated = bool(getattr(product, "_dated_rebased", lambda: False)())
        extended = product.rebased_extended_schedule if dated else {}
        source = list(grid["ki_record_indices"])
        rows: Dict[tuple, Tuple[Optional[object], Optional[object]]] = {}

        def add(event, det, timing, det_date):
            key = (event, float(det), float(timing.payment_time))
            if key not in rows:
                rows[key] = (det_date, timing.payment_date)

        for i, (t, pay) in enumerate(zip(self.pre_times, self.pre_settlement)):
            rows.setdefault(("ko_pre", float(t), float(pay)),
                            (None if self.pre_dates is None else self.pre_dates[i],
                             None if self.pre_payment_dates is None else self.pre_payment_dates[i]))
        allow_post = not product.barrier_config.disable_ko_after_ki
        for k in range(ki_times.size):
            post_k = np.asarray(by_ki[k], dtype=float) if n_post else np.zeros(0)
            post_dates = extended.get(source[k]) if dated and k < len(source) else None
            end = float(post_k[post_last]) if n_post else float(ki_times[k])
            end_date = post_dates[post_last] if post_dates else None
            if allow_post:
                for j in range(n_post):
                    if instant:
                        timing = engine._resolve_payment_timing(
                            product, env, float(post_k[j]), CashflowKind.REDEMPTION, record=post_records[j],
                            cashflow_id=f"ko_reset_rebased[{j}]")
                    else:
                        timing = engine._resolve_payment_timing(
                            product, env, end, CashflowKind.TERMINAL, cashflow_id="ko_reset_rebased_terminal")
                    add("ko_post", float(post_k[j]), timing, post_dates[j] if post_dates else None)
            add("ki_settlement", end, engine._resolve_payment_timing(
                product, env, end, CashflowKind.TERMINAL, cashflow_id="ko_reset_v1_rebased_terminal"), end_date)
        add("maturity", self.pre_maturity, self.v0_timing,
            self.pre_dates[int(np.argmax(self.pre_times))] if self.pre_dates else None)
        keys = sorted(rows, key=lambda key: (key[1], _EVENT_RANK[key[0]], key[2]))
        self.term_keys = keys
        self.term_index = {key: i for i, key in enumerate(keys)}
        self.term_times = np.array([key[1] for key in keys], dtype=float)
        self.term_payment = np.array([key[2] for key in keys], dtype=float)
        self.term_event = tuple(key[0] for key in keys)
        self.term_dates = _all_dates(rows[key][0] for key in keys)
        self.term_payment_dates = _all_dates(rows[key][1] for key in keys)
        self.survival_times = np.unique(self.term_times)
        self.pre_rows = [self.term_index[("ko_pre", float(t), float(p))]
                         for t, p in zip(self.pre_times, self.pre_settlement)]
        ki_rows = [i for i, key in enumerate(keys) if key[0] == "ki_settlement"]
        last = ki_rows[int(np.argmax(self.term_times[ki_rows]))] if ki_rows else None
        self.ki_settlement_row = last

    def _termination_columns(self, sums: Mapping) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = len(self.term_keys)
        columns = np.zeros((3, m), dtype=float)          # count, discounted, cash
        for i, row in enumerate(self.pre_rows):
            columns[0, row] += float(sums["pre_count"][i])
            columns[1, row] += float(sums["pre_disc_cash"][i])
            columns[2, row] += float(sums["pre_cash"][i])
        for (tag, det, pay), (discounted, cash, count) in sums.get("ledger", {}).items():
            row = self.term_index.get((_LEDGER_EVENT[tag], det, pay))
            if row is None:
                raise PricingError(
                    f"REBASED outcome {tag} at t={det!r} paid t={pay!r} is off the deterministic candidate axis"
                )
            columns[0, row] += count
            columns[1, row] += discounted
            columns[2, row] += cash
        return columns[0], columns[1], columns[2]

    def fields(self, sums: Mapping) -> Dict[str, object]:
        n = float(sums["n"])
        pre_count = np.asarray(sums["pre_count"], dtype=float)
        post_count = np.asarray(sums["post_count"], dtype=float)
        fields: Dict[str, object] = {
            "maturity.probability": float(sums["v0_count"]) / n,
            "maturity.discounted_cash": float(sums["v0_disc_cash"]) / n,
            "maturity.cash_to_pay": float(sums["v0_cash"]) / n,
            "ki_settlement.probability": float(sums["v1_count"]) / n,
            "ki_settlement.discounted_cash": float(sums["v1_disc_cash"]) / n,
            "ki_settlement.cash_to_pay": float(sums["v1_cash"]) / n,
        }
        fields["knock_in.probability_no_ko"] = fields["ki_settlement.probability"]
        knocked_out_now = _knocked_out_now(self.pre_times, pre_count, n) or (
            not self.rebased and _knocked_out_now(self.post_times, post_count, n))
        if self.rebased:
            count, discounted, cash = self._termination_columns(sums)
            probability = count / n
            fields["termination.probability"] = probability
            fields["termination.discounted_cash"] = discounted / n
            fields["termination.cash_to_pay"] = cash / n
            # survival by spec D29 (as the ABSOLUTE layout's distribution reads it): a knock-out leaves at its own
            # row; a maturity-type termination (maturity, knocked-in settlement) is alive at its determination and
            # gone at every later node
            events = np.asarray(self.term_event, dtype=object)
            knocks_out = np.isin(events, ("ko_pre", "ko_post"))
            ko_by_time = np.array([probability[(self.term_times == t) & knocks_out].sum() for t in self.survival_times])
            settled_by_time = np.array([probability[(self.term_times == t) & ~knocks_out].sum()
                                        for t in self.survival_times])
            settled_before = np.concatenate([[0.0], np.cumsum(settled_by_time)[:-1]])
            survival = 1.0 - np.cumsum(ko_by_time) - settled_before
            fields["termination.survival"] = survival
            ended = np.concatenate([[0.0], 1.0 - survival])
            fields["observations.probability"] = pre_count / n
            fields["observations.survival"] = 1.0 - ended[np.searchsorted(self.survival_times, self.pre_times,
                                                                           side="right")]
            fields["observations.discounted_cash"] = np.asarray(sums["pre_disc_cash"], dtype=float) / n
            fields["observations.cash_to_pay"] = np.asarray(sums["pre_cash"], dtype=float) / n
            offsets_p = np.concatenate([post_count / n, [fields["ki_settlement.probability"]]])
            offsets_d = np.concatenate([np.asarray(sums["post_disc_cash"], dtype=float) / n,
                                        [fields["ki_settlement.discounted_cash"]]])
            offsets_c = np.concatenate([np.asarray(sums["post_cash"], dtype=float) / n,
                                        [fields["ki_settlement.cash_to_pay"]]])
            order = self._offset_order()
            fields["post_ki_offsets.probability"] = offsets_p[order]
            fields["post_ki_offsets.discounted_cash"] = offsets_d[order]
            fields["post_ki_offsets.cash_to_pay"] = offsets_c[order]
            fields["expected_lifetime"] = float(np.sum(probability * self.term_times))
            fields["total_discounted_cash"] = float(np.sum(discounted)) / n
            fields["total_cash_to_pay"] = float(np.sum(cash)) / n
            clean_times, clean_counts = self.pre_times, pre_count
        else:
            probability = np.concatenate([pre_count, post_count])[self.order] / n
            discounted = np.concatenate([np.asarray(sums["pre_disc_cash"], dtype=float),
                                         np.asarray(sums["post_disc_cash"], dtype=float)])[self.order] / n
            cash = np.concatenate([np.asarray(sums["pre_cash"], dtype=float),
                                   np.asarray(sums["post_cash"], dtype=float)])[self.order] / n
            survival = 1.0 - np.cumsum(probability)
            if self.f5:
                survival = survival - np.where(self.obs_times >= self.pre_maturity, fields["maturity.probability"],
                                               0.0)
            fields["observations.probability"] = probability
            fields["observations.survival"] = survival
            fields["observations.discounted_cash"] = discounted
            fields["observations.cash_to_pay"] = cash
            fields["expected_lifetime"] = float(np.sum(probability * self.obs_times)) + fields[
                "maturity.probability"] * self.pre_maturity + fields["ki_settlement.probability"] * self.post_maturity
            fields["total_discounted_cash"] = float(np.sum(discounted)) + fields["maturity.discounted_cash"] + fields[
                "ki_settlement.discounted_cash"]
            fields["total_cash_to_pay"] = float(np.sum(cash)) + fields["maturity.cash_to_pay"] + fields[
                "ki_settlement.cash_to_pay"]
            clean_times, clean_counts = self.pre_times, pre_count
        # a first-schedule knock-out has no knock-in before it (a knock-in replaces that schedule for good)
        fields.update(self.knock_in_fields(sums, self.layout, knocked_out_now=knocked_out_now,
                                           clean_ko_times=clean_times, clean_ko_counts=clean_counts))
        return fields

    def _offset_order(self) -> np.ndarray:
        offsets = np.concatenate([self.post_times, [float(np.max(self.post_times)) if self.post_times.size else 0.0]])
        kinds = np.concatenate([np.zeros(self.post_times.size), [1.0]])
        return np.lexsort((kinds, offsets))

    def build(self, pv: float, totals: Mapping, records) -> EventStatistics:
        fields = self.fields(totals)
        v0 = self.v0_timing
        maturity = OutcomeRow(
            time=self.pre_maturity, payment_time=float(v0.payment_time), date=v0.determination_date,
            payment_date=v0.payment_date, probability=fields["maturity.probability"],
            discounted_cash=fields["maturity.discounted_cash"], cash_to_pay=fields["maturity.cash_to_pay"],
        )
        tables: dict = {"maturity": maturity}
        if self.rebased:
            n = self.pre_times.size
            observations = ObservationTable(
                times=self.pre_times, payment_times=self.pre_settlement, dates=self.pre_dates,
                payment_dates=self.pre_payment_dates, is_ko_obs=np.ones(n, dtype=bool), is_coupon_obs=np.zeros(n, dtype=bool), schedule=("pre",) * n,
                probability=fields["observations.probability"], survival=fields["observations.survival"],
                discounted_cash=fields["observations.discounted_cash"],
                cash_to_pay=fields["observations.cash_to_pay"], payment_is_path_dependent=np.zeros(n, dtype=bool),
            )
            # T9 convention: a REBASED knocked-in settlement has a path-dependent date; the scalar row stands at
            # the latest candidate settlement (the contract's last possible knocked-in end), the actual dates are
            # the termination distribution's ``ki_settlement`` rows
            row = self.ki_settlement_row
            if row is None:
                ki_time, ki_payment, ki_date, ki_payment_date = self.pre_maturity, float(v0.payment_time), None, None
            else:
                ki_time, ki_payment = float(self.term_times[row]), float(self.term_payment[row])
                ki_date = None if self.term_dates is None else self.term_dates[row]
                ki_payment_date = None if self.term_payment_dates is None else self.term_payment_dates[row]
            ki_settlement = OutcomeRow(
                time=ki_time, payment_time=ki_payment, date=ki_date, payment_date=ki_payment_date,
                probability=fields["ki_settlement.probability"],
                discounted_cash=fields["ki_settlement.discounted_cash"],
                cash_to_pay=fields["ki_settlement.cash_to_pay"],
            )
            order = self._offset_order()
            offsets = np.concatenate([self.post_times, [float(np.max(self.post_times)) if self.post_times.size
                                                        else 0.0]])[order]
            kinds = ("ko_post",) * self.post_times.size + ("ki_settlement",)
            tables["post_ki_offsets"] = PostKIOffsetTable(
                offsets=offsets, kind=tuple(kinds[i] for i in order),
                probability=fields["post_ki_offsets.probability"],
                discounted_cash=fields["post_ki_offsets.discounted_cash"],
                cash_to_pay=fields["post_ki_offsets.cash_to_pay"],
            )
            termination = TerminationDistribution(
                times=self.term_times, payment_times=self.term_payment, dates=self.term_dates,
                payment_dates=self.term_payment_dates, event=self.term_event,
                probability=fields["termination.probability"], discounted_cash=fields["termination.discounted_cash"],
                cash_to_pay=fields["termination.cash_to_pay"], survival_times=self.survival_times,
                survival=fields["termination.survival"],
            )
            tables["termination"] = termination
            ledger = build_rebased_cash_ledger(termination=termination, receivables=ReceivablesTable.empty())
        else:
            m = self.obs_times.size
            observations = ObservationTable(
                times=self.obs_times, payment_times=self.obs_payment, dates=self.obs_dates,
                payment_dates=self.obs_payment_dates, is_ko_obs=np.ones(m, dtype=bool), is_coupon_obs=np.zeros(m, dtype=bool), schedule=self.obs_tags,
                probability=fields["observations.probability"], survival=fields["observations.survival"],
                discounted_cash=fields["observations.discounted_cash"],
                cash_to_pay=fields["observations.cash_to_pay"], payment_is_path_dependent=np.zeros(m, dtype=bool),
            )
            v1 = self.v1_timing
            ki_settlement = OutcomeRow(
                time=self.post_maturity, payment_time=float(v1.payment_time), date=v1.determination_date,
                payment_date=v1.payment_date, probability=fields["ki_settlement.probability"],
                discounted_cash=fields["ki_settlement.discounted_cash"],
                cash_to_pay=fields["ki_settlement.cash_to_pay"],
            )
            ledger = build_cash_ledger(observations=observations, coupons=None, maturity=maturity,
                                       ki_settlement=ki_settlement, receivables=ReceivablesTable.empty())
        tables.update(observations=observations, ki_settlement=ki_settlement, ledger=ledger)
        return self.finish(pv=pv, totals=totals, records=records, fields=fields, tables=tables)


def ko_reset_event_statistics(engine, product, pricing_env, pv: float, totals: Mapping, records) -> EventStatistics:
    """The KO-reset statistics (ABSOLUTE or REBASED) of one recorded ``price()``."""
    return _KOResetAssembly(engine, product, pricing_env, totals).build(pv, totals, records)
