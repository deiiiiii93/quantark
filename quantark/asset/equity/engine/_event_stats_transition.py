"""Branch-only transition: legacy event statistics read as :class:`EventStatistics` (spec D7, D15).

While producers migrate one family at a time, consumers read statistics only through
:func:`_as_event_statistics`: the new type passes through; each legacy type converts by its own named
conversion (``_from_legacy_snowball`` / ``_phoenix`` / ``_ko_reset`` / ``_dcn``).

Conversion rule. Every number the legacy object carries is taken as the legacy producer computed it.
A field whose number is not the legacy producer's direct computation under the new definition (a
residual, a different definition, a layout assumption, a value the legacy type has no field for) is
listed in ``unavailable`` with :data:`LEGACY_REASON`. An optional value the legacy object does not carry
at all is ``None`` (no stand-in). A required number with no legacy source is the value that keeps the
type's own identities consistent with the legacy producer's ledger (for example the maturity cash as the
ledger total less the other outcome cash) and is listed as well, so no consumer may read it as computed.

The legacy ``EventDistribution`` converters (``from_autocallable_stats``, ``from_dcn_stats``) route
through this module, so ``EventDistribution.from_statistics`` is the one conversion. T21 deletes this
module together with the legacy types.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from quantark.asset.equity.engine.event_stats import (
    AutocallableEventStats,
    CashLedger,
    CouponTable,
    DCNEventStats,
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    KOResetEventStats,
    ObservationTable,
    OutcomeRow,
    PhoenixEventStats,
    ReceivablesTable,
    StatisticsProvenance,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance

__all__ = ["LEGACY_REASON", "LEGACY_PROVENANCE", "_as_event_statistics"]

#: the ``unavailable`` reason for every field a legacy producer does not compute directly (spec D15)
LEGACY_REASON = "transition: legacy producer"
#: provenance of a legacy object that does not state its own
LEGACY_PROVENANCE = StatisticsProvenance(statistics_model="legacy", price_model="legacy")

#: fields no legacy producer computes under the new definition
_ALWAYS_LEGACY = (
    "knock_in.expected_time",          # None: no legacy type carries E[first KI time | KI before KO]
    "knock_in.expected_loss",          # None: no legacy type carries the expected knocked-in loss
    "receivables",                     # legacy producers fold receivables into pv / their ledger
    "ledger.cash_to_pay",              # legacy undiscounted cash is discounted / DF, also on path-dependent rows
    "expected_lifetime",               # derived from the legacy tables (DCN: the producer's own value)
)


def _as_event_statistics(obj) -> EventStatistics:
    """``obj`` as :class:`EventStatistics`: the new type unchanged, a legacy type by its named conversion.

    Raises:
        NotImplementedError: a legacy REBASED KO-reset (post-KI rows as offsets; no absolute distribution).
        ValidationError: an unknown type, the path analyzer's record, or a legacy object whose layout the
            legacy converters also refused.
    """
    if isinstance(obj, EventStatistics):
        return obj
    if isinstance(obj, KOResetEventStats):
        return _from_legacy_ko_reset(obj)
    if isinstance(obj, PhoenixEventStats):
        return _from_legacy_phoenix(obj)
    if isinstance(obj, AutocallableEventStats):
        return _from_legacy_snowball(obj)
    if isinstance(obj, DCNEventStats):
        return _from_legacy_dcn(obj)
    name = type(obj).__name__
    if name == "RiskNeutralSnowballEventStats":
        raise ValidationError(
            "RiskNeutralSnowballEventStats has no payment times, no knock-in split and no ledger; it is not "
            "converted (the path analyzer produces EventStatistics once migrated)"
        )
    raise ValidationError(f"no event-statistics conversion for {name}")


# --- shared pieces ---------------------------------------------------------------------------------------


def _provenance(stats) -> StatisticsProvenance:
    provenance = getattr(stats, "provenance", None)
    return provenance if provenance is not None else LEGACY_PROVENANCE


def _legacy_ledger(stats) -> CashLedger:
    """The legacy producer's own payment ledger, verbatim."""
    return CashLedger(
        determination_times=stats.determination_times, payment_times=stats.payment_times,
        determination_dates=stats.determination_dates, payment_dates=stats.payment_dates,
        discounted_cash=stats.expected_discounted_cashflows, cash_to_pay=stats.expected_undiscounted_cashflows,
    )


def _sum(*arrays) -> float:
    return math.fsum(float(x) for array in arrays for x in np.asarray(array, dtype=float).ravel())


def _segment(values, start: int, stop: int, size: int) -> Optional[np.ndarray]:
    """``values[start:stop]`` when the legacy ledger reaches that far, else ``None``."""
    values = np.asarray(values, dtype=float)
    return values[start:stop].copy() if values.size >= size else None


def _dates_segment(dates, start: int, stop: int, size: int) -> Optional[tuple]:
    if dates is None or len(dates) < size:
        return None
    return tuple(dates[start:stop])


def _legacy_ki_table(stats, unavailable: Dict[str, str], ko_times: Optional[np.ndarray]) -> Optional[KnockInTable]:
    """The legacy per-date first-KI arrays, kept only where they are safe to read (CP2 ruling).

    Legacy KI arrays count a knock-in after a knock-out (KI-ever, MC), so they are not the table's definition
    "first KI before any KO". Kept:

    * the carried knock-in row (``times == [0]``, ``probability == [1]``): the new definition exactly, unflagged;
    * a table on the KO grid itself (``ko_times`` given and equal): the legacy distribution kept exactly this
      stream, so it is kept as base did, flagged ``LEGACY_REASON`` (the flag rides onto the distribution).

    Anything else (a KI-ever table on its own axis, which base dropped) is ``None`` with ``LEGACY_REASON``.
    """
    times = np.asarray(stats.ki_times, dtype=float)
    probability = np.asarray(stats.ki_event_probability, dtype=float)
    survival = np.asarray(stats.ki_survival_probability, dtype=float)
    shaped = times.size > 0 and times.size == probability.size == survival.size

    def table() -> KnockInTable:
        return KnockInTable(times=times, payment_times=times, probability=probability, survival=survival,
                            interpolated=np.zeros(times.size, dtype=bool))

    if shaped and times.size == 1 and float(times[0]) == 0.0 and float(probability[0]) == 1.0:
        return table()                              # already knocked in: one row at 0 with probability 1
    unavailable["ki_table"] = LEGACY_REASON
    on_ko_grid = (
        shaped and ko_times is not None and times.size == ko_times.size and np.array_equal(times, ko_times)
        and not np.any(np.diff(survival) > Tolerance.PROBABILITY)
        and not np.any(probability < -Tolerance.PROBABILITY) and not np.any(survival < -Tolerance.PROBABILITY)
    )
    return table() if on_ko_grid else None


def _legacy_maturity_split(ko_total: float, ki_settle: float) -> tuple[float, float]:
    """The legacy distribution's one-maturity split: ``(1 - ΣKO - ki, ki)``, both clipped as it clipped them."""
    maturity_probability = max(0.0, 1.0 - ko_total)
    with_ki = min(maturity_probability, max(0.0, float(ki_settle)))
    return max(0.0, maturity_probability - with_ki), with_ki


def _ki_settle(stats) -> float:
    ki = stats.ki_survive_knocked_in_probability
    return float(stats.ki_probability if ki is None else ki)


def _knock_in(stats, probability_no_ko: float, unavailable: Dict[str, str]) -> KnockInStatistics:
    unavailable["knock_in.probability"] = LEGACY_REASON      # legacy KI-ever counts breaches after a knock-out
    ever = getattr(stats, "ki_ever_probability", None)
    return KnockInStatistics(probability=None if ever is None else float(ever), probability_no_ko=probability_no_ko,
                             expected_time=None, expected_loss=None)


def _expected_lifetime(observations: ObservationTable, terminal_mass: float, final_time: float) -> float:
    """E[time of the deciding observation] (DESIGN §4.8) from the legacy tables."""
    return _sum(observations.probability * observations.times) + terminal_mass * final_time


def _residual_cash(ledger: CashLedger, *tables) -> tuple[float, float]:
    """Ledger totals less the other outcome cash: keeps the type's ledger identity on the legacy ledger."""
    discounted = ledger.total_discounted_cash - _sum(*(t[0] for t in tables))
    undiscounted = (math.fsum(ledger.cash_to_pay.tolist()) if ledger.cash_to_pay is not None else 0.0) \
        - _sum(*(t[1] for t in tables))
    return discounted, undiscounted


def _assemble(stats, *, family: str, observations: ObservationTable, coupons: Optional[CouponTable],
              maturity: OutcomeRow, ki_settlement: OutcomeRow, knock_in: KnockInStatistics,
              ki_table: Optional[KnockInTable], ledger: CashLedger, expected_lifetime: float,
              unavailable: Dict[str, str]) -> EventStatistics:
    for key in _ALWAYS_LEGACY:
        unavailable.setdefault(key, LEGACY_REASON)
    return EventStatistics(
        product_family=family, pv=float(stats.pv), observations=observations, coupons=coupons, maturity=maturity,
        ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table, receivables=ReceivablesTable.empty(),
        ledger=ledger, expected_lifetime=expected_lifetime, provenance=_provenance(stats), unavailable=unavailable,
    )


# --- snowball / phoenix ----------------------------------------------------------------------------------


def _autocallable_observations(stats, unavailable: Dict[str, str], *, coupon_rows: bool) -> ObservationTable:
    times = np.asarray(stats.ko_times, dtype=float)
    n = times.size
    ledger_size = np.asarray(stats.payment_times).size
    payment = _segment(stats.payment_times, 0, n, n + 1)
    if payment is None:
        unavailable["observations.payment_times"] = LEGACY_REASON
        payment = times.copy()                    # the legacy distribution paid at the observation
    discounted = np.asarray(stats.expected_discounted_ko_cashflow, dtype=float)
    if discounted.size != n:
        discounted = np.zeros(n)
    cash_to_pay = _segment(stats.expected_undiscounted_cashflows, 0, n, n + 1)
    if cash_to_pay is None:
        cash_to_pay = np.zeros(n)
    # legacy KO cash includes or excludes a simultaneous coupon by engine; undiscounted is discounted / DF
    unavailable["observations.discounted_cash"] = LEGACY_REASON
    unavailable["observations.cash_to_pay"] = LEGACY_REASON
    dates = _dates_segment(stats.determination_dates, 0, n, n + 1) if ledger_size >= n + 1 else None
    payment_dates = _dates_segment(stats.payment_dates, 0, n, n + 1) if ledger_size >= n + 1 else None
    return ObservationTable(
        times=times, payment_times=payment, dates=dates, payment_dates=payment_dates,
        is_ko_obs=np.ones(n, dtype=bool), is_coupon_obs=np.full(n, coupon_rows),
        probability=np.asarray(stats.ko_probability, dtype=float),
        survival=np.asarray(stats.survival_probability, dtype=float),
        discounted_cash=discounted, cash_to_pay=cash_to_pay, payment_is_path_dependent=np.zeros(n, dtype=bool),
    )


def _terminal_rows(stats, observations: ObservationTable, unavailable: Dict[str, str]):
    """Maturity and KI-settlement rows of a one-maturity legacy producer: the legacy split, both paid at the
    ledger's last payment and determined at its last determination (the last observation without a ledger)."""
    p_no_ko, p_with_ki = _legacy_maturity_split(_sum(observations.probability), _ki_settle(stats))
    unavailable["maturity.probability"] = LEGACY_REASON          # residual 1 - ΣKO - ki, not computed directly
    ledger_size = np.asarray(stats.payment_times).size
    if ledger_size >= observations.size + 1:          # the legacy converter's condition for its terminal row
        time = float(np.asarray(stats.determination_times)[-1])
        payment = float(np.asarray(stats.payment_times)[-1])
        payment_date = None if stats.payment_dates is None else stats.payment_dates[-1]
        date = None if stats.determination_dates is None else stats.determination_dates[-1]
    else:
        time = payment = float(observations.times[-1]) if observations.size else 0.0
        date = payment_date = None
        unavailable["maturity.time"] = unavailable["maturity.payment_time"] = LEGACY_REASON
        unavailable["ki_settlement.time"] = unavailable["ki_settlement.payment_time"] = LEGACY_REASON
    return (p_no_ko, p_with_ki), dict(time=time, payment_time=payment, date=date, payment_date=payment_date)


def _ledger_from_tables(observations: ObservationTable, coupons: Optional[CouponTable],
                        rows: list) -> CashLedger:
    """A ledger for a legacy object that carries none: each table row at its (placeholder) payment time."""
    determination = [observations.times]
    payment = [observations.payment_times]
    discounted = [observations.discounted_cash]
    to_pay = [observations.cash_to_pay]
    if coupons is not None:
        determination.append(observations.times)
        payment.append(coupons.payment_times)
        discounted.append(coupons.discounted_cash)
        to_pay.append(coupons.cash_to_pay)
    for row in rows:
        determination.append([row.time])
        payment.append([row.payment_time])
        discounted.append([row.discounted_cash])
        to_pay.append([row.cash_to_pay])
    return CashLedger(determination_times=np.concatenate(determination), payment_times=np.concatenate(payment),
                      discounted_cash=np.concatenate(discounted), cash_to_pay=np.concatenate(to_pay))


def _finish_autocallable(stats, family: str, observations: ObservationTable, coupons: Optional[CouponTable],
                         unavailable: Dict[str, str]) -> EventStatistics:
    (p_no_ko, p_with_ki), timing = _terminal_rows(stats, observations, unavailable)
    if np.asarray(stats.payment_times).size == 0:
        # no legacy ledger: the legacy maturity total on the maturity row, the ledger built from the tables
        unavailable["ledger"] = LEGACY_REASON
        for key in ("maturity.discounted_cash", "maturity.cash_to_pay", "ki_settlement.discounted_cash",
                    "ki_settlement.cash_to_pay"):
            unavailable[key] = LEGACY_REASON
        maturity = OutcomeRow(probability=p_no_ko, discounted_cash=float(stats.expected_discounted_maturity_cashflow),
                              cash_to_pay=0.0, **timing)
        ki_settlement = OutcomeRow(probability=p_with_ki, discounted_cash=0.0, cash_to_pay=0.0, **timing)
        return _finish_rows(stats, family, observations, coupons, maturity, ki_settlement,
                            _ledger_from_tables(observations, coupons, [maturity, ki_settlement]), unavailable)
    ledger = _legacy_ledger(stats)
    pieces = [(observations.discounted_cash, observations.cash_to_pay)]
    if coupons is not None:
        pieces.append((coupons.discounted_cash, coupons.cash_to_pay))
    maturity_cash, maturity_to_pay = _residual_cash(ledger, *pieces)
    # the legacy no-KO maturity total is not split by knock-in state: all of it sits on the maturity row
    for key in ("maturity.discounted_cash", "maturity.cash_to_pay", "ki_settlement.discounted_cash",
                "ki_settlement.cash_to_pay"):
        unavailable[key] = LEGACY_REASON
    maturity = OutcomeRow(probability=p_no_ko, discounted_cash=maturity_cash, cash_to_pay=maturity_to_pay, **timing)
    ki_settlement = OutcomeRow(probability=p_with_ki, discounted_cash=0.0, cash_to_pay=0.0, **timing)
    return _finish_rows(stats, family, observations, coupons, maturity, ki_settlement, ledger, unavailable)


def _finish_rows(stats, family: str, observations: ObservationTable, coupons: Optional[CouponTable],
                 maturity: OutcomeRow, ki_settlement: OutcomeRow, ledger: CashLedger,
                 unavailable: Dict[str, str]) -> EventStatistics:
    final_time = float(observations.times[-1]) if observations.size else maturity.time
    terminal_mass = maturity.probability + ki_settlement.probability
    # the legacy KO-reset converter never emitted a KI stream: no table is kept as base there
    ko_grid = None if family == "ko_reset" else observations.times
    return _assemble(
        stats, family=family, observations=observations, coupons=coupons, maturity=maturity,
        ki_settlement=ki_settlement, knock_in=_knock_in(stats, ki_settlement.probability, unavailable),
        ki_table=_legacy_ki_table(stats, unavailable, ko_grid), ledger=ledger,
        expected_lifetime=_expected_lifetime(observations, terminal_mass, final_time), unavailable=unavailable,
    )


def _from_legacy_snowball(stats: AutocallableEventStats) -> EventStatistics:
    """A legacy ``AutocallableEventStats`` (Snowball) as ``EventStatistics``."""
    unavailable: Dict[str, str] = {}
    observations = _autocallable_observations(stats, unavailable, coupon_rows=False)
    return _finish_autocallable(stats, "snowball", observations, None, unavailable)


def _from_legacy_phoenix(stats: PhoenixEventStats) -> EventStatistics:
    """A legacy ``PhoenixEventStats`` as ``EventStatistics``: coupons on the KO grid (one shared grid)."""
    unavailable: Dict[str, str] = {}
    observations = _autocallable_observations(stats, unavailable, coupon_rows=True)
    n = observations.size
    probability = np.asarray(stats.coupon_probability, dtype=float)
    if probability.size == 0:
        probability = None
        unavailable["coupons.probability"] = LEGACY_REASON
    elif probability.size != n:
        raise ValidationError(
            f"PhoenixEventStats carries {probability.size} coupon probabilities for {n} observations"
        )
    path_dependent = bool(stats.coupon_payment_is_path_dependent)
    payment = None if path_dependent else _segment(stats.payment_times, n, 2 * n, 2 * n + 1)
    payment_dates = None if path_dependent else _dates_segment(stats.payment_dates, n, 2 * n, 2 * n + 1)
    if payment is None:
        payment = observations.times.copy()       # a placeholder no consumer reads (path-dependent / unknown)
        if not path_dependent:
            unavailable["coupons.payment_times"] = LEGACY_REASON
    discounted = np.asarray(stats.expected_discounted_coupon_cashflow, dtype=float)
    if discounted.size != n:
        discounted = np.zeros(n)
    cash_to_pay = None if path_dependent else _segment(stats.expected_undiscounted_cashflows, n, 2 * n, 2 * n + 1)
    if cash_to_pay is None:
        cash_to_pay = np.zeros(n)
    for key in ("coupons.discounted_cash", "coupons.cash_to_pay", "coupons.survival"):
        unavailable[key] = LEGACY_REASON
    survival = np.r_[1.0, observations.survival[:-1]] if n else np.array([])   # P(alive going into the row)
    coupons = CouponTable(probability=probability, survival=survival, payment_times=payment,
                          payment_dates=payment_dates, discounted_cash=discounted, cash_to_pay=cash_to_pay,
                          payment_is_path_dependent=np.full(n, path_dependent))
    return _finish_autocallable(stats, "phoenix", observations, coupons, unavailable)


# --- KO-reset --------------------------------------------------------------------------------------------


def _from_legacy_ko_reset(stats: KOResetEventStats) -> EventStatistics:
    """A legacy ABSOLUTE ``KOResetEventStats``: both schedules on one grid (the pre record first at a tie), and the
    two maturity buckets where the producer reports them (PDE, QUAD V2), else the legacy one-maturity split."""
    if stats.post_ko_times_are_offsets:
        raise NotImplementedError(
            "a REBASED post-knock-in schedule holds offsets from the knock-in time; "
            "a distribution needs absolute times"
        )
    if stats.pre_ko_payment_times is None or stats.post_ko_payment_times is None:
        raise ValidationError(
            f"{type(stats).__name__} from this producer carries no knock-out payment times "
            "(pre_ko_payment_times / post_ko_payment_times)"
        )
    unavailable: Dict[str, str] = {}
    pre_t = np.asarray(stats.pre_ko_times, dtype=float)
    post_t = np.asarray(stats.post_ko_times, dtype=float)
    n_pre, n_post = pre_t.size, post_t.size
    n = n_pre + n_post
    times = np.r_[pre_t, post_t]
    tag = np.r_[np.zeros(n_pre), np.ones(n_post)]
    order = np.lexsort((tag, times))
    probability = np.r_[np.asarray(stats.pre_ko_probability, dtype=float),
                        np.asarray(stats.post_ko_probability, dtype=float)][order]
    payment = np.r_[np.asarray(stats.pre_ko_payment_times, dtype=float),
                    np.asarray(stats.post_ko_payment_times, dtype=float)][order]
    times = times[order]
    buckets = stats.pre_maturity_probability is not None
    if buckets:
        p_no_ko = float(stats.pre_maturity_probability)
        p_with_ki = float(stats.post_maturity_probability)
    else:
        p_no_ko, p_with_ki = _legacy_maturity_split(_sum(probability), _ki_settle(stats))
        unavailable["maturity.probability"] = LEGACY_REASON
    # F5: the pre-maturity mass leaves the survival at its own determination when that precedes the last knock-out
    survival = 1.0 - np.cumsum(probability)
    if buckets and n and float(stats.pre_maturity_time) < float(times[-1]) - Tolerance.PROBABILITY:
        survival = survival - np.where(times >= float(stats.pre_maturity_time), p_no_ko, 0.0)
    det_dates, pay_dates = stats.determination_dates, stats.payment_dates
    dates = None if det_dates is None or len(det_dates) < n else tuple(det_dates[i] for i in order)
    payment_dates = None if pay_dates is None or len(pay_dates) < n + 1 else tuple(pay_dates[i] for i in order)
    if dates is not None or payment_dates is not None:
        unavailable["observations.dates"] = unavailable["observations.payment_dates"] = LEGACY_REASON
    # per-row KO cash is not reported for both schedules; legacy KO cash and its undiscounted twin are layout-bound
    unavailable["observations.discounted_cash"] = unavailable["observations.cash_to_pay"] = LEGACY_REASON
    observations = ObservationTable(
        times=times, payment_times=payment, dates=dates, payment_dates=payment_dates,
        is_ko_obs=np.ones(n, dtype=bool), is_coupon_obs=np.zeros(n, dtype=bool),
        schedule=tuple("pre" if tag[i] == 0 else "post" for i in order), probability=probability,
        survival=survival, discounted_cash=np.zeros(n), cash_to_pay=np.zeros(n),
        payment_is_path_dependent=np.zeros(n, dtype=bool),
    )
    ledger = _legacy_ledger(stats)
    ledger_payment = np.asarray(stats.payment_times, dtype=float)
    if buckets:
        maturity_timing = dict(time=float(stats.pre_maturity_time), payment_time=float(stats.pre_maturity_payment_time),
                               payment_date=None if pay_dates is None or len(pay_dates) < n + 1 else pay_dates[-2])
        ki_timing = dict(time=float(stats.post_maturity_time), payment_time=float(stats.post_maturity_payment_time),
                         payment_date=None if pay_dates is None or len(pay_dates) < n + 1 else pay_dates[-1])
        ki_cash = float(stats.expected_discounted_post_maturity_cashflow or 0.0)
        # the producer reports the knocked-in bucket discounted only; its ledger's last row (the same bucket)
        # carries the undiscounted twin
        ki_to_pay = float(np.asarray(stats.expected_undiscounted_cashflows)[-1]) if ledger.size else 0.0
    else:
        # the legacy one-maturity outcome: determined at the last knock-out, paid at the ledger's last payment
        final = float(times[-1]) if n else 0.0
        pay = float(ledger_payment[-1]) if ledger_payment.size else final
        last_date = None if pay_dates is None or len(pay_dates) < n + 1 else pay_dates[-1]
        maturity_timing = dict(time=final, payment_time=pay, payment_date=last_date)
        ki_timing = dict(maturity_timing)
        ki_cash = ki_to_pay = 0.0
        for key in ("maturity.time", "maturity.payment_time", "ki_settlement.time"):
            unavailable[key] = LEGACY_REASON
    ki_settlement = OutcomeRow(probability=p_with_ki, discounted_cash=ki_cash, cash_to_pay=ki_to_pay, **ki_timing)
    for key in ("maturity.discounted_cash", "maturity.cash_to_pay", "ki_settlement.cash_to_pay"):
        unavailable[key] = LEGACY_REASON
    if not buckets:
        unavailable["ki_settlement.discounted_cash"] = LEGACY_REASON
    if ledger.size == 0:
        # no legacy ledger: the legacy maturity total less the knocked-in bucket, the ledger built from the tables
        unavailable["ledger"] = LEGACY_REASON
        maturity = OutcomeRow(probability=p_no_ko, cash_to_pay=0.0, **maturity_timing,
                              discounted_cash=float(stats.expected_discounted_maturity_cashflow) - ki_cash)
        ledger = _ledger_from_tables(observations, None, [maturity, ki_settlement])
    else:
        maturity_cash, maturity_to_pay = _residual_cash(
            ledger, (observations.discounted_cash, observations.cash_to_pay), ([ki_cash], [ki_to_pay]))
        maturity = OutcomeRow(probability=p_no_ko, discounted_cash=maturity_cash, cash_to_pay=maturity_to_pay,
                              **maturity_timing)
    return _finish_rows(stats, "ko_reset", observations, None, maturity, ki_settlement, ledger, unavailable)


# --- DCN -------------------------------------------------------------------------------------------------


def _from_legacy_dcn(stats: DCNEventStats) -> EventStatistics:
    """A legacy ``DCNEventStats`` read from its ledger, laid out ``[coupons..., KO..., loss]`` on the monthly rows."""
    ko_probability = np.asarray(stats.ko_timing_distribution, dtype=float)
    n = ko_probability.size
    determination = np.asarray(stats.determination_times, dtype=float)
    payment = np.asarray(stats.payment_times, dtype=float)
    discounted = np.asarray(stats.expected_discounted_cashflows, dtype=float)
    undiscounted = np.asarray(stats.expected_undiscounted_cashflows, dtype=float)
    if n == 0 or determination.size != 2 * n + 1:
        raise ValidationError(
            "DCNEventStats needs a [coupons..., KO..., loss] ledger aligned to its observations"
        )
    unavailable: Dict[str, str] = {}
    det_dates, pay_dates = stats.determination_dates, stats.payment_dates
    obs_times = determination[n:2 * n]
    # every monthly row is a KO slot (zero mass on lock-up rows) and a coupon row: the legacy padded grid
    unavailable["observations.is_ko_obs"] = unavailable["observations.is_coupon_obs"] = LEGACY_REASON
    unavailable["observations.cash_to_pay"] = unavailable["coupons.cash_to_pay"] = LEGACY_REASON
    survival = np.maximum(0.0, 1.0 - np.cumsum(ko_probability))       # as the legacy DCN converter
    observations = ObservationTable(
        times=obs_times, payment_times=payment[n:2 * n],
        dates=None if det_dates is None else tuple(det_dates[n:2 * n]),
        payment_dates=None if pay_dates is None else tuple(pay_dates[n:2 * n]),
        is_ko_obs=np.ones(n, dtype=bool), is_coupon_obs=np.ones(n, dtype=bool), probability=ko_probability,
        survival=survival, discounted_cash=discounted[n:2 * n], cash_to_pay=undiscounted[n:2 * n],
        payment_is_path_dependent=np.zeros(n, dtype=bool),
    )
    # legacy P(fixed coupon paid) excludes knock-out rows: not the table's "barrier met, incl. a simultaneous KO"
    unavailable["coupons.probability"] = unavailable["coupons.survival"] = LEGACY_REASON
    coupons = CouponTable(
        probability=None, survival=np.r_[1.0, survival[:-1]], payment_times=payment[:n],
        payment_dates=None if pay_dates is None else tuple(pay_dates[:n]), discounted_cash=discounted[:n],
        cash_to_pay=undiscounted[:n], payment_is_path_dependent=np.zeros(n, dtype=bool),
    )
    timing = dict(time=float(determination[-1]), payment_time=float(payment[-1]),
                  date=None if det_dates is None else det_dates[-1],
                  payment_date=None if pay_dates is None else pay_dates[-1])
    unavailable["ki_settlement.cash_to_pay"] = LEGACY_REASON
    # a DCN never knocked in pays nothing at maturity; the knocked-in survivor pays the loss leg
    maturity = OutcomeRow(probability=float(stats.prob_survive_no_ki), discounted_cash=0.0, cash_to_pay=0.0, **timing)
    ki_settlement = OutcomeRow(probability=float(stats.prob_survive_ki), discounted_cash=float(discounted[-1]),
                               cash_to_pay=float(undiscounted[-1]), **timing)
    unavailable["knock_in.probability"] = LEGACY_REASON
    knock_in = KnockInStatistics(probability=float(stats.ki_probability), probability_no_ko=ki_settlement.probability,
                                 expected_time=None, expected_loss=None)
    unavailable["ki_table"] = LEGACY_REASON
    for key in _ALWAYS_LEGACY:
        unavailable.setdefault(key, LEGACY_REASON)
    unavailable.pop("expected_lifetime")           # the DCN producer's own E[min(tau_KO, T_last)]
    return EventStatistics(
        product_family="dcn", pv=float(stats.pv), observations=observations, coupons=coupons, maturity=maturity,
        ki_settlement=ki_settlement, knock_in=knock_in, ki_table=None, receivables=ReceivablesTable.empty(),
        ledger=_legacy_ledger(stats), expected_lifetime=float(stats.expected_life_years),
        provenance=_provenance(stats), unavailable=unavailable,
    )
