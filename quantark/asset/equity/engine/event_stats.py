"""
Engine-level event stats and cashflow decomposition types.

The unified family (:class:`EventStatistics` and its tables, spec 2026-11-11 §Contracts) is what
every autocallable and DCN engine returns from ``calculate_event_stats``: per-observation outcome
probabilities, expected discounted and undiscounted cash, the knock-in path statistics and the
payment-date ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Iterator, Optional, Union

import numpy as np

from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import Tolerance, is_close


@dataclass(frozen=True)
class StatisticsProvenance:
    """Which implementation produced the statistics and which the price (spec 3B §5.6, decision 8).

    Implementation provenance only: it states nothing about the statistics' accuracy for a product
    and market; the certificates are the only such statements.
    """

    statistics_model: str
    price_model: str

    @property
    def surrogate(self) -> bool:
        return self.statistics_model != self.price_model


class StatisticsRefused(PricingError):
    """A solver declines to report statistics it does not compute natively (spec 3B §5.6)."""


class StatisticUnavailable(ValidationError, ValueError):
    """A statistic listed in ``EventStatistics.unavailable`` was read through ``EventStatistics.require``.

    The field is ``None`` or not the field's definition (not requested, or undefined for the trade); the message
    carries the ``unavailable`` reason.
    """


# =============================================================================
# Unified event statistics (spec 2026-11-11 §Contracts)
# =============================================================================
#
# Validation copies and freezes every array and raises ``ValidationError`` on shape, finiteness,
# negative-probability, non-monotone-survival and row-ordering violations. It never forces or
# rescales a number: the partition and cash identities are *reported* (``probability_residual``,
# ``reconciliation_error``), not enforced.

PRODUCT_FAMILIES = ("snowball", "phoenix", "ko_reset", "dcn")
SCHEDULE_TAGS = ("pre", "post")
OFFSET_KINDS = ("ko_post", "ki_settlement")
TERMINATION_EVENTS = ("ko_pre", "ko_post", "maturity", "ki_settlement")
RECEIVABLE_SOURCES = ("ko", "coupon", "maturity", "ki_settlement", "dcn_loss")

# Probabilities and survival may undershoot 0 / overshoot monotonicity by this much (grid
# round-off); the same slack ``EventDistribution`` applies to the distributions built from them.
_PROBABILITY_SLACK = Tolerance.PROBABILITY
# A payment may precede its determination by this much (time round-off).
_TIME_SLACK = Tolerance.ZERO


class _FrozenMapping(Mapping):
    """Read-only, picklable mapping (``types.MappingProxyType`` cannot be pickled)."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping) -> None:
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


def _set(instance, name: str, value) -> None:
    object.__setattr__(instance, name, value)


def _float_array(owner: str, name: str, value, size: Optional[int] = None) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{owner}.{name} must be a 1-D float array") from exc
    if array.ndim != 1:
        raise ValidationError(f"{owner}.{name} must be one-dimensional, got shape {array.shape}")
    if size is not None and array.size != size:
        raise ValidationError(f"{owner}.{name} has {array.size} rows, expected {size} (misaligned)")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{owner}.{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _bool_array(owner: str, name: str, value, size: int) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.size == 0:
        array = array.astype(bool).reshape(0)
    if array.dtype != np.bool_ or array.ndim != 1:
        raise ValidationError(f"{owner}.{name} must be a 1-D bool array")
    if array.size != size:
        raise ValidationError(f"{owner}.{name} has {array.size} rows, expected {size} (misaligned)")
    array.setflags(write=False)
    return array


def _finite_float(owner: str, name: str, value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{owner}.{name} must be a float") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{owner}.{name} must be finite, got {result}")
    return result


def _optional_float(owner: str, name: str, value) -> Optional[float]:
    return None if value is None else _finite_float(owner, name, value)


def _dates(owner: str, name: str, value, size: int) -> Optional[tuple]:
    if value is None:
        return None
    dates = tuple(value)
    if len(dates) != size:
        raise ValidationError(f"{owner}.{name} has {len(dates)} dates, expected {size} (misaligned)")
    if any(not isinstance(date, datetime) for date in dates):
        raise ValidationError(f"{owner}.{name} must contain datetime values")
    return dates


def _optional_date(owner: str, name: str, value) -> Optional[datetime]:
    if value is not None and not isinstance(value, datetime):
        raise ValidationError(f"{owner}.{name} must be a datetime or None")
    return value


def _tags(owner: str, name: str, value, size: int, allowed: tuple) -> tuple:
    tags = tuple(value)
    if len(tags) != size:
        raise ValidationError(f"{owner}.{name} has {len(tags)} entries, expected {size} (misaligned)")
    unknown = sorted({str(tag) for tag in tags if tag not in allowed})
    if unknown:
        raise ValidationError(f"{owner}.{name} has unknown values {unknown}; allowed {list(allowed)}")
    return tags


def _check_probability(owner: str, name: str, values: np.ndarray) -> None:
    if np.any(values < -_PROBABILITY_SLACK):
        raise ValidationError(f"{owner}.{name} has a negative probability (min {float(values.min())})")


def _check_survival(owner: str, name: str, survival: np.ndarray) -> None:
    _check_probability(owner, name, survival)
    if np.any(np.diff(survival) > _PROBABILITY_SLACK):
        raise ValidationError(f"{owner}.{name} must be monotone non-increasing")


def _check_not_before_valuation(owner: str, name: str, times: np.ndarray) -> None:
    if np.any(times < -_TIME_SLACK):
        raise ValidationError(
            f"{owner}.{name} precedes the valuation (t < 0); cash decided before valuation "
            "belongs in the receivables table"
        )


def _check_payment_not_before(owner: str, determination: np.ndarray, payment: np.ndarray) -> None:
    if np.any(payment + _TIME_SLACK < determination):
        raise ValidationError(f"{owner} payment times cannot precede determination times")


def _check_ordered_rows(owner: str, name: str, times: np.ndarray, keys: Optional[tuple] = None) -> None:
    """Rows sorted by time; a time repeats only with distinct ``keys`` (strictly increasing without keys)."""
    if np.any(np.diff(times) < 0.0):
        raise ValidationError(f"{owner}.{name} must be sorted in increasing time")
    pairs = list(zip(times.tolist(), keys if keys is not None else [None] * times.size))
    if len(set(pairs)) != len(pairs):
        raise ValidationError(f"{owner}.{name} has duplicate rows (repeated time{' and tag' if keys else ''})")


@dataclass(frozen=True, kw_only=True)
class ObservationTable:
    """Knock-out table on the shared KO/coupon observation grid (DESIGN §4.2).

    One row per observation in the union of KO and coupon observations, sorted by time. A
    coupon-only row (``is_ko_obs`` False) carries zero KO probability and cash. KO-reset rows of
    both schedules sit in one table tagged ``schedule`` ``"pre"``/``"post"`` (a time may repeat
    once per tag); REBASED ``post`` rows live in :class:`PostKIOffsetTable` instead. A dated
    knock-out-only table (no coupon row, no tag) may repeat a time with distinct ``dates``: a
    Snowball's knock-out records on one env instant, one row per contract record.

    Attributes:
        times, payment_times: determination and settlement time of each row (pricing axis).
        dates, payment_dates: the same as datetimes, or ``None`` (numeric clock).
        is_ko_obs, is_coupon_obs: which observation kinds the row is.
        schedule: ``"pre"``/``"post"`` per row on KO-reset, else ``None``.
        probability: P(KO at this row).
        survival: P(alive after this row). KO-reset ABSOLUTE: also drops the pre-maturity mass at
            its determination time.
        discounted_cash: E[DF(payment) x KO redemption x 1{KO here}], coupons excluded.
        cash_to_pay: E[KO redemption x 1{KO here}], undiscounted.
        payment_is_path_dependent: True where the row has no single payment time (its
            ``payment_times`` entry is a placeholder; the ledger holds the actual payments).
    """

    times: np.ndarray
    payment_times: np.ndarray
    dates: Optional[tuple[datetime, ...]] = None
    payment_dates: Optional[tuple[datetime, ...]] = None
    is_ko_obs: np.ndarray
    is_coupon_obs: np.ndarray
    schedule: Optional[tuple[str, ...]] = None
    probability: np.ndarray
    survival: np.ndarray
    discounted_cash: np.ndarray
    cash_to_pay: np.ndarray
    payment_is_path_dependent: np.ndarray

    def __post_init__(self) -> None:
        owner = type(self).__name__
        times = _float_array(owner, "times", self.times)
        n = times.size
        values = {"times": times}
        for name in ("payment_times", "probability", "survival", "discounted_cash", "cash_to_pay"):
            values[name] = _float_array(owner, name, getattr(self, name), n)
        for name in ("is_ko_obs", "is_coupon_obs", "payment_is_path_dependent"):
            values[name] = _bool_array(owner, name, getattr(self, name), n)
        for name in ("dates", "payment_dates"):
            values[name] = _dates(owner, name, getattr(self, name), n)
        schedule = None if self.schedule is None else _tags(owner, "schedule", self.schedule, n, SCHEDULE_TAGS)
        values["schedule"] = schedule

        _check_not_before_valuation(owner, "times", times)
        keys = schedule
        if schedule is None and values["dates"] is not None and not np.any(values["is_coupon_obs"]):
            # a dated knock-out-only table keeps one row per contract record: a Snowball's records on one env
            # instant repeat its time with their own dates (0.16.0 DCN/dated D8). Coupon tables (Phoenix, DCN) and
            # tagged tables (KO-reset) keep one row per time: their same-instant records are an open item.
            keys = values["dates"]
        _check_ordered_rows(owner, "times", times, keys)
        _check_payment_not_before(owner, times, values["payment_times"])
        _check_probability(owner, "probability", values["probability"])
        _check_survival(owner, "survival", values["survival"])
        if np.any(~(values["is_ko_obs"] | values["is_coupon_obs"])):
            raise ValidationError(f"{owner}: every row must be at least one of a KO or a coupon observation")
        not_ko = ~values["is_ko_obs"]
        if np.any(values["probability"][not_ko] != 0.0) or np.any(values["discounted_cash"][not_ko] != 0.0) \
                or np.any(values["cash_to_pay"][not_ko] != 0.0):
            raise ValidationError(f"{owner}: rows with is_ko_obs False carry no KO probability or cash")
        for name, value in values.items():
            _set(self, name, value)

    @property
    def size(self) -> int:
        """Number of rows."""
        return int(self.times.size)


@dataclass(frozen=True, kw_only=True)
class CouponTable:
    """Coupon table on the same row grid as :class:`ObservationTable` (DESIGN §4.3).

    Rows that are not coupon observations carry zero probability and cash. A coupon paid with a
    knock-out (current coupon and released memory arrears) is counted here, not in KO cash.

    Attributes:
        probability: P(alive just before the row and the coupon barrier met, incl. a simultaneous
            KO); ``None`` when pruned (``unavailable["coupons.probability"]``).
        survival: P(alive going into the row).
        payment_times, payment_dates: settlement of the row's coupon; a placeholder on
            path-dependent rows (DCN: the coupon pay times, distinct from KO pay).
        discounted_cash, cash_to_pay: E[DF x coupon paid x 1{paid here}] and its undiscounted
            counterpart, computed directly (memory arrears included).
        payment_is_path_dependent: True on rows paid at a random time (EXPIRY pay).
    """

    probability: Optional[np.ndarray]
    survival: np.ndarray
    payment_times: np.ndarray
    payment_dates: Optional[tuple[datetime, ...]] = None
    discounted_cash: np.ndarray
    cash_to_pay: np.ndarray
    payment_is_path_dependent: np.ndarray

    def __post_init__(self) -> None:
        owner = type(self).__name__
        survival = _float_array(owner, "survival", self.survival)
        n = survival.size
        values = {"survival": survival}
        for name in ("payment_times", "discounted_cash", "cash_to_pay"):
            values[name] = _float_array(owner, name, getattr(self, name), n)
        values["probability"] = (
            None if self.probability is None else _float_array(owner, "probability", self.probability, n)
        )
        values["payment_is_path_dependent"] = _bool_array(
            owner, "payment_is_path_dependent", self.payment_is_path_dependent, n
        )
        values["payment_dates"] = _dates(owner, "payment_dates", self.payment_dates, n)
        if values["probability"] is not None:
            _check_probability(owner, "probability", values["probability"])
        _check_survival(owner, "survival", survival)
        for name, value in values.items():
            _set(self, name, value)

    @property
    def size(self) -> int:
        """Number of rows."""
        return int(self.survival.size)


@dataclass(frozen=True, kw_only=True)
class OutcomeRow:
    """One scalar termination outcome: matured never KO/KI (DESIGN §4.4) or matured knocked in (§4.5).

    Attributes:
        time, payment_time: determination and settlement time.
        date, payment_date: the same as datetimes, or ``None``.
        probability: P(this outcome).
        discounted_cash, cash_to_pay: E[DF x redemption x 1{outcome}] and its undiscounted counterpart.
    """

    time: float
    payment_time: float
    date: Optional[datetime] = None
    payment_date: Optional[datetime] = None
    probability: float
    discounted_cash: float
    cash_to_pay: float

    def __post_init__(self) -> None:
        owner = type(self).__name__
        values = {
            name: _finite_float(owner, name, getattr(self, name))
            for name in ("time", "payment_time", "probability", "discounted_cash", "cash_to_pay")
        }
        values["date"] = _optional_date(owner, "date", self.date)
        values["payment_date"] = _optional_date(owner, "payment_date", self.payment_date)
        _check_not_before_valuation(owner, "time", np.array([values["time"]]))
        _check_payment_not_before(owner, np.array([values["time"]]), np.array([values["payment_time"]]))
        _check_probability(owner, "probability", np.array([values["probability"]]))
        for name, value in values.items():
            _set(self, name, value)


@dataclass(frozen=True, kw_only=True)
class KnockInStatistics:
    """Knock-in path statistics (DESIGN §4.5); they may overlap KO outcomes.

    Attributes:
        probability: P(KI before any KO); ``None`` when pruned.
        probability_no_ko: P(KI and no KO); equals ``ki_settlement.probability`` exactly.
        expected_time: E[first KI time | KI before any KO]; ``None`` when pruned or when
            P(KI) = 0 (reason ``"undefined: ..."``).
        expected_loss: E[DF(maturity payment) x (-downside) x 1{matured knocked in}] >= 0; ``None``
            when pruned.
    """

    probability: Optional[float]
    probability_no_ko: float
    expected_time: Optional[float]
    expected_loss: Optional[float]

    def __post_init__(self) -> None:
        owner = type(self).__name__
        values = {"probability_no_ko": _finite_float(owner, "probability_no_ko", self.probability_no_ko)}
        for name in ("probability", "expected_time", "expected_loss"):
            values[name] = _optional_float(owner, name, getattr(self, name))
        for name, value in values.items():
            if value is not None and value < -_PROBABILITY_SLACK:
                raise ValidationError(f"{owner}.{name} must be non-negative, got {value}")
        if values["probability"] == 0.0 and values["expected_time"] is not None:
            raise ValidationError(f"{owner}.expected_time is undefined when P(knock in) = 0; pass None")
        for name, value in values.items():
            _set(self, name, value)


@dataclass(frozen=True, kw_only=True)
class KnockInTable:
    """Per-date first-knock-in table on the product's KI schedule (DESIGN §4.5).

    Payment time = KI observation time (spec D8); cash legs add their own settlement offset.

    Attributes:
        times, payment_times: KI observation times (strictly increasing); ``payment_times == times``.
        dates, payment_dates: the same as datetimes, or ``None`` (numeric clock).
        probability: P(first KI at this row, before any KO).
        survival: P(not knocked in and not knocked out after this row).
        interpolated: True where the row boundary is not a grid time (labelled approximation).
        interpolation_scheme: the scheme name when any row is interpolated (MC: ``"step_overlap"``,
            spec D18), else ``None``.
    """

    times: np.ndarray
    payment_times: np.ndarray
    dates: Optional[tuple[datetime, ...]] = None
    payment_dates: Optional[tuple[datetime, ...]] = None
    probability: np.ndarray
    survival: np.ndarray
    interpolated: np.ndarray
    interpolation_scheme: Optional[str] = None

    def __post_init__(self) -> None:
        owner = type(self).__name__
        times = _float_array(owner, "times", self.times)
        n = times.size
        values = {"times": times}
        for name in ("payment_times", "probability", "survival"):
            values[name] = _float_array(owner, name, getattr(self, name), n)
        values["interpolated"] = _bool_array(owner, "interpolated", self.interpolated, n)
        for name in ("dates", "payment_dates"):
            values[name] = _dates(owner, name, getattr(self, name), n)
        _check_not_before_valuation(owner, "times", times)
        _check_ordered_rows(owner, "times", times)
        if not np.array_equal(values["payment_times"], times) or values["payment_dates"] != values["dates"]:
            raise ValidationError(f"{owner}: payment time/date must equal the KI observation time/date (D8)")
        _check_probability(owner, "probability", values["probability"])
        _check_survival(owner, "survival", values["survival"])
        scheme = self.interpolation_scheme
        if scheme is not None and (not isinstance(scheme, str) or not scheme):
            raise ValidationError(f"{owner}.interpolation_scheme must be a non-empty string or None")
        if bool(np.any(values["interpolated"])) != (scheme is not None):
            raise ValidationError(
                f"{owner}.interpolation_scheme must name the scheme exactly when a row is interpolated"
            )
        for name, value in values.items():
            _set(self, name, value)

    @classmethod
    def empty(cls) -> "KnockInTable":
        """The table of a note with no reachable knock-in (no barrier, or KO at valuation)."""
        return cls(times=[], payment_times=[], probability=[], survival=[], interpolated=[])

    @property
    def size(self) -> int:
        """Number of rows."""
        return int(self.times.size)


@dataclass(frozen=True, kw_only=True)
class PostKIOffsetTable:
    """REBASED KO-reset rows by offset from the knock-in time (DESIGN §4.2; no survival column).

    Attributes:
        offsets: offset from the knock-in time (>= 0), sorted; an offset repeats once per kind.
        kind: ``"ko_post"`` or ``"ki_settlement"`` per row.
        probability, discounted_cash, cash_to_pay: per row.
    """

    offsets: np.ndarray
    kind: tuple[str, ...]
    probability: np.ndarray
    discounted_cash: np.ndarray
    cash_to_pay: np.ndarray

    def __post_init__(self) -> None:
        owner = type(self).__name__
        offsets = _float_array(owner, "offsets", self.offsets)
        n = offsets.size
        values = {"offsets": offsets, "kind": _tags(owner, "kind", self.kind, n, OFFSET_KINDS)}
        for name in ("probability", "discounted_cash", "cash_to_pay"):
            values[name] = _float_array(owner, name, getattr(self, name), n)
        if np.any(offsets < 0.0):
            raise ValidationError(f"{owner}.offsets must be non-negative")
        _check_ordered_rows(owner, "offsets", offsets, values["kind"])
        _check_probability(owner, "probability", values["probability"])
        for name, value in values.items():
            _set(self, name, value)

    @property
    def size(self) -> int:
        """Number of rows."""
        return int(self.offsets.size)


@dataclass(frozen=True, kw_only=True)
class TerminationDistribution:
    """REBASED absolute-time termination distribution (DESIGN §4.2, spec D13).

    One row per (absolute determination time, event) on the full deterministic candidate date set
    (every reachable KI date + each offset, calendar-rolled, plus maturity), zeros kept, so the
    axis depends only on the contract and the valuation date. ``ki_settlement`` rows form an array
    stream (path-dependent settlement dates).

    Attributes:
        times, payment_times, dates, payment_dates: per row; sorted by time.
        event: ``"ko_pre"``, ``"ko_post"``, ``"maturity"`` or ``"ki_settlement"`` per row.
        probability, discounted_cash, cash_to_pay: per row.
        survival_times, survival: absolute survival curve (strictly increasing times).
    """

    times: np.ndarray
    payment_times: np.ndarray
    dates: Optional[tuple[datetime, ...]] = None
    payment_dates: Optional[tuple[datetime, ...]] = None
    event: tuple[str, ...]
    probability: np.ndarray
    discounted_cash: np.ndarray
    cash_to_pay: np.ndarray
    survival_times: np.ndarray
    survival: np.ndarray

    def __post_init__(self) -> None:
        owner = type(self).__name__
        times = _float_array(owner, "times", self.times)
        n = times.size
        values = {"times": times, "event": _tags(owner, "event", self.event, n, TERMINATION_EVENTS)}
        for name in ("payment_times", "probability", "discounted_cash", "cash_to_pay"):
            values[name] = _float_array(owner, name, getattr(self, name), n)
        for name in ("dates", "payment_dates"):
            values[name] = _dates(owner, name, getattr(self, name), n)
        survival_times = _float_array(owner, "survival_times", self.survival_times)
        values["survival_times"] = survival_times
        values["survival"] = _float_array(owner, "survival", self.survival, survival_times.size)
        _check_not_before_valuation(owner, "times", times)
        # a (time, event) pair repeats only with distinct payment times: an EXPIRY-paid REBASED knock-out on
        # one date settles with the knocked-in end of the knock-in that put its schedule in force
        _check_ordered_rows(owner, "times", times, tuple(zip(values["event"], values["payment_times"].tolist())))
        _check_payment_not_before(owner, times, values["payment_times"])
        _check_probability(owner, "probability", values["probability"])
        _check_not_before_valuation(owner, "survival_times", survival_times)
        _check_ordered_rows(owner, "survival_times", survival_times)
        _check_survival(owner, "survival", values["survival"])
        for name, value in values.items():
            _set(self, name, value)

    @property
    def size(self) -> int:
        """Number of rows."""
        return int(self.times.size)


@dataclass(frozen=True, kw_only=True)
class ReceivablesTable:
    """Cash decided at or before valuation and paid after it (DESIGN §4.9c).

    Inside the cash identity, outside the probability identity; never events.

    Attributes:
        source: ``"ko"``, ``"coupon"``, ``"maturity"``, ``"ki_settlement"`` or ``"dcn_loss"``.
        decision_times: decision time on the pricing axis (<= 0: at or before valuation).
        decision_dates: decision dates, or ``None``.
        payment_times, payment_dates: settlement.
        discounted_cash, cash_to_pay: per row.
    """

    source: tuple[str, ...]
    decision_times: np.ndarray
    decision_dates: Optional[tuple[datetime, ...]] = None
    payment_times: np.ndarray
    payment_dates: Optional[tuple[datetime, ...]] = None
    discounted_cash: np.ndarray
    cash_to_pay: np.ndarray

    def __post_init__(self) -> None:
        owner = type(self).__name__
        decision = _float_array(owner, "decision_times", self.decision_times)
        n = decision.size
        values = {"decision_times": decision, "source": _tags(owner, "source", self.source, n, RECEIVABLE_SOURCES)}
        for name in ("payment_times", "discounted_cash", "cash_to_pay"):
            values[name] = _float_array(owner, name, getattr(self, name), n)
        for name in ("decision_dates", "payment_dates"):
            values[name] = _dates(owner, name, getattr(self, name), n)
        if np.any(decision > _TIME_SLACK):
            raise ValidationError(f"{owner}.decision_times must be at or before valuation (<= 0)")
        _check_payment_not_before(owner, decision, values["payment_times"])
        for name, value in values.items():
            _set(self, name, value)

    @classmethod
    def empty(cls) -> "ReceivablesTable":
        """No cash decided-but-unpaid."""
        return cls(source=(), decision_times=[], payment_times=[], discounted_cash=[], cash_to_pay=[])

    @property
    def size(self) -> int:
        """Number of rows."""
        return int(self.decision_times.size)


@dataclass(frozen=True, kw_only=True)
class CashLedger:
    """Payment-date view of every cash amount (DESIGN §4.6): one row per (determination, payment).

    Its discounted sum equals the tables' discounted cash. ``cash_to_pay`` is computed directly
    (never recovered from a payment DF on a path-dependent row) and is ``None`` when pruned (spec D17).

    Attributes:
        determination_times, payment_times: per row (payment never precedes determination).
        determination_dates, payment_dates: datetimes, or ``None`` unless every row has one.
        discounted_cash: E[DF x cash] per row.
        cash_to_pay: undiscounted E[cash] per row, or ``None`` when pruned.
    """

    determination_times: np.ndarray
    payment_times: np.ndarray
    determination_dates: Optional[tuple[datetime, ...]] = None
    payment_dates: Optional[tuple[datetime, ...]] = None
    discounted_cash: np.ndarray
    cash_to_pay: Optional[np.ndarray]

    def __post_init__(self) -> None:
        owner = type(self).__name__
        determination = _float_array(owner, "determination_times", self.determination_times)
        n = determination.size
        values = {
            "determination_times": determination,
            "payment_times": _float_array(owner, "payment_times", self.payment_times, n),
            "discounted_cash": _float_array(owner, "discounted_cash", self.discounted_cash, n),
            "cash_to_pay": (
                None if self.cash_to_pay is None else _float_array(owner, "cash_to_pay", self.cash_to_pay, n)
            ),
        }
        for name in ("determination_dates", "payment_dates"):
            values[name] = _dates(owner, name, getattr(self, name), n)
        _check_payment_not_before(owner, determination, values["payment_times"])
        for name, value in values.items():
            _set(self, name, value)

    @property
    def size(self) -> int:
        """Number of rows."""
        return int(self.determination_times.size)

    @property
    def total_discounted_cash(self) -> float:
        """Sum of the discounted cash (``math.fsum``)."""
        return math.fsum(self.discounted_cash.tolist())

    @property
    def total_cash_to_pay(self) -> Optional[float]:
        """Sum of the undiscounted cash, or ``None`` when pruned."""
        return None if self.cash_to_pay is None else math.fsum(self.cash_to_pay.tolist())


_COMPONENT_TYPES: dict[str, type] = {
    "observations": ObservationTable,
    "coupons": CouponTable,
    "maturity": OutcomeRow,
    "ki_settlement": OutcomeRow,
    "knock_in": KnockInStatistics,
    "ki_table": KnockInTable,
    "post_ki_offsets": PostKIOffsetTable,
    "termination": TerminationDistribution,
    "receivables": ReceivablesTable,
    "ledger": CashLedger,
}
_STANDARD_ERRORS_PREFIX = "standard_errors."


@dataclass(frozen=True, kw_only=True)
class EventStatistics:
    """One event-statistics result for Snowball, Phoenix, KO-reset and DCN (spec 2026-11-11 §Contracts).

    Every path ends in exactly one outcome: a KO row, the maturity row (never KO, never KI) or the
    KI settlement row (KI, no KO). Cash carries the sign of ``price()`` (DCN: ``direction_sign``);
    probabilities, times and expected loss are unsigned.

    Computed by the type (never passed in):
        total_discounted_cash: Σ observations + Σ coupons + maturity + KI settlement + Σ receivables
            discounted cash (REBASED: Σ termination + Σ receivables).
        total_cash_to_pay: the same sum of ``cash_to_pay``.
        reconciliation_error: ``pv - total_discounted_cash``.
        probability_residual: ``1 - (Σ observations.probability + maturity + KI settlement)``
            (REBASED: ``1 - Σ termination.probability``; a terminated note: 0, not checked).

    Attributes:
        product_family: ``"snowball"``, ``"phoenix"``, ``"ko_reset"`` or ``"dcn"``.
        pv: ``engine.price(...)``.
        observations, coupons, maturity, ki_settlement, knock_in, ki_table: see the table types;
            ``coupons`` is ``None`` exactly for snowball / ko_reset.
        post_ki_offsets, termination: REBASED KO-reset only (both or neither).
        receivables: cash decided but unpaid (possibly empty).
        ledger: payment-date view; its discounted sum equals ``total_discounted_cash``.
        expected_lifetime: E[time to the deciding observation] on the engine's pricing axis.
        provenance: required.
        unavailable: field path -> reason for every value that is ``None`` because it was pruned
            (``"not requested"``) or is undefined (``"undefined: ..."``). Keys are dotted field
            paths (``"knock_in.expected_time"``) or ``"standard_errors.<path>"``.
        standard_errors: MC only; dotted field path -> SE (array values align with their table).
            ``None`` on grid engines and on a single-batch MC run (with
            ``unavailable["standard_errors"] = "undefined: single batch"``).
    """

    product_family: str
    pv: float
    observations: ObservationTable
    coupons: Optional[CouponTable]
    maturity: OutcomeRow
    ki_settlement: OutcomeRow
    knock_in: KnockInStatistics
    ki_table: Optional[KnockInTable]
    post_ki_offsets: Optional[PostKIOffsetTable] = None
    termination: Optional[TerminationDistribution] = None
    receivables: ReceivablesTable
    ledger: CashLedger
    expected_lifetime: float
    total_discounted_cash: float = field(init=False)
    total_cash_to_pay: float = field(init=False)
    reconciliation_error: float = field(init=False)
    probability_residual: float = field(init=False)
    provenance: StatisticsProvenance
    unavailable: Mapping[str, str] = field(default_factory=dict)
    standard_errors: Optional[Mapping[str, Union[float, np.ndarray]]] = None

    def __post_init__(self) -> None:
        owner = type(self).__name__
        if self.product_family not in PRODUCT_FAMILIES:
            raise ValidationError(
                f"{owner}.product_family must be one of {list(PRODUCT_FAMILIES)}, got {self.product_family!r}"
            )
        if not isinstance(self.provenance, StatisticsProvenance):
            raise ValidationError(f"{owner}.provenance is required (a StatisticsProvenance)")
        _set(self, "pv", _finite_float(owner, "pv", self.pv))
        expected_lifetime = _finite_float(owner, "expected_lifetime", self.expected_lifetime)
        if expected_lifetime < -_TIME_SLACK:
            raise ValidationError(f"{owner}.expected_lifetime must be non-negative")
        _set(self, "expected_lifetime", expected_lifetime)
        self._check_component_types()
        self._check_product_shape()
        if self.knock_in.probability_no_ko != self.ki_settlement.probability:
            raise ValidationError(
                f"{owner}: knock_in.probability_no_ko must equal ki_settlement.probability exactly"
            )
        if self.terminated and self.receivables.size == 0:
            raise ValidationError(
                f"{owner}: a terminated, fully settled note has no statistics (return None)"
            )
        _set(self, "unavailable", self._frozen_unavailable())
        self._compute_identities()
        # after the identities: a summary value (``total_discounted_cash``, ...) may carry an error too
        _set(self, "standard_errors", self._frozen_standard_errors())

    # ------------------------------------------------------------------ properties

    @property
    def rebased(self) -> bool:
        """True for a REBASED KO-reset (offset and termination tables present)."""
        return self.termination is not None

    @property
    def terminated(self) -> bool:
        """True for a terminated note reporting receivables only (no remaining termination mass)."""
        return (
            self.termination is None
            and self.observations.size == 0
            and (self.coupons is None or self.coupons.size == 0)
            and self.maturity.probability == 0.0
            and self.ki_settlement.probability == 0.0
        )

    # ------------------------------------------------------------------ validation

    def _check_component_types(self) -> None:
        for name, cls in _COMPONENT_TYPES.items():
            value = getattr(self, name)
            optional = name in ("coupons", "ki_table", "post_ki_offsets", "termination")
            if value is None and optional:
                continue
            if not isinstance(value, cls):
                raise ValidationError(
                    f"{type(self).__name__}.{name} must be a {cls.__name__}, got {type(value).__name__}"
                )

    def _check_product_shape(self) -> None:
        owner = type(self).__name__
        family = self.product_family
        has_coupons = family in ("phoenix", "dcn")
        if has_coupons and self.coupons is None:
            raise ValidationError(f"{owner}: a {family} carries a coupons table")
        if not has_coupons and self.coupons is not None:
            raise ValidationError(f"{owner}: coupons is None for {family} (AC11)")
        schedule = self.observations.schedule
        if (family == "ko_reset") != (schedule is not None):
            raise ValidationError(f"{owner}: observations.schedule tags are present exactly on ko_reset")
        if self.coupons is not None:
            obs = self.observations
            if self.coupons.size != obs.size:
                raise ValidationError(
                    f"{owner}: coupons has {self.coupons.size} rows, the observation grid {obs.size}"
                )
            _check_payment_not_before(f"{owner}.coupons", obs.times, self.coupons.payment_times)
            not_coupon = ~obs.is_coupon_obs
            columns = [self.coupons.discounted_cash, self.coupons.cash_to_pay]
            if self.coupons.probability is not None:
                columns.append(self.coupons.probability)
            if any(np.any(column[not_coupon] != 0.0) for column in columns):
                raise ValidationError(
                    f"{owner}: coupon rows that are not coupon observations carry no probability or cash"
                )
        if (self.post_ki_offsets is None) != (self.termination is None):
            raise ValidationError(f"{owner}: REBASED carries both post_ki_offsets and termination, or neither")
        if self.termination is not None:
            if family != "ko_reset":
                raise ValidationError(f"{owner}: REBASED tables exist only on ko_reset, not {family}")
            if "post" in schedule:
                raise ValidationError(
                    f"{owner}: REBASED post rows belong in post_ki_offsets, not the observation table"
                )

    def _frozen_unavailable(self) -> _FrozenMapping:
        owner = type(self).__name__
        unavailable = dict(self.unavailable)
        for key, reason in unavailable.items():
            if not isinstance(key, str) or not _is_unavailable_key(key):
                raise ValidationError(f"{owner}.unavailable key {key!r} is not a dotted field path")
            if not isinstance(reason, str) or not reason:
                raise ValidationError(f"{owner}.unavailable[{key!r}] needs a non-empty reason")
        required = [
            ("ki_table", self.ki_table),
            ("knock_in.probability", self.knock_in.probability),
            ("knock_in.expected_time", self.knock_in.expected_time),
            ("knock_in.expected_loss", self.knock_in.expected_loss),
            ("ledger.cash_to_pay", self.ledger.cash_to_pay),
        ]
        if self.coupons is not None:
            required.append(("coupons.probability", self.coupons.probability))
        missing = [path for path, value in required if value is None and path not in unavailable]
        if missing:
            raise ValidationError(f"{owner}: {missing} are None without an unavailable reason")
        return _FrozenMapping(unavailable)

    def _frozen_standard_errors(self) -> Optional[_FrozenMapping]:
        owner = type(self).__name__
        if self.standard_errors is None:
            return None
        if "standard_errors" in self.unavailable:
            raise ValidationError(
                f"{owner}: standard_errors must be None when unavailable['standard_errors'] is set"
            )
        if len(self.standard_errors) == 0:
            raise ValidationError(f"{owner}.standard_errors must be None or non-empty")
        frozen: dict[str, Union[float, np.ndarray]] = {}
        for key, value in self.standard_errors.items():
            if not isinstance(key, str) or not _is_field_path(key):
                raise ValidationError(f"{owner}.standard_errors key {key!r} is not a dotted field path")
            target = self._resolve(key)
            label = f"standard_errors[{key!r}]"
            if isinstance(target, np.ndarray) and target.dtype.kind == "f":
                se = _float_array(owner, label, value, target.size)
                if np.any(se < 0.0):
                    raise ValidationError(f"{owner}.{label} must be non-negative")
            elif isinstance(target, float):
                if isinstance(value, (np.ndarray, list, tuple)):
                    raise ValidationError(f"{owner}.{label} must be a scalar for a scalar field")
                se = _finite_float(owner, label, value)
                if se < 0.0:
                    raise ValidationError(f"{owner}.{label} must be non-negative")
            else:
                raise ValidationError(
                    f"{owner}.{label}: the field is not a computed number here (got {type(target).__name__})"
                )
            frozen[key] = se
        return _FrozenMapping(frozen)

    # ------------------------------------------------------------------ guarded reading

    def flag(self, path: str) -> Optional[str]:
        """The ``unavailable`` reason covering a dotted field path (the path itself or its component), else None.

        Raises:
            ValidationError: ``path`` is not a field path of this type.
        """
        if not isinstance(path, str) or not _is_field_path(path):
            raise ValidationError(f"{path!r} is not a field path of {type(self).__name__}")
        head = path.partition(".")[0]
        reason = self.unavailable.get(path)
        return reason if reason is not None else self.unavailable.get(head)

    def require(self, path: str) -> Any:
        """The value at a dotted field path (``"ki_settlement.discounted_cash"``), refusing a flagged one.

        Raises:
            StatisticUnavailable: the path (or its component) is listed in ``unavailable``; the message carries
                the reason. A flagged value may be present but is not the field's definition.
            ValidationError: ``path`` is not a field path of this type.
        """
        reason = self.flag(path)
        if reason is not None:
            raise StatisticUnavailable(f"{type(self).__name__}.{path} is unavailable: {reason}")
        return self._resolve(path)

    def _resolve(self, path: str) -> Any:
        value: Any = self
        for part in path.split("."):
            if value is None:
                return None
            value = getattr(value, part)
        return value

    # ------------------------------------------------------------------ identities

    def _compute_identities(self) -> None:
        receivables = self.receivables
        if self.termination is not None:
            discounted = [self.termination.discounted_cash, receivables.discounted_cash]
            undiscounted = [self.termination.cash_to_pay, receivables.cash_to_pay]
            probabilities = [self.termination.probability]
        else:
            discounted = [self.observations.discounted_cash, receivables.discounted_cash,
                          np.array([self.maturity.discounted_cash, self.ki_settlement.discounted_cash])]
            undiscounted = [self.observations.cash_to_pay, receivables.cash_to_pay,
                            np.array([self.maturity.cash_to_pay, self.ki_settlement.cash_to_pay])]
            probabilities = [self.observations.probability,
                             np.array([self.maturity.probability, self.ki_settlement.probability])]
            if self.coupons is not None:
                discounted.append(self.coupons.discounted_cash)
                undiscounted.append(self.coupons.cash_to_pay)
        discounted_values = np.concatenate(discounted)
        total_discounted = math.fsum(discounted_values.tolist())
        _set(self, "total_discounted_cash", total_discounted)
        _set(self, "total_cash_to_pay", math.fsum(np.concatenate(undiscounted).tolist()))
        _set(self, "reconciliation_error", self.pv - total_discounted)
        residual = 0.0 if self.terminated else 1.0 - math.fsum(np.concatenate(probabilities).tolist())
        _set(self, "probability_residual", residual)

        gross = math.fsum(np.abs(discounted_values).tolist())
        ledger_total = self.ledger.total_discounted_cash
        if not is_close(ledger_total, total_discounted, rel_tol=Tolerance.RELATIVE,
                        abs_tol=max(Tolerance.ZERO, Tolerance.RELATIVE * gross)):
            raise ValidationError(
                f"{type(self).__name__}: Σ ledger.discounted_cash = {ledger_total!r} differs from the "
                f"outcome cash Σ = {total_discounted!r}"
            )


_EVENT_STATISTICS_FIELDS = frozenset(f.name for f in fields(EventStatistics))
_COMPONENT_FIELDS = {name: frozenset(f.name for f in fields(cls)) for name, cls in _COMPONENT_TYPES.items()}


def _is_field_path(path: str) -> bool:
    """``"<field>"`` or ``"<component>.<field>"`` naming a field of :class:`EventStatistics`."""
    head, dot, rest = path.partition(".")
    if head not in _EVENT_STATISTICS_FIELDS:
        return False
    if not dot:
        return True
    return head in _COMPONENT_FIELDS and rest in _COMPONENT_FIELDS[head]


def _is_unavailable_key(path: str) -> bool:
    if path.startswith(_STANDARD_ERRORS_PREFIX):
        return _is_field_path(path[len(_STANDARD_ERRORS_PREFIX):])
    return _is_field_path(path)


# ----------------------------------------------------------------------------- ledger builders


def _ledger_piece(determination, payment, determination_dates, payment_dates, discounted, cash_to_pay) -> dict:
    return {
        "determination": np.asarray(determination, dtype=float),
        "payment": np.asarray(payment, dtype=float),
        "determination_dates": None if determination_dates is None else tuple(determination_dates),
        "payment_dates": None if payment_dates is None else tuple(payment_dates),
        "discounted": np.asarray(discounted, dtype=float),
        "cash_to_pay": None if cash_to_pay is None else np.asarray(cash_to_pay, dtype=float),
    }


def _select_dates(dates: Optional[tuple], mask: np.ndarray) -> Optional[tuple]:
    if dates is None:
        return None
    return tuple(date for date, keep in zip(dates, mask.tolist()) if keep)


def _row_piece(row: OutcomeRow) -> dict:
    return _ledger_piece(
        [row.time], [row.payment_time],
        None if row.date is None else (row.date,),
        None if row.payment_date is None else (row.payment_date,),
        [row.discounted_cash], [row.cash_to_pay],
    )


def _receivables_piece(receivables: ReceivablesTable) -> dict:
    return _ledger_piece(
        receivables.decision_times, receivables.payment_times, receivables.decision_dates,
        receivables.payment_dates, receivables.discounted_cash, receivables.cash_to_pay,
    )


def _ledger_piece_from(ledger: CashLedger) -> dict:
    return _ledger_piece(
        ledger.determination_times, ledger.payment_times, ledger.determination_dates,
        ledger.payment_dates, ledger.discounted_cash, ledger.cash_to_pay,
    )


def _assemble_ledger(pieces: list) -> CashLedger:
    def _joined_dates(key: str) -> Optional[tuple]:
        if any(piece[key] is None for piece in pieces if piece["determination"].size):
            return None
        return tuple(date for piece in pieces for date in (piece[key] or ()))

    pruned = any(piece["cash_to_pay"] is None for piece in pieces)
    return CashLedger(
        determination_times=np.concatenate([piece["determination"] for piece in pieces]),
        payment_times=np.concatenate([piece["payment"] for piece in pieces]),
        determination_dates=_joined_dates("determination_dates"),
        payment_dates=_joined_dates("payment_dates"),
        discounted_cash=np.concatenate([piece["discounted"] for piece in pieces]),
        cash_to_pay=None if pruned else np.concatenate([piece["cash_to_pay"] for piece in pieces]),
    )


def build_cash_ledger(
    *,
    observations: ObservationTable,
    coupons: Optional[CouponTable],
    maturity: OutcomeRow,
    ki_settlement: OutcomeRow,
    receivables: ReceivablesTable,
    path_dependent_cash: Optional[CashLedger] = None,
) -> CashLedger:
    """Build the cash ledger from directly computed outcome cash (DESIGN §4.6).

    Rows, in this order: KO rows (``is_ko_obs``) and coupon rows (``is_coupon_obs``) whose payment
    time is deterministic, at (observation, payment) time; then ``path_dependent_cash``; then the
    maturity row, the KI settlement row and the receivables (at their decision times). Nothing is
    recovered from a discount factor: every ``cash_to_pay`` is taken as computed.

    Args:
        observations, coupons, maturity, ki_settlement, receivables: the tables of the result.
        path_dependent_cash: the cash of every row flagged ``payment_is_path_dependent`` (EXPIRY
            coupons, ...), bucketed at its actual payment events and computed directly by the
            producer; required exactly when such a row exists. Its ``cash_to_pay`` may be ``None``
            (pruned, spec D17), which prunes the ledger's ``cash_to_pay``.

    Returns:
        The :class:`CashLedger`; dates are kept only when every row has them.
    """
    pieces = []
    path_dependent = False
    ko_rows = observations.is_ko_obs & ~observations.payment_is_path_dependent
    path_dependent |= bool(np.any(observations.is_ko_obs & observations.payment_is_path_dependent))
    pieces.append(_ledger_piece(
        observations.times[ko_rows], observations.payment_times[ko_rows],
        _select_dates(observations.dates, ko_rows), _select_dates(observations.payment_dates, ko_rows),
        observations.discounted_cash[ko_rows], observations.cash_to_pay[ko_rows],
    ))
    if coupons is not None:
        if coupons.size != observations.size:
            raise ValidationError("coupons must share the observation grid")
        coupon_rows = observations.is_coupon_obs & ~coupons.payment_is_path_dependent
        path_dependent |= bool(np.any(observations.is_coupon_obs & coupons.payment_is_path_dependent))
        pieces.append(_ledger_piece(
            observations.times[coupon_rows], coupons.payment_times[coupon_rows],
            _select_dates(observations.dates, coupon_rows), _select_dates(coupons.payment_dates, coupon_rows),
            coupons.discounted_cash[coupon_rows], coupons.cash_to_pay[coupon_rows],
        ))
    if path_dependent and path_dependent_cash is None:
        raise ValidationError(
            "path-dependent payment rows need path_dependent_cash bucketed at their actual payment events"
        )
    if path_dependent_cash is not None:
        if not path_dependent:
            raise ValidationError("path_dependent_cash given but no row is flagged path-dependent")
        pieces.append(_ledger_piece_from(path_dependent_cash))
    pieces.extend([_row_piece(maturity), _row_piece(ki_settlement), _receivables_piece(receivables)])
    return _assemble_ledger(pieces)


def build_rebased_cash_ledger(
    *, termination: TerminationDistribution, receivables: ReceivablesTable
) -> CashLedger:
    """Cash ledger of a REBASED KO-reset: the termination rows, then the receivables (spec D13).

    Each termination row has one deterministic payment date, so its row is its own payment event.
    """
    return _assemble_ledger([
        _ledger_piece(termination.times, termination.payment_times, termination.dates,
                      termination.payment_dates, termination.discounted_cash, termination.cash_to_pay),
        _receivables_piece(receivables),
    ])


def checked_event_statistics(obj: Any) -> EventStatistics:
    """``obj`` unchanged when it is :class:`EventStatistics`; anything else is refused.

    Consumers read a producer's ``calculate_event_stats`` result through this: every engine returns
    the unified family, and there is no conversion from any other type.

    Raises:
        ValidationError: ``obj`` is not an :class:`EventStatistics`.
    """
    if isinstance(obj, EventStatistics):
        return obj
    raise ValidationError(
        f"calculate_event_stats must return EventStatistics, got {type(obj).__name__}"
    )


__all__ = [
    "CashLedger",
    "CouponTable",
    "EventStatistics",
    "KnockInStatistics",
    "KnockInTable",
    "ObservationTable",
    "OFFSET_KINDS",
    "OutcomeRow",
    "PostKIOffsetTable",
    "PRODUCT_FAMILIES",
    "RECEIVABLE_SOURCES",
    "ReceivablesTable",
    "SCHEDULE_TAGS",
    "StatisticsProvenance",
    "StatisticsRefused",
    "StatisticUnavailable",
    "TERMINATION_EVENTS",
    "TerminationDistribution",
    "build_cash_ledger",
    "build_rebased_cash_ledger",
    "checked_event_statistics",
]
