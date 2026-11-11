"""Engine-emitted timing distributions for cash-leg valuation.

Each event stream has its own time axis (unified event statistics, DESIGN §4.11): the KO and COUPON
streams share the observation grid, the KI stream sits on the per-date KI table's axis, and a scalar
(terminal) outcome keeps its own determination time. ``EventDistribution.from_statistics`` is the one
conversion from :class:`~quantark.asset.equity.engine.event_stats.EventStatistics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np

from quantark.asset.equity.engine.event_stats import StatisticsProvenance
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import Tolerance, almost_equal, is_zero


class EventType(Enum):
    """Standard event streams emitted by pricing engines."""

    KO = "knock_out"
    KI = "knock_in"
    COUPON = "coupon"
    MATURITY_NO_KO = "maturity_no_ko"
    MATURITY_WITH_KI = "maturity_with_ki"


_TERMINATION_EVENTS = {
    EventType.KO,
    EventType.MATURITY_NO_KO,
    EventType.MATURITY_WITH_KI,
}
#: streams that always live on an axis (array probabilities)
_ARRAY_STREAMS = frozenset({EventType.KO, EventType.KI, EventType.COUPON})
_KO_EVENTS = ("ko_pre", "ko_post")


@dataclass(frozen=True, kw_only=True)
class EventDistribution:
    """Probability distribution over termination and auxiliary payment events, one time axis per stream.

    Attributes:
        probabilities: per stream, an array on that stream's axis (KO, KI, COUPON; and MATURITY_WITH_KI on a
            REBASED KO-reset) or a scalar (MATURITY_NO_KO, MATURITY_WITH_KI otherwise).
        survival_times, survival_probability: P(alive) on its own axis, starting at the valuation (0). Without
            an observation at 0 the first value is 1; with one, it is that observation's post-event survival.
        stream_times: determination time of every row of each array stream (year fractions).
        stream_dates: the same as datetimes per stream, where known.
        payment_times, payment_dates: settlement per stream (an array on the stream's axis, or a scalar).
        path_dependent_payment_events: streams whose payment needs the engine's cash ledger.
        determination_times: determination time of each scalar outcome; one absent here is determined at the
            last survival time.
        provenance: which implementation produced the statistics and the price.
        probability_residual: the statistics' ``probability_residual``, absorbed into MATURITY_NO_KO by
            ``from_statistics`` (spec D16) and carried here so a reader sees it.
        rebased: a REBASED KO-reset (spec D13): events sit on the absolute candidate axis and legs value each
            event at its own time.
    """

    probabilities: Dict[EventType, Union[np.ndarray, float]]
    survival_times: np.ndarray
    survival_probability: np.ndarray
    stream_times: Dict[EventType, np.ndarray] = field(default_factory=dict)
    stream_dates: Optional[Dict[EventType, tuple[datetime, ...]]] = None
    payment_times: Dict[EventType, Union[np.ndarray, float]] = field(default_factory=dict)
    payment_dates: Optional[Dict[EventType, Union[tuple[datetime, ...], datetime]]] = None
    # Such events retain trigger probabilities, but their settlement requires a
    # joint timing distribution or the engine's expected-cashflow ledger.
    path_dependent_payment_events: frozenset[EventType] = field(default_factory=frozenset)
    determination_times: Dict[EventType, float] = field(default_factory=dict)
    provenance: Optional[StatisticsProvenance] = None
    probability_residual: float = 0.0
    rebased: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "survival_times", np.asarray(self.survival_times, dtype=float))
        object.__setattr__(self, "survival_probability", np.asarray(self.survival_probability, dtype=float))
        object.__setattr__(
            self,
            "probabilities",
            {
                event_type: (
                    np.asarray(probability, dtype=float)
                    if isinstance(probability, np.ndarray)
                    else float(probability)
                )
                for event_type, probability in self.probabilities.items()
            },
        )
        object.__setattr__(
            self,
            "stream_times",
            {event_type: np.asarray(times, dtype=float) for event_type, times in self.stream_times.items()},
        )
        if self.stream_dates is not None:
            object.__setattr__(
                self, "stream_dates", {event_type: tuple(dates) for event_type, dates in self.stream_dates.items()}
            )
        object.__setattr__(
            self,
            "payment_times",
            {
                event_type: (
                    np.asarray(payment_time, dtype=float)
                    if isinstance(payment_time, np.ndarray)
                    else float(payment_time)
                )
                for event_type, payment_time in self.payment_times.items()
            },
        )
        if self.payment_dates is not None:
            object.__setattr__(
                self,
                "payment_dates",
                {
                    event_type: (
                        tuple(payment_date)
                        if not isinstance(payment_date, datetime)
                        else payment_date
                    )
                    for event_type, payment_date in self.payment_dates.items()
                },
            )
        object.__setattr__(
            self,
            "determination_times",
            {event_type: float(t) for event_type, t in self.determination_times.items()},
        )
        object.__setattr__(self, "probability_residual", float(self.probability_residual))
        object.__setattr__(self, "path_dependent_payment_events", frozenset(self.path_dependent_payment_events))
        self._validate_invariants()

    # --- construction ---------------------------------------------------------------------------------

    @classmethod
    def trivial(cls, maturity: float) -> "EventDistribution":
        """Return a no-KO distribution with one terminal maturity outcome."""

        maturity = float(maturity)
        return cls(
            probabilities={EventType.MATURITY_NO_KO: 1.0},
            survival_times=np.array([0.0, maturity], dtype=float),
            survival_probability=np.array([1.0, 1.0], dtype=float),
            payment_times={
                EventType.MATURITY_NO_KO: maturity,
                EventType.MATURITY_WITH_KI: maturity,
            },
            determination_times={EventType.MATURITY_NO_KO: maturity, EventType.MATURITY_WITH_KI: maturity},
        )

    @classmethod
    def from_statistics(cls, stats) -> "EventDistribution":
        """The distribution of an :class:`EventStatistics` (DESIGN §4.11; spec D13, D16).

        * KO: every observation row (coupon-only rows carry KO probability 0), paid at its payment time.
        * COUPON: the coupon table's probabilities on the same grid; path-dependent payment where flagged.
          Absent when the coupon probability is ``None`` (pruned or unavailable).
        * KI: the per-date KI table on its own axis, paid at the KI observation (D8; a leg adds its own
          settlement offset). An empty table is an empty stream; ``None`` (pruned/unavailable) is no stream.
        * MATURITY_NO_KO: the maturity row's probability plus ``probability_residual`` (D16, never raised on);
          MATURITY_WITH_KI: the KI settlement row.
        * REBASED (D13): KO = the termination distribution's ``ko_pre`` + ``ko_post`` rows and MATURITY_WITH_KI
          its ``ki_settlement`` rows, both array streams on the absolute candidate axis; MATURITY_NO_KO the
          ``maturity`` row; survival the termination distribution's absolute curve.
        * Receivables are ledger rows, not events: they are not read.
        """
        if stats.termination is not None:
            return cls._from_rebased(stats)
        obs = stats.observations
        probabilities: Dict[EventType, Union[np.ndarray, float]] = {EventType.KO: obs.probability}
        stream_times: Dict[EventType, np.ndarray] = {EventType.KO: obs.times}
        payment_times: Dict[EventType, Union[np.ndarray, float]] = {}
        stream_dates: Dict[EventType, tuple] = {}
        payment_dates: Dict[EventType, object] = {}
        path_dependent = set()
        if np.any(obs.payment_is_path_dependent):
            path_dependent.add(EventType.KO)
        else:
            payment_times[EventType.KO] = obs.payment_times
            if obs.payment_dates is not None:
                payment_dates[EventType.KO] = obs.payment_dates
        if obs.dates is not None:
            stream_dates[EventType.KO] = obs.dates

        coupons = stats.coupons
        if coupons is not None and coupons.probability is not None:
            probabilities[EventType.COUPON] = coupons.probability
            stream_times[EventType.COUPON] = obs.times
            if obs.dates is not None:
                stream_dates[EventType.COUPON] = obs.dates
            coupon_rows = obs.is_coupon_obs
            if np.any(coupons.payment_is_path_dependent[coupon_rows]):
                path_dependent.add(EventType.COUPON)
            else:
                payment_times[EventType.COUPON] = coupons.payment_times
                if coupons.payment_dates is not None:
                    payment_dates[EventType.COUPON] = coupons.payment_dates

        cls._add_ki_stream(stats, probabilities, stream_times, stream_dates, payment_times, payment_dates)
        cls._add_outcome(EventType.MATURITY_NO_KO, stats.maturity, stats.maturity.probability + stats.probability_residual,
                         probabilities, payment_times, payment_dates)
        cls._add_outcome(EventType.MATURITY_WITH_KI, stats.ki_settlement, stats.ki_settlement.probability,
                         probabilities, payment_times, payment_dates)
        survival_times, survival = cls._survival_axis(obs.times, obs.survival, stats.maturity.time)
        return cls(
            probabilities=probabilities,
            survival_times=survival_times,
            survival_probability=survival,
            stream_times=stream_times,
            stream_dates=stream_dates or None,
            payment_times=payment_times,
            payment_dates=payment_dates or None,
            path_dependent_payment_events=frozenset(path_dependent),
            determination_times={EventType.MATURITY_NO_KO: stats.maturity.time,
                                 EventType.MATURITY_WITH_KI: stats.ki_settlement.time},
            provenance=stats.provenance,
            probability_residual=stats.probability_residual,
        )

    @classmethod
    def _from_rebased(cls, stats) -> "EventDistribution":
        term = stats.termination
        events = np.asarray(term.event, dtype=object)

        def rows(mask):
            idx = np.flatnonzero(mask)
            dates = None if term.dates is None else tuple(term.dates[i] for i in idx)
            pay_dates = None if term.payment_dates is None else tuple(term.payment_dates[i] for i in idx)
            return idx, dates, pay_dates

        probabilities: Dict[EventType, Union[np.ndarray, float]] = {}
        stream_times: Dict[EventType, np.ndarray] = {}
        payment_times: Dict[EventType, Union[np.ndarray, float]] = {}
        stream_dates: Dict[EventType, tuple] = {}
        payment_dates: Dict[EventType, object] = {}
        for event_type, mask in ((EventType.KO, np.isin(events, _KO_EVENTS)),
                                 (EventType.MATURITY_WITH_KI, events == "ki_settlement")):
            idx, dates, pay_dates = rows(mask)
            probabilities[event_type] = term.probability[idx]
            stream_times[event_type] = term.times[idx]
            payment_times[event_type] = term.payment_times[idx]
            if dates is not None:
                stream_dates[event_type] = dates
            if pay_dates is not None:
                payment_dates[event_type] = pay_dates
        idx, _, pay_dates = rows(events == "maturity")
        if idx.size == 0 or not np.all(term.times[idx] == term.times[idx[0]]):
            raise ValidationError(
                "a REBASED termination distribution carries one maturity determination time (zeros kept)"
            )
        maturity_time = float(term.times[idx[0]])
        probabilities[EventType.MATURITY_NO_KO] = float(np.sum(term.probability[idx])) + stats.probability_residual
        payment_times[EventType.MATURITY_NO_KO] = float(term.payment_times[idx[0]])
        if pay_dates is not None:
            payment_dates[EventType.MATURITY_NO_KO] = pay_dates[0]
        cls._add_ki_stream(stats, probabilities, stream_times, stream_dates, payment_times, payment_dates)
        survival_times, survival = cls._survival_axis(term.survival_times, term.survival, maturity_time)
        return cls(
            probabilities=probabilities,
            survival_times=survival_times,
            survival_probability=survival,
            stream_times=stream_times,
            stream_dates=stream_dates or None,
            payment_times=payment_times,
            payment_dates=payment_dates or None,
            determination_times={EventType.MATURITY_NO_KO: maturity_time},
            provenance=stats.provenance,
            probability_residual=stats.probability_residual,
            rebased=True,
        )

    @staticmethod
    def _add_ki_stream(stats, probabilities, stream_times, stream_dates, payment_times, payment_dates) -> None:
        table = stats.ki_table
        if table is None:
            return
        probabilities[EventType.KI] = table.probability
        stream_times[EventType.KI] = table.times
        payment_times[EventType.KI] = table.payment_times
        if table.dates is not None:
            stream_dates[EventType.KI] = table.dates
        if table.payment_dates is not None:
            payment_dates[EventType.KI] = table.payment_dates

    @staticmethod
    def _add_outcome(event_type, row, probability, probabilities, payment_times, payment_dates) -> None:
        if -Tolerance.PROBABILITY <= probability < 0.0:
            probability = 0.0                      # round-off below zero; the termination sum stays within tolerance
        probabilities[event_type] = float(probability)
        payment_times[event_type] = row.payment_time
        if row.payment_date is not None:
            payment_dates[event_type] = row.payment_date

    @staticmethod
    def _survival_axis(times, survival, terminal_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Survival from the valuation on: 1 at 0 unless an observation sits at 0 (its post-event survival)."""
        times = np.asarray(times, dtype=float)
        survival = np.asarray(survival, dtype=float)
        if times.size == 0:
            end = max(float(terminal_time), 0.0)
            if almost_equal(end, 0.0, tol=Tolerance.PROBABILITY):
                return np.array([0.0]), np.array([1.0])
            return np.array([0.0, end]), np.array([1.0, 1.0])
        if almost_equal(float(times[0]), 0.0, tol=Tolerance.PROBABILITY):
            return times, survival
        return np.r_[0.0, times], np.r_[1.0, survival]

    # --- legacy converters (D15: routed through the transition module until T21) -----------------------

    @classmethod
    def from_autocallable_stats(cls, stats) -> "EventDistribution":
        """A legacy autocallable statistics object, read through the transition module (spec D15)."""
        from quantark.asset.equity.engine._event_stats_transition import _as_event_statistics

        return cls.from_statistics(_as_event_statistics(stats))

    @classmethod
    def from_dcn_stats(cls, stats) -> "EventDistribution":
        """A legacy ``DCNEventStats``, read through the transition module (spec D15)."""
        from quantark.asset.equity.engine._event_stats_transition import _as_event_statistics

        return cls.from_statistics(_as_event_statistics(stats))

    # --- reading ----------------------------------------------------------------------------------------

    def payment_times_for(
        self,
        event_type: EventType,
    ) -> Union[np.ndarray, float]:
        """Return fixed payment timing, or reject path-dependent settlement."""

        if event_type in self.path_dependent_payment_events:
            raise NotImplementedError(
                f"{event_type.value} payment timing is path-dependent; "
                "use the engine's expected-cashflow ledger"
            )
        if event_type in self.payment_times:
            return self.payment_times[event_type]
        probability = self.probabilities.get(event_type)
        if isinstance(probability, np.ndarray):
            return self.stream_times[event_type]
        return self.determination_time(event_type)

    def determination_time(self, event_type: EventType) -> float:
        """Determination time of a scalar outcome (the last survival time unless stated)."""
        return float(self.determination_times.get(event_type, self.survival_times[-1]))

    def survival_at(self, t: float) -> float:
        """Linearly interpolate survival probability at a year fraction.

        A strictly past time reads 1. A time at the valuation instant (``is_zero``) reads the post-observation
        survival of an event the grid keeps at 0 — a theta-shifted contract keeps the observation landing on its
        new origin — else 1.
        """

        t = float(t)
        time_grid, survival_grid = self.survival_times, self.survival_probability
        if is_zero(t):
            if almost_equal(float(time_grid[0]), 0.0, tol=Tolerance.PROBABILITY):
                return float(survival_grid[0])
            return 1.0
        if t < 0.0:
            return 1.0
        if t <= time_grid[0]:
            return float(survival_grid[0])
        if t >= time_grid[-1]:
            return float(survival_grid[-1])
        return float(np.interp(t, time_grid, survival_grid))

    def normalized(self) -> "EventDistribution":
        """Return a copy with tiny termination-probability drift normalized."""

        total = self._termination_probability_total()
        drift = total - 1.0
        if almost_equal(drift, 0.0, tol=Tolerance.PROBABILITY):
            return self

        probabilities = dict(self.probabilities)
        if EventType.MATURITY_NO_KO in probabilities:
            probabilities[EventType.MATURITY_NO_KO] = max(
                0.0, float(probabilities[EventType.MATURITY_NO_KO]) - drift
            )
        elif EventType.MATURITY_WITH_KI in probabilities and not isinstance(
            probabilities[EventType.MATURITY_WITH_KI], np.ndarray
        ):
            probabilities[EventType.MATURITY_WITH_KI] = max(
                0.0, float(probabilities[EventType.MATURITY_WITH_KI]) - drift
            )
        else:
            return self

        return EventDistribution(
            probabilities=probabilities,
            survival_times=self.survival_times,
            survival_probability=self.survival_probability,
            stream_times=self.stream_times,
            stream_dates=self.stream_dates,
            payment_times=self.payment_times,
            payment_dates=self.payment_dates,
            path_dependent_payment_events=self.path_dependent_payment_events,
            determination_times=self.determination_times,
            provenance=self.provenance,
            probability_residual=self.probability_residual,
            rebased=self.rebased,
        )

    # --- invariants -------------------------------------------------------------------------------------

    def _validate_invariants(self) -> None:
        self._validate_survival()
        for event_type, probability in self.probabilities.items():
            arr = np.asarray(probability, dtype=float)
            if np.any(~np.isfinite(arr)):
                raise NumericalError(
                    f"EventDistribution probability {event_type.value} is non-finite"
                )
            if np.any(arr < -Tolerance.PROBABILITY):
                raise NumericalError(
                    f"EventDistribution probability {event_type.value} is negative"
                )
            self._validate_stream_shape(event_type, probability)
        for event_type in self.stream_times:
            if not isinstance(self.probabilities.get(event_type), np.ndarray):
                raise NumericalError(
                    f"EventDistribution stream_times[{event_type.value}] names no array stream"
                )
        if self.stream_dates is not None:
            for event_type, dates in self.stream_dates.items():
                if event_type not in self.stream_times or len(dates) != self.stream_times[event_type].size:
                    raise NumericalError(
                        f"EventDistribution stream_dates[{event_type.value}] is not aligned to its axis"
                    )
        self._validate_payment_times()
        if not np.isfinite(self.probability_residual):
            raise NumericalError("EventDistribution.probability_residual must be finite")

        total = self._termination_probability_total()
        if not almost_equal(total, 1.0, tol=Tolerance.PROBABILITY):
            raise NumericalError(
                f"EventDistribution termination probability sum = {total}, expected 1.0 "
                f"(tolerance {Tolerance.PROBABILITY})"
            )

    def _validate_survival(self) -> None:
        times, survival = self.survival_times, self.survival_probability
        if times.ndim != 1 or survival.ndim != 1:
            raise NumericalError("EventDistribution survival_times / survival_probability must be 1D")
        if times.size == 0:
            raise NumericalError("EventDistribution.survival_times must not be empty")
        if survival.shape != times.shape:
            raise NumericalError(
                f"len(survival_probability) = {survival.size}, expected {times.size} (one per survival time)"
            )
        if np.any(~np.isfinite(times)) or np.any(~np.isfinite(survival)):
            raise NumericalError("EventDistribution survival axis contains non-finite values")
        if np.any(np.diff(times) < -Tolerance.PROBABILITY):
            raise NumericalError("EventDistribution.survival_times must be sorted")
        if not almost_equal(float(times[0]), 0.0, tol=Tolerance.PROBABILITY):
            raise NumericalError(
                f"EventDistribution.survival_times starts at {times[0]}, expected the valuation (0)"
            )
        if float(survival[0]) > 1.0 + Tolerance.PROBABILITY:
            raise NumericalError(f"survival_probability[0] = {survival[0]} exceeds 1")
        if np.any(np.diff(survival) > Tolerance.PROBABILITY):
            raise NumericalError(
                "EventDistribution survival_probability is not monotone non-increasing"
            )

    def _validate_stream_shape(self, event_type: EventType, probability) -> None:
        is_array = isinstance(probability, np.ndarray)
        if event_type in _ARRAY_STREAMS and not is_array:
            raise NumericalError(f"EventDistribution stream {event_type.value} must be an array on its own axis")
        if event_type is EventType.MATURITY_NO_KO and is_array:
            raise NumericalError("EventDistribution outcome maturity_no_ko is a scalar")
        if event_type is EventType.MATURITY_WITH_KI and is_array and not self.rebased:
            raise NumericalError(
                "EventDistribution outcome maturity_with_ki is a scalar except on a REBASED KO-reset (spec D13)"
            )
        if not is_array:
            return
        times = self.stream_times.get(event_type)
        if times is None:
            raise NumericalError(f"EventDistribution array stream {event_type.value} has no time axis")
        if probability.ndim != 1 or times.shape != probability.shape:
            raise NumericalError(
                f"EventDistribution probability {event_type.value} has shape {probability.shape}, "
                f"its axis {times.shape}"
            )
        if np.any(~np.isfinite(times)):
            raise NumericalError(f"EventDistribution axis of {event_type.value} contains non-finite values")
        if np.any(np.diff(times) < -Tolerance.PROBABILITY):
            raise NumericalError(f"EventDistribution axis of {event_type.value} must be sorted")

    def _validate_payment_times(self) -> None:
        for event_type, payment_time in self.payment_times.items():
            if event_type not in self.probabilities:
                continue
            times = np.asarray(payment_time, dtype=float)
            if np.any(~np.isfinite(times)):
                raise NumericalError(
                    f"EventDistribution payment time {event_type.value} is non-finite"
                )
            probability = self.probabilities[event_type]
            if isinstance(probability, np.ndarray):
                axis = self.stream_times[event_type]
                if times.shape != axis.shape:
                    raise NumericalError(
                        f"EventDistribution payment time {event_type.value} has "
                        f"shape {times.shape}, expected {axis.shape}"
                    )
                if np.any(times + Tolerance.PROBABILITY < axis):
                    raise NumericalError(
                        f"EventDistribution payment time {event_type.value} "
                        "precedes its determination time"
                    )
            elif times.ndim != 0:
                raise NumericalError(
                    f"EventDistribution scalar event {event_type.value} requires "
                    "a scalar payment time"
                )
            elif float(times) + Tolerance.PROBABILITY < self.determination_time(event_type):
                raise NumericalError(
                    f"EventDistribution payment time {event_type.value} "
                    "precedes terminal determination"
                )

    def _termination_probability_total(self) -> float:
        total = 0.0
        for event_type, probability in self.probabilities.items():
            if event_type not in _TERMINATION_EVENTS:
                continue
            total += float(np.asarray(probability, dtype=float).sum())
        return total


@dataclass(frozen=True)
class PricingResult:
    """Engine result containing NPV and optional event timing distribution."""

    npv: float
    event_distribution: Optional[EventDistribution] = None
