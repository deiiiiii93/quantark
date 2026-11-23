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
from functools import cached_property
from typing import Dict, Optional, Union

import numpy as np

from quantark.asset.equity.engine.event_stats import StatisticsProvenance
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.asset.equity.product.option.observation_schedule import at_valuation_instant
from quantark.util.numerical import Tolerance, almost_equal, is_zero


class EventType(Enum):
    """Standard event streams emitted by pricing engines."""

    KO = "knock_out"
    KI = "knock_in"
    COUPON = "coupon"
    MATURITY_NO_KO = "maturity_no_ko"
    MATURITY_WITH_KI = "maturity_with_ki"


class SurvivalAnchor(Enum):
    """When a terminated note stops counting as alive for a survival-reading cash leg (spec D29, D32).

    ``DETERMINATION`` (the default): spec D29 -- a knock-out leaves at its own observation row; a maturity-type
    termination (never-knocked-in maturity, knocked-in settlement, each REBASED path's own settlement) is alive at its
    determination and gone right after it, whatever its payment lag.
    ``PAYMENT`` (opt-in): every termination, knock-outs included, is alive until its own payment time and gone after
    it -- the note accrues until its cash is actually paid.
    """

    DETERMINATION = "determination"
    PAYMENT = "payment"


_TERMINATION_EVENTS = {
    EventType.KO,
    EventType.MATURITY_NO_KO,
    EventType.MATURITY_WITH_KI,
}
#: streams that always live on an axis (array probabilities)
_ARRAY_STREAMS = frozenset({EventType.KO, EventType.KI, EventType.COUPON})
_KO_EVENTS = ("ko_pre", "ko_post")
#: the statistics field each stream is read from (for ``flagged_streams``)
_STREAM_SOURCES = {
    EventType.KO: "observations.probability",
    EventType.COUPON: "coupons.probability",
    EventType.KI: "ki_table",
    EventType.MATURITY_NO_KO: "maturity.probability",
    EventType.MATURITY_WITH_KI: "ki_settlement.probability",
}
_REBASED_SOURCES = {
    EventType.KO: "termination",
    EventType.MATURITY_WITH_KI: "termination",
    EventType.MATURITY_NO_KO: "termination",
    EventType.KI: "ki_table",
}


def _anchor(anchor) -> "SurvivalAnchor":
    if not isinstance(anchor, SurvivalAnchor):
        raise ValidationError(f"survival anchor must be a SurvivalAnchor, got {anchor!r}")
    return anchor


def knock_in_was_carried(product, lifecycle_state=None) -> bool:
    """Whether ``product`` was already knocked in at the valuation (the carried state the engines read): the OTC
    lifecycle flag of the autocallables (``_otc_lifecycle_knocked_in``), a DCN's ``knocked_in_at_valuation``, or a
    lifecycle state's ``knocked_in``. Feeds ``EventDistribution.from_statistics(..., knock_in_carried=...)``."""
    return bool(
        getattr(product, "_otc_lifecycle_knocked_in", False)
        or getattr(product, "knocked_in_at_valuation", False)
        or getattr(lifecycle_state, "knocked_in", False)
    )


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
        payment_times, payment_dates: settlement per stream (an array on the stream's axis, or a scalar). The
            termination streams' payment times (KO, MATURITY_NO_KO, MATURITY_WITH_KI; the REBASED rows each at
            their own settlement) are what ``SurvivalAnchor.PAYMENT`` reads (spec D32).
        path_dependent_payment_events: streams whose payment needs the engine's cash ledger.
        determination_times: determination time of each scalar outcome; one absent here is determined at the
            last survival time.
        provenance: which implementation produced the statistics and the price.
        probability_residual: the statistics' ``probability_residual``, absorbed into MATURITY_NO_KO by
            ``from_statistics`` (spec D16) and carried here so a reader sees it.
        rebased: a REBASED KO-reset (spec D13): events sit on the absolute candidate axis and legs value each
            event at its own time.
        flagged_streams: stream -> the statistics' ``unavailable`` reason for the field the stream was built from,
            when that field is listed there (for example pruned as not requested). A leg reading the stream can
            see that here.
        terminated: the note ended before the valuation (receivables only, spec §4.9c): no event is left, every
            reader of events refuses (a leg riding an ended note settles with it; its value is not derivable
            from events).
        terminal_times, terminal_probability: the maturity-type terminations (the never-knocked-in maturity, the
            knocked-in settlement; on a REBASED KO-reset every knocked-in settlement date) by determination time and
            mass. Spec D29: such a termination is alive at its determination and gone immediately after, so
            ``survival_at`` steps it out right after its time (never interpolated back) and ``survival_after`` reads
            past it; a knock-out leaves at its own row (``survival_probability`` at a node already excludes it).
        knock_in_carried: the note was already knocked in at the valuation. The knock-in is a past event (DESIGN
            §4.9c): the KI stream is empty -- nothing of the remaining life triggers a knock-in leg; cash a past
            knock-in decided is paid, or a receivable where the producer emits one. The statistics keep DESIGN
            §4.5's carried row (t = 0, P = 1) for readers.
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
    flagged_streams: Dict[EventType, str] = field(default_factory=dict)
    terminated: bool = False
    knock_in_carried: bool = False
    terminal_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    terminal_probability: np.ndarray = field(default_factory=lambda: np.zeros(0))

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
        object.__setattr__(self, "flagged_streams", dict(self.flagged_streams))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "knock_in_carried", bool(self.knock_in_carried))
        object.__setattr__(self, "terminal_times", np.asarray(self.terminal_times, dtype=float))
        object.__setattr__(self, "terminal_probability", np.asarray(self.terminal_probability, dtype=float))
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
    def ended(cls, provenance: Optional[StatisticsProvenance] = None) -> "EventDistribution":
        """The distribution of a note that ended before the valuation (spec §4.9c, AC17): no event is left.

        Empty and flagged ``terminated``; every event-dependent reader refuses it (``ensure_live``). Built for a
        terminated note with unpaid receivables (``from_statistics``) and for one fully settled (no statistics).
        """
        return cls(probabilities={}, survival_times=np.array([0.0]), survival_probability=np.array([0.0]),
                   provenance=provenance, probability_residual=0.0, terminated=True)

    @classmethod
    def from_statistics(cls, stats, *, knock_in_carried: bool = False) -> "EventDistribution":
        """The distribution of an :class:`EventStatistics` (DESIGN §4.11; spec D13, D16).

        ``knock_in_carried``: the note was already knocked in at the valuation (``knock_in_was_carried``). Its
        statistics hold DESIGN §4.5's single carried KI row (t = 0, P = 1); the knock-in is past (DESIGN §4.9c), so
        the KI stream is empty (a pruned table stays no stream) and the distribution is marked ``knock_in_carried``.

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
        * A terminated note (receivables only) is an empty distribution flagged ``terminated``; every reader of
          events refuses it.
        * Every stream records in ``flagged_streams`` the statistics' ``unavailable`` reason for the field it was
          built from (``EventStatistics.flag``), so a flagged number never reaches a leg unmarked.
        """
        if stats.terminated:
            return cls.ended(provenance=stats.provenance)
        if knock_in_carried:
            cls._require_carried_row(stats)
        if stats.termination is not None:
            return cls._from_rebased(stats, knock_in_carried=knock_in_carried)
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

        cls._add_ki_stream(stats, probabilities, stream_times, stream_dates, payment_times, payment_dates,
                           knock_in_carried=knock_in_carried)
        cls._add_outcome(EventType.MATURITY_NO_KO, stats.maturity, stats.maturity.probability + stats.probability_residual,
                         probabilities, payment_times, payment_dates)
        cls._add_outcome(EventType.MATURITY_WITH_KI, stats.ki_settlement, stats.ki_settlement.probability,
                         probabilities, payment_times, payment_dates)
        terminals = ((stats.maturity.time, stats.maturity.probability),
                     (stats.ki_settlement.time, stats.ki_settlement.probability))
        node_survival = cls._d29_absolute_survival(stats, obs.times, obs.survival, terminals)
        survival_times, survival = cls._survival_axis(obs.times, node_survival, stats.maturity.time)
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
            flagged_streams=cls._flags(stats, probabilities, _STREAM_SOURCES),
            knock_in_carried=knock_in_carried,
            terminal_times=np.array([t for t, _ in terminals], dtype=float),
            terminal_probability=np.array([max(float(p), 0.0) for _, p in terminals], dtype=float),
        )

    @staticmethod
    def _flags(stats, probabilities, sources) -> Dict[EventType, str]:
        """``unavailable`` reasons of the fields the emitted streams were read from."""
        flags = {}
        for event_type, path in sources.items():
            if event_type in probabilities:
                reason = stats.flag(path)
                if reason is not None:
                    flags[event_type] = reason
        return flags

    @staticmethod
    def _require_carried_row(stats) -> None:
        """A carried knock-in's statistics hold DESIGN §4.5's single row (t = 0, P = 1); no table (pruned) or an empty
        one (a note without a KI barrier, whose carried flag means nothing) carries nothing either way."""
        table = stats.ki_table
        if table is None or np.asarray(table.times).size == 0:
            return
        times = np.asarray(table.times, dtype=float)
        probability = np.asarray(table.probability, dtype=float)
        if times.size != 1 or not is_zero(float(times[0])) or not almost_equal(
                float(probability[0]), 1.0, tol=Tolerance.PROBABILITY):
            raise ValidationError(
                "a carried knock-in's KI table is the single row at t = 0 with probability 1 (DESIGN §4.5); got "
                f"times {times.tolist()}, probabilities {probability.tolist()}"
            )

    @classmethod
    def _from_rebased(cls, stats, *, knock_in_carried: bool = False) -> "EventDistribution":
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
        cls._add_ki_stream(stats, probabilities, stream_times, stream_dates, payment_times, payment_dates,
                           knock_in_carried=knock_in_carried)
        survival_times, survival = cls._survival_axis(term.survival_times, term.survival, maturity_time)
        settles = np.isin(events, ("maturity", "ki_settlement"))
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
            flagged_streams=cls._flags(stats, probabilities, _REBASED_SOURCES),
            knock_in_carried=knock_in_carried,
            terminal_times=term.times[settles],
            terminal_probability=term.probability[settles],
        )

    @staticmethod
    def _add_ki_stream(stats, probabilities, stream_times, stream_dates, payment_times, payment_dates, *,
                       knock_in_carried: bool = False) -> None:
        table = stats.ki_table
        if table is None:
            return
        if knock_in_carried:
            # the knock-in happened before the valuation: no knock-in event is left in the remaining life
            empty = np.zeros(0, dtype=float)
            probabilities[EventType.KI] = empty
            stream_times[EventType.KI] = empty
            payment_times[EventType.KI] = empty
            if table.dates is not None:
                stream_dates[EventType.KI] = ()
            if table.payment_dates is not None:
                payment_dates[EventType.KI] = ()
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
    def _d29_absolute_survival(stats, times, survival, terminals) -> np.ndarray:
        """``ObservationTable.survival`` restated by spec D29 at each row: a maturity-type termination counts alive at
        its own determination and leaves after it. The tables drop the first schedule's maturity mass of a KO-reset
        AT its determination (F5, when it precedes the last row): that mass is added back on the rows at its time;
        a termination before the last row that the tables never drop (a knocked-in settlement inside the axis)
        leaves on the rows after it. Rows that need neither keep the table's value bit for bit."""
        times = np.asarray(times, dtype=float)
        survival = np.asarray(survival, dtype=float)
        if times.size == 0:
            return survival
        last = float(times[-1])
        out = survival.copy()
        maturity = stats.maturity
        f5 = (stats.product_family == "ko_reset" and stats.termination is None
              and float(maturity.time) < last - Tolerance.ZERO)
        if f5:
            at = (times >= float(maturity.time)) & (times <= float(maturity.time) + Tolerance.ZERO)
            out[at] = out[at] + float(maturity.probability)
        for k, (t_end, mass) in enumerate(terminals):
            if f5 and k == 0:
                continue                                   # the tables already drop it after its time
            after = times > float(t_end) + Tolerance.ZERO
            if np.any(after) and float(mass) != 0.0:
                out[after] = out[after] - float(mass)
        return out

    @staticmethod
    def _survival_axis(times, survival, terminal_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Survival from the valuation on: 1 at 0 unless an observation sits at 0 (its post-event survival)."""
        times = np.asarray(times, dtype=float)
        survival = np.asarray(survival, dtype=float)
        if times.size > 1:
            # one node per time, holding the value after its last row (a tied first-schedule row repeats a time):
            # the same nodes on every layout of one contract, so reads between nodes agree (spec D29)
            keep = np.r_[np.abs(np.diff(times)) > Tolerance.ZERO, True]
            times, survival = times[keep], survival[keep]
        if times.size == 0:
            end = max(float(terminal_time), 0.0)
            if at_valuation_instant(end):
                return np.array([0.0]), np.array([1.0])
            return np.array([0.0, end]), np.array([1.0, 1.0])
        if at_valuation_instant(float(times[0])):
            return times, survival
        return np.r_[0.0, times], np.r_[1.0, survival]

    # --- reading ----------------------------------------------------------------------------------------

    def ensure_live(self, reader: str = "an event-dependent reader") -> None:
        """Refuse a terminated note: no event is left, so nothing event-dependent can be valued from it.

        Raises:
            ValidationError: the distribution is ``terminated`` (receivables only).
        """
        if self.terminated:
            raise ValidationError(
                f"{reader} cannot read the event distribution of a terminated note (receivables only): no "
                "event is left; a leg riding an ended note settles with it, its value is not derivable from events"
            )

    def payment_times_for(
        self,
        event_type: EventType,
    ) -> Union[np.ndarray, float]:
        """Return fixed payment timing, or reject path-dependent settlement."""

        self.ensure_live(f"payment_times_for({event_type.value})")
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

    def survival_at(self, t: float, *, anchor: SurvivalAnchor = SurvivalAnchor.DETERMINATION) -> float:
        """Survival probability at a year fraction (spec D29; ``anchor``: spec D32).

        ``anchor=SurvivalAnchor.PAYMENT`` reads every termination -- knock-out, never-knocked-in maturity, knocked-in
        settlement, each REBASED path's own settlement -- as alive until its own payment time and gone after it:
        ``1 - (termination mass paid strictly before t)``, from the per-stream probabilities and payment times (a
        step; nothing is interpolated, so a paid note is never revived). It raises ``ValidationError`` when a
        termination stream carries no payment time (``_paid_terminations``). The rest of this docstring is the
        ``DETERMINATION`` (default) rule.

        At a node it is the node's value: knock-outs at that time excluded, maturity-type terminations at that time
        still alive. Between nodes the knock-out part is interpolated linearly, while a maturity-type termination
        leaves as a step right after its own time (a settled note is never interpolated back to life). A strictly
        past time reads 1. A time at the valuation instant (``is_zero``) reads the post-observation survival of an
        event the grid keeps at 0 — a theta-shifted contract keeps the observation landing on its new origin — else 1.

        "Gone after" counts from the termination's DETERMINATION time, never its payment time (CP8 ruling I1), as a
        knock-out leaves at its observation row although its cash settles later: a note whose maturity is determined
        at T and settles at T + lag reads 0 for every t > T. A COMPLETE_PERIOD leg whose last period ends on such a
        later settlement date therefore pays nothing on the matured mass (pinned by
        ``test/cashleg/test_d29_determination_time.py``). ``SurvivalAnchor.PAYMENT`` (spec D32) is the opt-in rule
        that counts from the payment time instead.
        """

        self.ensure_live("survival_at")
        if _anchor(anchor) is SurvivalAnchor.PAYMENT:
            return self._paid_survival(float(t), entering=False)
        t = float(t)
        time_grid, survival_grid = self.survival_times, self.survival_probability
        if is_zero(t):
            if at_valuation_instant(float(time_grid[0])):
                return float(survival_grid[0])
            return 1.0
        if t < 0.0:
            return 1.0
        if self.terminal_times.size:
            # the knock-out part: every node with the maturity-type mass determined before it put back
            return float(self._interpolate(t, time_grid, self._knock_out_part)) - self._settled_before(t)
        return float(self._interpolate(t, time_grid, survival_grid))

    def survival_after(self, t: float, *, anchor: SurvivalAnchor = SurvivalAnchor.DETERMINATION) -> float:
        """Survival just after ``t``: ``survival_at(t)`` less the maturity-type terminations determined at ``t``
        (spec D29: a note settled at ``t`` does not enter a period starting at ``t``). ``t`` is compared with the
        terminations' determination times, not their payment times (CP8 ruling I1; see ``survival_at``).

        ``anchor=SurvivalAnchor.PAYMENT`` (spec D32): ``survival_at(t, anchor=PAYMENT)`` less every termination PAID
        at ``t`` (a note paid at ``t`` does not enter a period starting at ``t``)."""
        if _anchor(anchor) is SurvivalAnchor.PAYMENT:
            self.ensure_live("survival_after")
            return self._paid_survival(float(t), entering=True)
        settled_now = 0.0
        if self.terminal_times.size and float(t) > 0.0:
            hit = np.abs(self.terminal_times - float(t)) <= Tolerance.ZERO
            settled_now = float(np.sum(self.terminal_probability[hit]))
        return self.survival_at(t) - settled_now

    @cached_property
    def _knock_out_part(self) -> np.ndarray:
        """The D29 survival nodes with the maturity-type mass determined before each node put back (the part
        ``survival_at`` interpolates). Computed once per distribution; the arithmetic is the per-read one it
        replaces, so every read is bit-identical (CP8 minor: it was rebuilt on every read, quadratic on REBASED)."""
        settled_before = np.array([self._settled_before(float(u)) for u in self.survival_times])
        return self.survival_probability + settled_before

    def _settled_before(self, t: float) -> float:
        """Maturity-type mass determined strictly before ``t`` (beyond ``Tolerance.ZERO``)."""
        return float(np.sum(self.terminal_probability[self.terminal_times < t - Tolerance.ZERO]))

    @cached_property
    def _paid_terminations(self) -> tuple[np.ndarray, np.ndarray]:
        """Payment time and mass of every termination row (spec D32): the KO stream's rows, the never-knocked-in
        maturity (its probability carries the D16 residual) and the knocked-in settlement (a scalar, or a REBASED
        path-dependent settlement's rows on the candidate axis, each at its own payment time).

        Raises:
            ValidationError: a termination stream with probability carries no fixed payment time (path-dependent, or
                absent from ``payment_times``) -- a PAYMENT read is refused, never approximated.
        """
        times, masses = [], []
        for event_type in (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI):
            if event_type not in self.probabilities:
                continue
            if event_type in self.path_dependent_payment_events:
                raise ValidationError(
                    f"survival to the payment time (SurvivalAnchor.PAYMENT) needs the {event_type.value} payment "
                    "times, which are path-dependent here (the engine's cash ledger holds them); no fallback"
                )
            if event_type not in self.payment_times:
                raise ValidationError(
                    f"survival to the payment time (SurvivalAnchor.PAYMENT) needs the {event_type.value} payment "
                    "times, which this distribution does not carry; no fallback"
                )
            mass = np.atleast_1d(np.asarray(self.probabilities[event_type], dtype=float))
            pay = np.atleast_1d(np.asarray(self.payment_times[event_type], dtype=float))
            if pay.shape != mass.shape:
                raise ValidationError(
                    f"{event_type.value} carries {pay.size} payment times for {mass.size} probability rows"
                )
            times.append(pay)
            masses.append(mass)
        if not times:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(times), np.concatenate(masses)

    def _paid_survival(self, t: float, *, entering: bool) -> float:
        """Spec D32: 1 - the termination mass paid strictly before ``t`` (beyond ``Tolerance.ZERO``); ``entering``
        also removes the mass paid at ``t``. A time before the valuation reads 1."""
        pay, mass = self._paid_terminations
        if t < 0.0 and not is_zero(t):
            return 1.0
        gone = pay < t - Tolerance.ZERO
        if entering:
            gone = gone | (np.abs(pay - t) <= Tolerance.ZERO)
        return 1.0 - float(np.sum(mass[gone]))

    @staticmethod
    def _interpolate(t: float, time_grid: np.ndarray, values: np.ndarray) -> float:
        if t <= time_grid[0]:
            return float(values[0])
        if t >= time_grid[-1]:
            return float(values[-1])
        return float(np.interp(t, time_grid, values))

    def normalized(self) -> "EventDistribution":
        """Return a copy with tiny termination-probability drift normalized."""

        if self.terminated:
            return self
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
            flagged_streams=self.flagged_streams,
            knock_in_carried=self.knock_in_carried,
            terminal_times=self.terminal_times,
            terminal_probability=self.terminal_probability,
        )

    # --- invariants -------------------------------------------------------------------------------------

    def _validate_invariants(self) -> None:
        self._validate_survival()
        if self.terminated:
            if self.probabilities or self.stream_times or self.payment_times or self.rebased:
                raise NumericalError("a terminated EventDistribution carries no event streams")
            if self.probability_residual != 0.0:
                raise NumericalError("a terminated EventDistribution has no probability residual")
            return
        for event_type in self.flagged_streams:
            if event_type not in self.probabilities:
                raise NumericalError(f"EventDistribution flags {event_type.value}, which is not a stream")
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
        if self.terminal_times.shape != self.terminal_probability.shape or self.terminal_times.ndim != 1:
            raise NumericalError("EventDistribution terminal_times / terminal_probability must align (1D)")
        if np.any(~np.isfinite(self.terminal_times)) or np.any(~np.isfinite(self.terminal_probability)):
            raise NumericalError("EventDistribution terminal steps contain non-finite values")
        if np.any(self.terminal_probability < -Tolerance.PROBABILITY):
            raise NumericalError("EventDistribution terminal_probability is negative")
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
        if not at_valuation_instant(float(times[0])):
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
