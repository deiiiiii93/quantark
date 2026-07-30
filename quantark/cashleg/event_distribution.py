"""Engine-emitted timing distributions for cash-leg valuation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np

from quantark.asset.equity.engine.event_stats import (
    AutocallableEventStats,
    PhoenixEventStats,
)
from quantark.util.exceptions import NumericalError
from quantark.util.numerical import Tolerance, almost_equal


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


@dataclass(frozen=True)
class EventDistribution:
    """Probability distribution over termination and auxiliary payment events.

    ``event_times`` is the event grid in year fractions. For ordinary engine
    output this is the observation grid and ``survival_probability`` has one
    additional leading value at time 0. For the trivial vanilla fallback, the
    grid includes 0.0 and maturity and the survival vector has the same length.
    """

    event_times: np.ndarray
    event_dates: Optional[List[datetime]]
    probabilities: Dict[EventType, Union[np.ndarray, float]]
    survival_probability: np.ndarray
    payment_times: Dict[EventType, Union[np.ndarray, float]] = field(
        default_factory=dict
    )
    payment_dates: Optional[
        Dict[EventType, Union[tuple[datetime, ...], datetime]]
    ] = None
    mc_ko_times: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_times", np.asarray(self.event_times, dtype=float))
        object.__setattr__(
            self,
            "survival_probability",
            np.asarray(self.survival_probability, dtype=float),
        )
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
        self._validate_invariants()

    @classmethod
    def trivial(cls, maturity: float) -> "EventDistribution":
        """Return a no-KO distribution with one terminal maturity outcome."""

        maturity = float(maturity)
        return cls(
            event_times=np.array([0.0, maturity], dtype=float),
            event_dates=None,
            probabilities={EventType.MATURITY_NO_KO: 1.0},
            survival_probability=np.array([1.0, 1.0], dtype=float),
            payment_times={
                EventType.MATURITY_NO_KO: maturity,
                EventType.MATURITY_WITH_KI: maturity,
            },
        )

    @classmethod
    def from_autocallable_stats(
        cls, stats: AutocallableEventStats
    ) -> "EventDistribution":
        """Convert the existing engine event-stats API into EventDistribution."""

        ko_times = np.asarray(stats.ko_times, dtype=float)
        ko_probability = np.asarray(stats.ko_probability, dtype=float)
        survival = np.concatenate(
            [[1.0], np.asarray(stats.survival_probability, dtype=float)]
        )

        ko_total = float(np.sum(ko_probability))
        maturity_probability = max(0.0, 1.0 - ko_total)
        # MATURITY_WITH_KI is the mass that reaches maturity in the knocked-in
        # state, so it must use P(KI ever AND never KO), NOT P(KI ever): a path
        # that knocks in and later autocalls redeems at par and carries no KI
        # loss. Prefer the unambiguous `ki_survive_knocked_in_probability`; fall
        # back to the legacy `ki_probability` for engines that do not populate it
        # (QUAD/PDE already report the "settles knocked-in" value there).
        ki_settle = stats.ki_survive_knocked_in_probability
        if ki_settle is None:
            ki_settle = stats.ki_probability
        maturity_with_ki = min(maturity_probability, max(0.0, float(ki_settle)))
        maturity_no_ko = max(0.0, maturity_probability - maturity_with_ki)

        probabilities: Dict[EventType, Union[np.ndarray, float]] = {
            EventType.KO: ko_probability,
            EventType.MATURITY_NO_KO: maturity_no_ko,
            EventType.MATURITY_WITH_KI: maturity_with_ki,
        }

        ki_probability = np.asarray(
            getattr(stats, "ki_event_probability", np.array([])), dtype=float
        )
        if ki_probability.size == ko_times.size:
            probabilities[EventType.KI] = ki_probability
        if isinstance(stats, PhoenixEventStats) and stats.coupon_probability.size > 0:
            probabilities[EventType.COUPON] = np.asarray(
                stats.coupon_probability, dtype=float
            )

        payment_times: Dict[EventType, Union[np.ndarray, float]] = {}
        payment_dates = {}
        event_dates = None
        ledger_payment_times = np.asarray(
            getattr(stats, "payment_times", np.array([])),
            dtype=float,
        )
        n_ko = ko_times.size
        if ledger_payment_times.size >= n_ko + 1:
            payment_times[EventType.KO] = ledger_payment_times[:n_ko]
            next_index = n_ko
            if (
                isinstance(stats, PhoenixEventStats)
                and stats.coupon_probability.size == n_ko
                and ledger_payment_times.size >= 2 * n_ko + 1
            ):
                payment_times[EventType.COUPON] = ledger_payment_times[
                    n_ko : 2 * n_ko
                ]
                next_index = 2 * n_ko
            terminal_payment_time = float(ledger_payment_times[-1])
            payment_times[EventType.MATURITY_NO_KO] = terminal_payment_time
            payment_times[EventType.MATURITY_WITH_KI] = terminal_payment_time

            ledger_payment_dates = getattr(stats, "payment_dates", None)
            ledger_determination_dates = getattr(
                stats,
                "determination_dates",
                None,
            )
            if ledger_determination_dates is not None:
                event_dates = list(ledger_determination_dates[:n_ko])
            if ledger_payment_dates is not None:
                payment_dates[EventType.KO] = tuple(
                    ledger_payment_dates[:n_ko]
                )
                if next_index == 2 * n_ko:
                    payment_dates[EventType.COUPON] = tuple(
                        ledger_payment_dates[n_ko:next_index]
                    )
                payment_dates[EventType.MATURITY_NO_KO] = (
                    ledger_payment_dates[-1]
                )
                payment_dates[EventType.MATURITY_WITH_KI] = (
                    ledger_payment_dates[-1]
                )

        dist = cls(
            event_times=ko_times,
            event_dates=event_dates,
            probabilities=probabilities,
            survival_probability=survival,
            payment_times=payment_times,
            payment_dates=payment_dates or None,
        )
        return dist.normalized()

    def payment_times_for(
        self,
        event_type: EventType,
    ) -> Union[np.ndarray, float]:
        """Return payment timing, falling back to determination timing."""

        if event_type in self.payment_times:
            return self.payment_times[event_type]
        probability = self.probabilities.get(event_type)
        if isinstance(probability, np.ndarray):
            return self.event_times
        return float(self.event_times[-1])

    def survival_at(self, t: float) -> float:
        """Linearly interpolate survival probability at a year fraction."""

        t = float(t)
        if t <= 0.0:
            return 1.0

        time_grid, survival_grid = self._survival_grid()
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
        elif EventType.MATURITY_WITH_KI in probabilities:
            probabilities[EventType.MATURITY_WITH_KI] = max(
                0.0, float(probabilities[EventType.MATURITY_WITH_KI]) - drift
            )
        else:
            return self

        return EventDistribution(
            event_times=self.event_times,
            event_dates=self.event_dates,
            probabilities=probabilities,
            survival_probability=self.survival_probability,
            payment_times=self.payment_times,
            payment_dates=self.payment_dates,
            mc_ko_times=self.mc_ko_times,
        )

    def _validate_invariants(self) -> None:
        if self.event_times.ndim != 1:
            raise NumericalError("EventDistribution.event_times must be 1D")
        if self.survival_probability.ndim != 1:
            raise NumericalError("EventDistribution.survival_probability must be 1D")
        if len(self.event_times) == 0:
            raise NumericalError("EventDistribution.event_times must not be empty")
        if np.any(~np.isfinite(self.event_times)):
            raise NumericalError("EventDistribution.event_times contains non-finite values")
        if np.any(~np.isfinite(self.survival_probability)):
            raise NumericalError(
                "EventDistribution.survival_probability contains non-finite values"
            )
        if np.any(np.diff(self.event_times) < -Tolerance.PROBABILITY):
            raise NumericalError("EventDistribution.event_times must be sorted")

        self._validate_survival_shape()

        if (
            abs(float(self.survival_probability[0]) - 1.0)
            > Tolerance.PROBABILITY
        ):
            raise NumericalError(
                f"survival_probability[0] = {self.survival_probability[0]}, expected 1.0"
            )

        diffs = np.diff(self.survival_probability)
        if np.any(diffs > Tolerance.PROBABILITY):
            raise NumericalError(
                "EventDistribution survival_probability is not monotone non-increasing"
            )

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
            if isinstance(probability, np.ndarray) and probability.shape != self.event_times.shape:
                raise NumericalError(
                    f"EventDistribution probability {event_type.value} has shape "
                    f"{probability.shape}, expected {self.event_times.shape}"
                )

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
                if times.shape != self.event_times.shape:
                    raise NumericalError(
                        f"EventDistribution payment time {event_type.value} has "
                        f"shape {times.shape}, expected {self.event_times.shape}"
                    )
                if np.any(
                    times + Tolerance.PROBABILITY < self.event_times
                ):
                    raise NumericalError(
                        f"EventDistribution payment time {event_type.value} "
                        "precedes its determination time"
                    )
            elif times.ndim != 0:
                raise NumericalError(
                    f"EventDistribution scalar event {event_type.value} requires "
                    "a scalar payment time"
                )
            elif (
                float(times) + Tolerance.PROBABILITY
                < float(self.event_times[-1])
            ):
                raise NumericalError(
                    f"EventDistribution payment time {event_type.value} "
                    "precedes terminal determination"
                )

        total = self._termination_probability_total()
        if not almost_equal(total, 1.0, tol=Tolerance.PROBABILITY):
            raise NumericalError(
                f"EventDistribution termination probability sum = {total}, expected 1.0 "
                f"(tolerance {Tolerance.PROBABILITY})"
            )

    def _validate_survival_shape(self) -> None:
        n_times = len(self.event_times)
        n_survival = len(self.survival_probability)
        starts_at_zero = almost_equal(
            float(self.event_times[0]), 0.0, tol=Tolerance.PROBABILITY
        )
        expected_lengths = {n_times + 1}
        if starts_at_zero:
            expected_lengths.add(n_times)
        if n_survival not in expected_lengths:
            raise NumericalError(
                f"len(survival_probability) = {n_survival}, expected one of "
                f"{sorted(expected_lengths)} for {n_times} event times"
            )

    def _termination_probability_total(self) -> float:
        total = 0.0
        for event_type, probability in self.probabilities.items():
            if event_type not in _TERMINATION_EVENTS:
                continue
            total += float(np.asarray(probability, dtype=float).sum())
        return total

    def _survival_grid(self) -> tuple[np.ndarray, np.ndarray]:
        starts_at_zero = almost_equal(
            float(self.event_times[0]), 0.0, tol=Tolerance.PROBABILITY
        )
        if len(self.survival_probability) == len(self.event_times):
            return self.event_times, self.survival_probability
        if starts_at_zero:
            return self.event_times, self.survival_probability[1:]
        return (
            np.concatenate([[0.0], self.event_times]),
            self.survival_probability,
        )


@dataclass(frozen=True)
class PricingResult:
    """Engine result containing NPV and optional event timing distribution."""

    npv: float
    event_distribution: Optional[EventDistribution] = None
