"""
Observation schedule structures for barrier-like products.
"""

from dataclasses import dataclass, field
from datetime import datetime
from math import isclose
from typing import List, Optional

from util.enum import ObservationAggregation
from util.exceptions import ValidationError

try:
    from util.calendar import calculate_year_fraction
except Exception:
    # Fallback for environments where calendar utilities are unavailable at import time.
    calculate_year_fraction = None


@dataclass
class ObservationRecord:
    """Single observation entry with optional per-date barrier and payoff data."""

    observation_time: Optional[float] = None
    observation_date: Optional[datetime] = None
    barrier: Optional[float] = None
    upper_barrier: Optional[float] = None
    lower_barrier: Optional[float] = None
    payoff: Optional[float] = None
    return_rate: Optional[float] = None

    def resolve_time(self, pricing_env) -> float:
        """Resolve observation time to a year fraction."""
        if self.observation_time is not None:
            return self.observation_time
        if self.observation_date is not None:
            if pricing_env is None or calculate_year_fraction is None:
                raise ValidationError(
                    "PricingEnvironment is required to resolve observation_date."
                )
            return calculate_year_fraction(
                pricing_env.valuation_date,
                self.observation_date,
                pricing_env.day_count_convention,
                pricing_env.bus_days_in_year,
            )
        raise ValidationError("ObservationRecord requires observation_time or observation_date.")

    def get_payoff(self, default_payoff: float) -> float:
        """Return the payoff for this record, falling back to a provided default."""
        if self.payoff is None and self.return_rate is None:
            return default_payoff
        if self.payoff is not None:
            return self.payoff
        return self.return_rate if self.return_rate is not None else default_payoff

    def validate(self, require_single: bool = False, require_double: bool = False) -> None:
        """Validate record fields."""
        if self.observation_time is None and self.observation_date is None:
            raise ValidationError("ObservationRecord must provide observation_time or observation_date.")
        if require_single:
            if self.barrier is None:
                raise ValidationError("Single-barrier observation requires barrier level.")
            if self.upper_barrier is not None or self.lower_barrier is not None:
                raise ValidationError("Single-barrier observation must not include upper/lower barriers.")
        if require_double:
            if self.upper_barrier is None or self.lower_barrier is None:
                raise ValidationError("Double-barrier observation requires both upper_barrier and lower_barrier.")
            if self.upper_barrier <= 0 or self.lower_barrier <= 0:
                raise ValidationError("Barrier levels must be positive.")
            if self.lower_barrier >= self.upper_barrier:
                raise ValidationError("lower_barrier must be less than upper_barrier.")


@dataclass
class ResolvedObservationRecord:
    """ObservationRecord with resolved time and concrete barrier/payoff values."""

    observation_time: float
    barrier: Optional[float] = None
    upper_barrier: Optional[float] = None
    lower_barrier: Optional[float] = None
    payoff: float = 0.0
    return_rate: Optional[float] = None


@dataclass
class ObservationSchedule:
    """Ordered schedule of observation records with aggregation semantics."""

    records: List[ObservationRecord] = field(default_factory=list)
    aggregation_mode: ObservationAggregation = ObservationAggregation.STOP_FIRST_HIT
    frequency: Optional[float] = None

    def validate(self, require_single: bool = False, require_double: bool = False) -> None:
        """Validate schedule ordering and record completeness."""
        if len(self.records) == 0:
            raise ValidationError("ObservationSchedule must contain at least one record for discrete monitoring.")
        if not isinstance(self.aggregation_mode, ObservationAggregation):
            raise ValidationError(f"Invalid aggregation mode: {self.aggregation_mode}")
        times: List[float] = []
        for rec in self.records:
            rec.validate(require_single=require_single, require_double=require_double)
            if rec.observation_time is not None:
                times.append(rec.observation_time)
        if times and not self._is_sorted(times):
            raise ValidationError("Observation records must be ordered by observation_time.")
        if self.frequency is not None and self.frequency <= 0:
            raise ValidationError("frequency must be positive when provided.")
        if times:
            self._validate_frequency(times)

    def resolve(
        self,
        pricing_env,
        default_barrier: Optional[float] = None,
        default_upper: Optional[float] = None,
        default_lower: Optional[float] = None,
        default_payoff: float = 0.0,
        require_single: bool = False,
        require_double: bool = False,
    ) -> List[ResolvedObservationRecord]:
        """Resolve records to concrete numeric times and defaults."""
        self.validate(require_single=require_single, require_double=require_double)
        resolved: List[ResolvedObservationRecord] = []
        times: List[float] = []
        for rec in self.records:
            t = rec.resolve_time(pricing_env)
            times.append(t)
            resolved.append(
                ResolvedObservationRecord(
                    observation_time=t,
                    barrier=rec.barrier if rec.barrier is not None else default_barrier,
                    upper_barrier=rec.upper_barrier if rec.upper_barrier is not None else default_upper,
                    lower_barrier=rec.lower_barrier if rec.lower_barrier is not None else default_lower,
                    payoff=rec.get_payoff(default_payoff),
                    return_rate=rec.return_rate,
                )
            )
        if times and not self._is_sorted(times):
            raise ValidationError("Observation records must be ordered by resolved observation time.")
        # Keep inferred frequency handy when available
        self._validate_frequency(times)
        return resolved

    def infer_frequency(self, times: List[float], tolerance: float = 1e-8) -> Optional[float]:
        """Infer regular spacing frequency from a list of times."""
        if len(times) < 2:
            return None
        deltas = [round(times[i + 1] - times[i], 12) for i in range(len(times) - 1)]
        base = deltas[0]
        for d in deltas[1:]:
            if not isclose(d, base, rel_tol=tolerance, abs_tol=tolerance):
                return None
        return base

    def ensure_regular_frequency(self, times: List[float], tolerance: float = 1e-8) -> float:
        """Ensure schedule has a regular observation frequency; return the interval."""
        inferred = self.infer_frequency(times, tolerance=tolerance)
        effective = self.frequency if self.frequency is not None else inferred
        if effective is None:
            raise ValidationError("ObservationSchedule requires a regular frequency for analytical usage.")
        if inferred is not None and not isclose(effective, inferred, rel_tol=tolerance, abs_tol=tolerance):
            raise ValidationError(
                f"ObservationSchedule frequency {effective} does not match inferred spacing {inferred}."
            )
        return effective

    def has_fixed_payoff(self, default_payoff: float = 0.0, tolerance: float = 1e-8) -> bool:
        """Check if all records share the same payoff within tolerance."""
        base = self.records[0].get_payoff(default_payoff)
        for rec in self.records[1:]:
            if not isclose(rec.get_payoff(default_payoff), base, rel_tol=tolerance, abs_tol=tolerance):
                return False
        return True

    def assert_analytical_ready(self, default_payoff: float = 0.0, tolerance: float = 1e-8) -> None:
        """
        Validate preconditions for analytical engines that rely on barrier shift:
        regular observation frequency and fixed payoff across observations.
        """
        times = [rec.observation_time for rec in self.records if rec.observation_time is not None]
        self.ensure_regular_frequency(times, tolerance=tolerance)
        if not self.has_fixed_payoff(default_payoff=default_payoff, tolerance=tolerance):
            raise ValidationError("Analytical barrier shift requires a fixed payoff across observation records.")

    @property
    def times(self) -> List[float]:
        """Return observation_time values when present."""
        return [rec.observation_time for rec in self.records if rec.observation_time is not None]

    @classmethod
    def from_legacy(
        cls,
        observation_dates: List[float],
        default_barrier: Optional[float],
        default_payoff: float,
        aggregation_mode: ObservationAggregation = ObservationAggregation.STOP_FIRST_HIT,
        frequency: Optional[float] = None,
        upper_barrier: Optional[float] = None,
        lower_barrier: Optional[float] = None,
    ) -> "ObservationSchedule":
        """Create a schedule from legacy observation_dates and uniform barrier/payoff."""
        if observation_dates is None or len(observation_dates) == 0:
            raise ValidationError("observation_dates required to build ObservationSchedule from legacy fields.")
        records = [
            ObservationRecord(
                observation_time=t,
                barrier=default_barrier,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
                payoff=default_payoff,
            )
            for t in observation_dates
        ]
        schedule = cls(records=records, aggregation_mode=aggregation_mode, frequency=frequency)
        schedule.validate(require_single=upper_barrier is None and lower_barrier is None, require_double=upper_barrier is not None or lower_barrier is not None)
        return schedule

    @staticmethod
    def _is_sorted(values: List[float]) -> bool:
        """Check if list is non-decreasing."""
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def _validate_frequency(self, times: List[float]) -> None:
        """Validate provided frequency against observation times when available."""
        if not times or len(times) < 2:
            return
        if self.frequency is None:
            return
        inferred = self.infer_frequency(times)
        if inferred is None:
            raise ValidationError("Provided frequency requires regularly spaced observation_times.")
        if not isclose(self.frequency, inferred, rel_tol=1e-8, abs_tol=1e-8):
            raise ValidationError(
                f"frequency {self.frequency} does not match inferred interval {inferred} from observation_times."
            )

