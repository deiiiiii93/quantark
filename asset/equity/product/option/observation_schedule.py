"""
Observation schedule structures for barrier-like products.
"""

from dataclasses import dataclass, field
from datetime import datetime
from math import isclose
from typing import List, Optional, TYPE_CHECKING

from util.enum import ObservationAggregation, TenorEnd, ObservationFrequency
from util.calendar import DayCountConvention
from util.exceptions import ValidationError

if TYPE_CHECKING:
    from priceenv import PricingEnvironment

# Type alias for cleaner signatures
PricingEnv = Optional["PricingEnvironment"]


try:
    from util.calendar import calculate_year_fraction, calculate_day_count_fraction
except ImportError:
    # Fallback for environments where calendar utilities are unavailable at import time.
    calculate_year_fraction = None
    calculate_day_count_fraction = None


@dataclass
class ObservationRecord:
    """
    Single observation entry with optional per-date barrier and payoff data.

    Fields for annualized rate handling:
        is_rate_annualized: Whether return_rate is annualized (default: False)
        initial_date: Start date for accrual period calculation
        settlement_date: Settlement date (used if tenor_end = SETTLEMENT)
        maturity_date: Maturity date (used if tenor_end = MATURITY)
        day_count_convention: Per-observation day count convention override
        tenor_end: Defines end date for accrual (SETTLEMENT/MATURITY/default to observation)
        day_count_fraction: Pre-calculated day count fraction (optional, computed if not provided)
    """

    observation_time: Optional[float] = None
    observation_date: Optional[datetime] = None
    barrier: Optional[float] = None
    upper_barrier: Optional[float] = None
    lower_barrier: Optional[float] = None
    payoff: Optional[float] = None
    return_rate: Optional[float] = None
    is_rate_annualized: Optional[bool] = False

    # Date-based accrual fields
    initial_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    day_count_convention: Optional[DayCountConvention] = None
    tenor_end: Optional[TenorEnd] = None
    day_count_fraction: Optional[float] = None

    def resolve_time(self, pricing_env: PricingEnv) -> float:
        """Resolve observation time to a year fraction.
        
        Args:
            pricing_env: PricingEnvironment containing valuation date and calendar conventions
            
        Returns:
            Year fraction from valuation date to observation date
            
        Raises:
            ValidationError: If observation time/date cannot be resolved
        """
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

    def get_payoff(self, default_payoff: float, pricing_env: PricingEnv = None) -> float:
        """
        Return the payoff for this record, following the calculation flow:
        1. If payoff is explicitly set, return it
        2. If return_rate is not annualized, return return_rate
        3. If day_count_fraction is pre-calculated, return day_count_fraction * return_rate
        4. Calculate day_count_fraction based on tenor_end and dates, then return day_count_fraction * return_rate
        
        Args:
            default_payoff: Default payoff value if not specified
            pricing_env: Optional pricing environment for date resolution
            
        Returns:
            Resolved payoff value for this observation
        """
        # If explicit payoff is set, use it
        if self.payoff is not None:
            return self.payoff

        # If no return_rate, use default
        if self.return_rate is None:
            return default_payoff

        # If return_rate is not annualized, return it directly
        if not self.is_rate_annualized:
            return self.return_rate

        # If day_count_fraction is pre-calculated, use it
        if self.day_count_fraction is not None:
            return self.day_count_fraction * self.return_rate

        # Calculate day_count_fraction based on tenor_end
        if self.initial_date is None:
            # Cannot calculate day_count_fraction without initial_date
            return self.return_rate

        # Determine end_date based on tenor_end
        end_date = self._determine_end_date()
        if end_date is None:
            # Cannot calculate day_count_fraction without end_date
            return self.return_rate

        # Calculate day_count_fraction
        return self._calculate_annualized_payoff(end_date)

    def _determine_end_date(self) -> Optional[datetime]:
        """Determine the end date for accrual period based on tenor_end setting."""
        if self.tenor_end == TenorEnd.SETTLEMENT and self.settlement_date is not None:
            return self.settlement_date
        if self.tenor_end == TenorEnd.MATURITY and self.maturity_date is not None:
            return self.maturity_date
        if self.observation_date is not None:
            return self.observation_date
        return None

    def _calculate_annualized_payoff(self, end_date: datetime) -> float:
        """Calculate annualized payoff using day count fraction."""
        if self.day_count_convention is not None and calculate_day_count_fraction is not None:
            try:
                dcf = calculate_day_count_fraction(
                    self.initial_date, end_date, self.day_count_convention
                )
                return dcf * self.return_rate
            except ImportError:
                # If calculation fails due to import issues, return raw return_rate
                return self.return_rate

        # Fallback to raw return_rate if no day count convention
        return self.return_rate

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

        # Validate date consistency for annualized rates
        if self.is_rate_annualized and self.return_rate is not None:
            if self.day_count_fraction is None and self.initial_date is not None:
                # Check that we have sufficient info to calculate day_count_fraction
                has_end_date = (
                    (self.tenor_end == TenorEnd.SETTLEMENT and self.settlement_date is not None)
                    or (self.tenor_end == TenorEnd.MATURITY and self.maturity_date is not None)
                    or self.observation_date is not None
                )
                if not has_end_date:
                    raise ValidationError(
                        "Annualized return_rate requires either day_count_fraction, "
                        "or initial_date with appropriate tenor_end and corresponding date."
                    )


@dataclass
class ResolvedObservationRecord:
    """
    ObservationRecord with resolved time and concrete barrier/payoff values.

    This is the final state used by pricing engines. It contains only:
    - observation_time: When to check for barrier breach
    - barrier levels: What levels to check against
    - payoff: The final payoff amount if barrier is hit
    - settlement_time: When the payoff is settled (for discounting)

    All intermediate calculation values (return_rate, day_count_fraction)
    have been consumed during resolution to produce the final payoff.
    """

    observation_time: float
    barrier: Optional[float] = None
    upper_barrier: Optional[float] = None
    lower_barrier: Optional[float] = None
    payoff: float = 0.0
    settlement_time: Optional[float] = None


@dataclass
class ObservationSchedule:
    """Ordered schedule of observation records with aggregation semantics."""

    records: List[ObservationRecord] = field(default_factory=list)
    aggregation_mode: ObservationAggregation = ObservationAggregation.STOP_FIRST_HIT
    frequency: ObservationFrequency = ObservationFrequency.CUSTOM

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
        """
        Resolve records to concrete numeric times and defaults.
        Payoff calculation is delegated to ObservationRecord.get_payoff().
        """
        self.validate(require_single=require_single, require_double=require_double)

        resolved: List[ResolvedObservationRecord] = []
        times: List[float] = []

        for rec in self.records:
            # Resolve observation time
            t = rec.resolve_time(pricing_env)
            times.append(t)

            # Get payoff using the record's get_payoff method
            payoff_value = rec.get_payoff(default_payoff, pricing_env)

            # Resolve settlement time
            settlement_t: Optional[float] = None
            if rec.settlement_date is not None and calculate_year_fraction is not None:
                try:
                    settlement_t = calculate_year_fraction(
                        pricing_env.valuation_date,
                        rec.settlement_date,
                        pricing_env.day_count_convention,
                        pricing_env.bus_days_in_year,
                    )
                except Exception:
                    # If calculation fails, default to observation time
                    settlement_t = t
            else:
                # Default: settlement at observation time
                settlement_t = t

            resolved.append(
                ResolvedObservationRecord(
                    observation_time=t,
                    barrier=rec.barrier if rec.barrier is not None else default_barrier,
                    upper_barrier=rec.upper_barrier if rec.upper_barrier is not None else default_upper,
                    lower_barrier=rec.lower_barrier if rec.lower_barrier is not None else default_lower,
                    payoff=payoff_value,
                    settlement_time=settlement_t,
                )
            )

        if times and not self._is_sorted(times):
            raise ValidationError("Observation records must be ordered by resolved observation time.")
        return resolved

    def ensure_regular_frequency(self, times: List[float], tolerance: float = 1e-8) -> float:
        """Ensure schedule has a regular observation frequency; return the interval.
        
        Args:
            times: List of observation times in years
            tolerance: Tolerance for comparing floating-point intervals (default: 1e-8)
            
        Returns:
            The regular frequency interval (float)
            
        Raises:
            ValidationError: If frequency is CUSTOM or if spacing does not match declared frequency.
        """
        if self.frequency == ObservationFrequency.CUSTOM:
            raise ValidationError("ObservationSchedule requires a regular frequency for analytical usage, but is set to CUSTOM.")
        
        if len(times) < 2:
            raise ValidationError("Need at least 2 observation times to determine interval.")
            
        # Get expected dt from the declared frequency
        expected_dt = self.frequency.to_year_fraction()
        
        # Calculate inferred dt from the first interval
        inferred_dt = times[1] - times[0]
        
        # Validate that the inferred dt matches the expected dt (check both calendar and business day conventions)
        expected_dt_calendar = self.frequency.to_year_fraction(use_business_days=False)
        expected_dt_business = self.frequency.to_year_fraction(use_business_days=True)
        
        matches_calendar = isclose(inferred_dt, expected_dt_calendar, rel_tol=tolerance, abs_tol=tolerance)
        matches_business = isclose(inferred_dt, expected_dt_business, rel_tol=tolerance, abs_tol=tolerance)
        
        if not (matches_calendar or matches_business):
            raise ValidationError(
                f"ObservationSchedule declared frequency '{self.frequency.name}' "
                f"(expected dt={expected_dt_calendar:.6f} or {expected_dt_business:.6f}) "
                f"does not match inferred interval {inferred_dt:.6f} from observation_times."
            )
            
        return inferred_dt

    def has_fixed_payoff(self, default_payoff: float = 0.0, tolerance: float = 1e-8) -> bool:
        """Check if all records share the same payoff within tolerance.
        
        Args:
            default_payoff: Default payoff value to use for records without explicit payoffs
            tolerance: Tolerance for comparing payoff values (default: 1e-8 for floating-point precision)
            
        Returns:
            True if all payoffs are equal within tolerance, False otherwise
            
        Note:
            Payoffs are computed once per record and cached for efficiency.
        """
        if not self.records:
            return True
        
        # Compute payoffs once and cache in list
        payoffs = [rec.get_payoff(default_payoff) for rec in self.records]
        base = payoffs[0]
        
        for payoff in payoffs[1:]:
            if not isclose(payoff, base, rel_tol=tolerance, abs_tol=tolerance):
                return False
        return True

    def assert_analytical_ready(self, default_payoff: float = 0.0, tolerance: float = 1e-8) -> None:
        """
        Validate preconditions for analytical engines that rely on barrier shift:
        regular observation frequency and fixed payoff across observations.
        
        Args:
            default_payoff: Default payoff value to use for records without explicit payoffs
            tolerance: Tolerance for validation checks (default: 1e-8 for numerical stability)
            
        Raises:
            ValidationError: If schedule is not suitable for analytical pricing
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
        frequency: ObservationFrequency = ObservationFrequency.CUSTOM,
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
        schedule = cls(
            records=records,
            aggregation_mode=aggregation_mode,
            frequency=frequency,
        )
        schedule.validate(require_single=upper_barrier is None and lower_barrier is None, require_double=upper_barrier is not None or lower_barrier is not None)
        return schedule

    @staticmethod
    def _is_sorted(values: List[float]) -> bool:
        """Check if list is non-decreasing."""
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))

