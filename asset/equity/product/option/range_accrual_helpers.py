"""
Factory functions for creating common Range Accrual option structures.

This module provides helper functions that simplify the creation of Range Accrual
options by providing sensible defaults for common market structures.

Available helpers:
- create_standard_range_accrual(): Basic range accrual with flat barriers
- create_reverse_range_accrual(): Pays when spot is OUTSIDE range
- create_stepdown_range_accrual(): Barriers decrease over time (easier to hit)
- generate_range_observation_records(): Create observation schedule with weights
- assign_calendar_day_weights(): Assign Friday=3 weights for calendar day convention

Example:
    >>> from asset.equity.product.option import create_standard_range_accrual
    >>> option = create_standard_range_accrual(
    ...     initial_price=100.0,
    ...     upper_barrier=110.0,
    ...     lower_barrier=90.0,
    ...     maturity=1.0,
    ...     accrual_rate=0.05,
    ... )
"""

from datetime import datetime, timedelta
from typing import List, Optional

from util.calendar.day_counter import DayCountConvention
from util.enum import CouponPayType, ObservationFrequency
from util.exceptions import ValidationError

from .range_accrual_config import RangeAccrualConfig, RangeAccrualObservationRecord
from .range_accrual_option import RangeAccrualOption


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _validate_core_params(
    initial_price: float,
    upper_barrier: float,
    lower_barrier: float,
    maturity: float,
    func_name: str,
) -> None:
    """Validate core parameters common to all helpers."""
    if initial_price <= 0:
        raise ValidationError(
            f"{func_name}: initial_price must be positive, got {initial_price}"
        )
    if upper_barrier <= 0:
        raise ValidationError(
            f"{func_name}: upper_barrier must be positive, got {upper_barrier}"
        )
    if lower_barrier <= 0:
        raise ValidationError(
            f"{func_name}: lower_barrier must be positive, got {lower_barrier}"
        )
    if lower_barrier >= upper_barrier:
        raise ValidationError(
            f"{func_name}: lower_barrier ({lower_barrier}) must be less than "
            f"upper_barrier ({upper_barrier})"
        )
    if maturity <= 0:
        raise ValidationError(
            f"{func_name}: maturity must be positive, got {maturity}"
        )


# =============================================================================
# Calendar Day Weight Functions
# =============================================================================


def assign_calendar_day_weights(
    observation_dates: List[datetime],
) -> List[float]:
    """
    Assign weights based on calendar day convention.

    Friday observations get weight=3.0 (covers Saturday and Sunday).
    Other weekdays get weight=1.0.

    Args:
        observation_dates: List of observation dates (business days only)

    Returns:
        List of weights where Friday=3, other weekdays=1

    Example:
        >>> from datetime import datetime
        >>> dates = [
        ...     datetime(2025, 1, 6),   # Monday
        ...     datetime(2025, 1, 7),   # Tuesday
        ...     datetime(2025, 1, 10),  # Friday
        ... ]
        >>> assign_calendar_day_weights(dates)
        [1.0, 1.0, 3.0]
    """
    weights = []
    for date in observation_dates:
        # weekday(): 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        if date.weekday() == 4:  # Friday
            weights.append(3.0)
        else:
            weights.append(1.0)
    return weights


def generate_range_observation_records(
    start_date: datetime,
    end_date: datetime,
    frequency: ObservationFrequency = ObservationFrequency.DAILY,
    use_calendar_day_weights: bool = False,
    upper_barrier: Optional[float] = None,
    lower_barrier: Optional[float] = None,
) -> List[RangeAccrualObservationRecord]:
    """
    Generate observation records with optional calendar day weighting.

    If use_calendar_day_weights=True:
    - Friday observations get weight=3.0 (covers Sat+Sun)
    - Other weekdays get weight=1.0
    - Weekends are skipped (business days only)

    This matches market convention for calendar day accrual.

    Args:
        start_date: First observation date (inclusive)
        end_date: Last observation date (inclusive)
        frequency: Observation frequency (DAILY, WEEKLY, MONTHLY, etc.)
        use_calendar_day_weights: If True, auto-assign Friday=3 weights
        upper_barrier: Optional per-observation upper barrier
        lower_barrier: Optional per-observation lower barrier

    Returns:
        List of RangeAccrualObservationRecord

    Example:
        >>> from datetime import datetime
        >>> records = generate_range_observation_records(
        ...     start_date=datetime(2025, 1, 6),
        ...     end_date=datetime(2025, 1, 10),
        ...     use_calendar_day_weights=True,
        ... )
        >>> [r.weight for r in records]
        [1.0, 1.0, 1.0, 1.0, 3.0]  # Mon, Tue, Wed, Thu, Fri
    """
    if start_date > end_date:
        raise ValidationError(
            f"start_date ({start_date}) must be before or equal to end_date ({end_date})"
        )

    dates = _generate_observation_dates(start_date, end_date, frequency)

    if use_calendar_day_weights:
        weights = assign_calendar_day_weights(dates)
    else:
        weights = [1.0] * len(dates)

    records = []
    for date, weight in zip(dates, weights):
        records.append(
            RangeAccrualObservationRecord(
                observation_date=date,
                weight=weight,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
            )
        )

    return records


def _generate_observation_dates(
    start_date: datetime,
    end_date: datetime,
    frequency: ObservationFrequency,
) -> List[datetime]:
    """Generate observation dates based on frequency, skipping weekends."""
    dates = []
    current = start_date

    if frequency == ObservationFrequency.DAILY:
        while current <= end_date:
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)

    elif frequency == ObservationFrequency.WEEKLY:
        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(weeks=1)

    elif frequency == ObservationFrequency.MONTHLY:
        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                dates.append(current)
            # Move to same day next month
            month = current.month + 1
            year = current.year
            if month > 12:
                month = 1
                year += 1
            # Handle end-of-month edge cases
            day = min(current.day, _days_in_month(year, month))
            current = datetime(year, month, day)

    elif frequency == ObservationFrequency.QUARTERLY:
        while current <= end_date:
            if current.weekday() < 5:
                dates.append(current)
            # Move 3 months forward
            month = current.month + 3
            year = current.year
            while month > 12:
                month -= 12
                year += 1
            day = min(current.day, _days_in_month(year, month))
            current = datetime(year, month, day)

    elif frequency == ObservationFrequency.SEMI_ANNUALLY:
        while current <= end_date:
            if current.weekday() < 5:
                dates.append(current)
            # Move 6 months forward
            month = current.month + 6
            year = current.year
            while month > 12:
                month -= 12
                year += 1
            day = min(current.day, _days_in_month(year, month))
            current = datetime(year, month, day)

    elif frequency == ObservationFrequency.ANNUALLY:
        while current <= end_date:
            if current.weekday() < 5:
                dates.append(current)
            current = datetime(current.year + 1, current.month, current.day)

    else:
        raise ValidationError(
            f"Unsupported observation frequency: {frequency}. "
            "Use CUSTOM frequency with explicit observation_records instead."
        )

    return dates


def _days_in_month(year: int, month: int) -> int:
    """Get number of days in a month."""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return 29
        return 28
    raise ValidationError(f"Invalid month: {month}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_standard_range_accrual(
    initial_price: float,
    upper_barrier: float,
    lower_barrier: float,
    maturity: float,
    contract_multiplier: float = 1.0,
    accrual_rate: float = 0.01,
    num_observations: int = 252,
    is_rate_annualized: bool = True,
    use_calendar_day_weights: bool = False,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    accrual_pay_type: CouponPayType = CouponPayType.EXPIRY,
    observation_records: Optional[List[RangeAccrualObservationRecord]] = None,
) -> RangeAccrualOption:
    """
    Create a standard range accrual option with flat barriers.

    The most common range accrual structure with:
    - Flat (constant) upper and lower barriers
    - Daily observations by default (252 per year)
    - Pay at expiry
    - Optional calendar day weighting (Friday=3)

    Args:
        initial_price: Reference price for payoff calculations
        upper_barrier: Upper barrier level
        lower_barrier: Lower barrier level
        maturity: Time to maturity in years
        contract_multiplier: Underlying units represented by one contract
        accrual_rate: Per-period or annualized accrual rate (default: 1%)
        num_observations: Number of observations (default: 252 for daily)
        is_rate_annualized: Whether accrual_rate is annualized (default: True)
        use_calendar_day_weights: If True, auto-assign Friday=3 weights
        day_count_convention: Day count for year fraction (default: ACT/365)
        accrual_pay_type: INSTANT or EXPIRY (default: EXPIRY)
        observation_records: Custom observation records (overrides num_observations)

    Returns:
        Configured RangeAccrualOption instance

    Example:
        >>> option = create_standard_range_accrual(
        ...     initial_price=100.0,
        ...     upper_barrier=110.0,
        ...     lower_barrier=90.0,
        ...     maturity=1.0,
        ...     accrual_rate=0.05,
        ... )
    """
    _validate_core_params(
        initial_price,
        upper_barrier,
        lower_barrier,
        maturity,
        "create_standard_range_accrual",
    )

    range_config = RangeAccrualConfig(
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        accrual_rate=accrual_rate,
        is_rate_annualized=is_rate_annualized,
        day_count_convention=day_count_convention,
        accrual_pay_type=accrual_pay_type,
        is_reverse=False,
        use_calendar_day_weights=use_calendar_day_weights,
    )

    # Build observation records if using calendar day weights
    obs_records = observation_records
    if obs_records is None and use_calendar_day_weights:
        # Generate uniform schedule with Friday=3 weights
        obs_times = [
            (i + 1) / num_observations * maturity for i in range(num_observations)
        ]
        # Estimate weekday pattern based on observation index
        # Simplified: assume starting Monday, every 5th observation is Friday
        weights = []
        for i in range(num_observations):
            day_of_week = i % 5  # 0=Mon, 1=Tue, ..., 4=Fri
            if day_of_week == 4:  # Friday
                weights.append(3.0)
            else:
                weights.append(1.0)
        obs_records = [
            RangeAccrualObservationRecord(observation_time=t, weight=w)
            for t, w in zip(obs_times, weights)
        ]

    return RangeAccrualOption(
        initial_price=initial_price,
        range_config=range_config,
        maturity=maturity,
        observation_records=obs_records,
        num_observations=num_observations if obs_records is None else None,
        contract_multiplier=contract_multiplier,
    )


def create_reverse_range_accrual(
    initial_price: float,
    upper_barrier: float,
    lower_barrier: float,
    maturity: float,
    contract_multiplier: float = 1.0,
    accrual_rate: float = 0.01,
    num_observations: int = 252,
    is_rate_annualized: bool = True,
    use_calendar_day_weights: bool = False,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    accrual_pay_type: CouponPayType = CouponPayType.EXPIRY,
    observation_records: Optional[List[RangeAccrualObservationRecord]] = None,
) -> RangeAccrualOption:
    """
    Create a reverse range accrual option that pays when OUTSIDE the range.

    Same as standard range accrual, but accrual occurs when spot is
    BELOW lower_barrier OR ABOVE upper_barrier.

    Args:
        initial_price: Reference price for payoff calculations
        upper_barrier: Upper barrier level
        lower_barrier: Lower barrier level
        maturity: Time to maturity in years
        contract_multiplier: Underlying units represented by one contract
        accrual_rate: Per-period or annualized accrual rate (default: 1%)
        num_observations: Number of observations (default: 252 for daily)
        is_rate_annualized: Whether accrual_rate is annualized (default: True)
        use_calendar_day_weights: If True, auto-assign Friday=3 weights
        day_count_convention: Day count for year fraction (default: ACT/365)
        accrual_pay_type: INSTANT or EXPIRY (default: EXPIRY)
        observation_records: Custom observation records (overrides num_observations)

    Returns:
        Configured RangeAccrualOption instance with is_reverse=True

    Example:
        >>> option = create_reverse_range_accrual(
        ...     initial_price=100.0,
        ...     upper_barrier=110.0,
        ...     lower_barrier=90.0,
        ...     maturity=1.0,
        ...     accrual_rate=0.05,
        ... )
        >>> option.range_config.is_reverse
        True
    """
    _validate_core_params(
        initial_price,
        upper_barrier,
        lower_barrier,
        maturity,
        "create_reverse_range_accrual",
    )

    range_config = RangeAccrualConfig(
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        accrual_rate=accrual_rate,
        is_rate_annualized=is_rate_annualized,
        day_count_convention=day_count_convention,
        accrual_pay_type=accrual_pay_type,
        is_reverse=True,  # Key difference: pay when OUTSIDE range
        use_calendar_day_weights=use_calendar_day_weights,
    )

    # Build observation records if using calendar day weights
    obs_records = observation_records
    if obs_records is None and use_calendar_day_weights:
        obs_times = [
            (i + 1) / num_observations * maturity for i in range(num_observations)
        ]
        weights = []
        for i in range(num_observations):
            day_of_week = i % 5
            if day_of_week == 4:
                weights.append(3.0)
            else:
                weights.append(1.0)
        obs_records = [
            RangeAccrualObservationRecord(observation_time=t, weight=w)
            for t, w in zip(obs_times, weights)
        ]

    return RangeAccrualOption(
        initial_price=initial_price,
        range_config=range_config,
        maturity=maturity,
        observation_records=obs_records,
        num_observations=num_observations if obs_records is None else None,
        contract_multiplier=contract_multiplier,
    )


def create_stepdown_range_accrual(
    initial_price: float,
    initial_upper_barrier: float,
    initial_lower_barrier: float,
    maturity: float,
    upper_stepdown_rate: float = 0.005,
    lower_stepdown_rate: float = 0.005,
    contract_multiplier: float = 1.0,
    accrual_rate: float = 0.01,
    num_observations: int = 12,
    is_rate_annualized: bool = True,
    use_calendar_day_weights: bool = False,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    accrual_pay_type: CouponPayType = CouponPayType.EXPIRY,
    is_reverse: bool = False,
) -> RangeAccrualOption:
    """
    Create a step-down range accrual where barriers narrow over time.

    Both upper and lower barriers move toward initial_price at each observation,
    making it progressively harder to stay in-range. This structure provides
    higher accrual rates to compensate for the narrowing corridor.

    Upper barrier: decreases by upper_stepdown_rate * initial_price each period
    Lower barrier: increases by lower_stepdown_rate * initial_price each period

    Args:
        initial_price: Reference price for payoff calculations
        initial_upper_barrier: Starting upper barrier
        initial_lower_barrier: Starting lower barrier
        maturity: Time to maturity in years
        upper_stepdown_rate: Upper barrier step-down as % of initial_price per period
        lower_stepdown_rate: Lower barrier step-up as % of initial_price per period
        contract_multiplier: Underlying units represented by one contract
        accrual_rate: Per-period or annualized accrual rate (default: 1%)
        num_observations: Number of observations (default: 12 for monthly)
        is_rate_annualized: Whether accrual_rate is annualized (default: True)
        use_calendar_day_weights: If True, auto-assign Friday=3 weights
        day_count_convention: Day count for year fraction (default: ACT/365)
        accrual_pay_type: INSTANT or EXPIRY (default: EXPIRY)
        is_reverse: If True, pay when OUTSIDE range (default: False)

    Returns:
        Configured RangeAccrualOption with time-varying barriers

    Example:
        >>> option = create_stepdown_range_accrual(
        ...     initial_price=100.0,
        ...     initial_upper_barrier=115.0,
        ...     initial_lower_barrier=85.0,
        ...     maturity=1.0,
        ...     upper_stepdown_rate=0.01,  # -1% per period
        ...     lower_stepdown_rate=0.01,  # +1% per period
        ...     num_observations=12,
        ... )
        >>> option.range_config.upper_barrier[0]  # First: 115
        115.0
        >>> option.range_config.upper_barrier[-1]  # Last: ~103
        104.0
    """
    _validate_core_params(
        initial_price,
        initial_upper_barrier,
        initial_lower_barrier,
        maturity,
        "create_stepdown_range_accrual",
    )

    if upper_stepdown_rate < 0:
        raise ValidationError(
            f"upper_stepdown_rate must be non-negative, got {upper_stepdown_rate}"
        )
    if lower_stepdown_rate < 0:
        raise ValidationError(
            f"lower_stepdown_rate must be non-negative, got {lower_stepdown_rate}"
        )

    # Generate step-down barriers
    upper_barriers = []
    lower_barriers = []
    upper_step = upper_stepdown_rate * initial_price
    lower_step = lower_stepdown_rate * initial_price

    for i in range(num_observations):
        upper = initial_upper_barrier - i * upper_step
        lower = initial_lower_barrier + i * lower_step

        # Ensure barriers don't cross
        if lower >= upper:
            raise ValidationError(
                f"Barriers cross at observation {i}: lower ({lower}) >= upper ({upper}). "
                "Reduce stepdown rates or use fewer observations."
            )

        upper_barriers.append(upper)
        lower_barriers.append(lower)

    range_config = RangeAccrualConfig(
        upper_barrier=upper_barriers,
        lower_barrier=lower_barriers,
        accrual_rate=accrual_rate,
        is_rate_annualized=is_rate_annualized,
        day_count_convention=day_count_convention,
        accrual_pay_type=accrual_pay_type,
        is_reverse=is_reverse,
        use_calendar_day_weights=use_calendar_day_weights,
    )

    # Build observation records
    obs_times = [
        (i + 1) / num_observations * maturity for i in range(num_observations)
    ]

    if use_calendar_day_weights:
        weights = []
        for i in range(num_observations):
            day_of_week = i % 5
            if day_of_week == 4:
                weights.append(3.0)
            else:
                weights.append(1.0)
    else:
        weights = [1.0] * num_observations

    obs_records = [
        RangeAccrualObservationRecord(
            observation_time=t,
            weight=w,
            upper_barrier=upper_barriers[i],
            lower_barrier=lower_barriers[i],
        )
        for i, (t, w) in enumerate(zip(obs_times, weights))
    ]

    return RangeAccrualOption(
        initial_price=initial_price,
        range_config=range_config,
        maturity=maturity,
        observation_records=obs_records,
        contract_multiplier=contract_multiplier,
    )
