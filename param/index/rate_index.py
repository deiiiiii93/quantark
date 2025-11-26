"""
Rate index definitions for floating rate instruments.

This module provides:
- RateIndex base class for defining reference rate indices
- IndexFixing for historical rate fixings
- IndexFixingStore for managing historical fixings
- Predefined indices for common markets (US, Europe, China)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from util.calendar import DayCountConvention, CalendarType
from util.enum import ResetConvention
from util.exceptions import ValidationError


@dataclass
class IndexFixing:
    """
    Represents a single rate fixing for an index.
    
    Attributes:
        fixing_date: Date when the rate was fixed
        rate: The fixed rate value (e.g., 0.05 for 5%)
        index_name: Name of the index
    """
    fixing_date: datetime
    rate: float
    index_name: str
    
    def __post_init__(self):
        """Validate fixing data."""
        if self.rate < -0.10:  # Allow negative rates but sanity check
            raise ValidationError(f"Rate {self.rate} seems unreasonably low")
        if self.rate > 0.50:
            raise ValidationError(f"Rate {self.rate} seems unreasonably high")
    
    def __repr__(self):
        return f"IndexFixing({self.index_name}, {self.fixing_date.date()}, {self.rate:.4%})"


class IndexFixingStore:
    """
    Store for managing historical rate fixings.
    
    Provides efficient lookup of historical fixings by date and supports
    multiple indices.
    """
    
    def __init__(self):
        """Initialize empty fixing store."""
        # Store fixings by index name, then by date
        self._fixings: Dict[str, Dict[datetime, float]] = {}
    
    def add_fixing(self, fixing: IndexFixing) -> None:
        """
        Add a single fixing to the store.
        
        Args:
            fixing: IndexFixing to add
        """
        if fixing.index_name not in self._fixings:
            self._fixings[fixing.index_name] = {}
        
        # Normalize date to midnight
        date_key = datetime(
            fixing.fixing_date.year,
            fixing.fixing_date.month,
            fixing.fixing_date.day
        )
        self._fixings[fixing.index_name][date_key] = fixing.rate
    
    def add_fixings(self, fixings: List[IndexFixing]) -> None:
        """
        Add multiple fixings to the store.
        
        Args:
            fixings: List of IndexFixing objects
        """
        for fixing in fixings:
            self.add_fixing(fixing)
    
    def add_fixings_from_dict(
        self,
        index_name: str,
        fixings: Dict[datetime, float]
    ) -> None:
        """
        Add fixings from a dictionary.
        
        Args:
            index_name: Name of the index
            fixings: Dictionary mapping dates to rates
        """
        for date, rate in fixings.items():
            self.add_fixing(IndexFixing(
                fixing_date=date,
                rate=rate,
                index_name=index_name
            ))
    
    def get_fixing(
        self,
        index_name: str,
        fixing_date: datetime
    ) -> Optional[float]:
        """
        Get a fixing for a specific date.
        
        Args:
            index_name: Name of the index
            fixing_date: Date to look up
            
        Returns:
            Rate if found, None otherwise
        """
        if index_name not in self._fixings:
            return None
        
        # Normalize date
        date_key = datetime(
            fixing_date.year,
            fixing_date.month,
            fixing_date.day
        )
        
        return self._fixings[index_name].get(date_key)
    
    def get_latest_fixing(
        self,
        index_name: str,
        before_date: datetime
    ) -> Optional[Tuple[datetime, float]]:
        """
        Get the most recent fixing before a given date.
        
        Args:
            index_name: Name of the index
            before_date: Get latest fixing before this date
            
        Returns:
            Tuple of (date, rate) if found, None otherwise
        """
        if index_name not in self._fixings:
            return None
        
        # Normalize date
        date_key = datetime(
            before_date.year,
            before_date.month,
            before_date.day
        )
        
        # Find latest date before the given date
        available_dates = [d for d in self._fixings[index_name].keys() if d <= date_key]
        
        if not available_dates:
            return None
        
        latest_date = max(available_dates)
        return (latest_date, self._fixings[index_name][latest_date])
    
    def get_all_fixings(
        self,
        index_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get all fixings for an index within a date range.
        
        Args:
            index_name: Name of the index
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of (date, rate) tuples, sorted by date
        """
        if index_name not in self._fixings:
            return []
        
        fixings = []
        for date, rate in self._fixings[index_name].items():
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            fixings.append((date, rate))
        
        return sorted(fixings, key=lambda x: x[0])
    
    def has_fixing(self, index_name: str, fixing_date: datetime) -> bool:
        """
        Check if a fixing exists for a given date.
        
        Args:
            index_name: Name of the index
            fixing_date: Date to check
            
        Returns:
            True if fixing exists
        """
        return self.get_fixing(index_name, fixing_date) is not None
    
    def __repr__(self):
        counts = {name: len(fixings) for name, fixings in self._fixings.items()}
        return f"IndexFixingStore({counts})"


@dataclass
class RateIndex:
    """
    Base class for reference rate indices.
    
    Defines the characteristics of a floating rate index such as SOFR,
    EURIBOR, SHIBOR, etc.
    
    Attributes:
        name: Index name (e.g., "SOFR", "EURIBOR_3M")
        tenor_months: Tenor in months (0 for overnight rates)
        day_count_convention: Day count convention for rate calculation
        fixing_lag_days: Number of business days before rate is fixed
        calendar_type: Calendar for business day adjustments
        currency: Currency code (e.g., "USD", "EUR", "CNY")
        description: Human-readable description
    """
    name: str
    tenor_months: int
    day_count_convention: DayCountConvention
    fixing_lag_days: int
    calendar_type: CalendarType
    currency: str
    description: str = ""
    
    # Fixing store for historical rates
    _fixing_store: Optional[IndexFixingStore] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate index parameters."""
        if self.tenor_months < 0:
            raise ValidationError(f"Tenor must be non-negative, got {self.tenor_months}")
        if self.fixing_lag_days < 0:
            raise ValidationError(f"Fixing lag must be non-negative, got {self.fixing_lag_days}")
    
    @property
    def is_overnight(self) -> bool:
        """Check if this is an overnight rate."""
        return self.tenor_months == 0
    
    @property
    def tenor_years(self) -> float:
        """Get tenor in years."""
        return self.tenor_months / 12.0
    
    def set_fixing_store(self, store: IndexFixingStore) -> None:
        """
        Set the fixing store for this index.
        
        Args:
            store: IndexFixingStore instance
        """
        self._fixing_store = store
    
    def get_fixing(self, fixing_date: datetime) -> Optional[float]:
        """
        Get historical fixing for a date.
        
        Args:
            fixing_date: Date to look up
            
        Returns:
            Rate if found, None otherwise
        """
        if self._fixing_store is None:
            return None
        return self._fixing_store.get_fixing(self.name, fixing_date)
    
    def get_latest_fixing(self, before_date: datetime) -> Optional[Tuple[datetime, float]]:
        """
        Get the most recent fixing before a date.
        
        Args:
            before_date: Get fixing before this date
            
        Returns:
            Tuple of (date, rate) if found
        """
        if self._fixing_store is None:
            return None
        return self._fixing_store.get_latest_fixing(self.name, before_date)
    
    def add_fixing(self, fixing_date: datetime, rate: float) -> None:
        """
        Add a fixing to the store.
        
        Args:
            fixing_date: Date of fixing
            rate: Fixed rate value
        """
        if self._fixing_store is None:
            self._fixing_store = IndexFixingStore()
        
        self._fixing_store.add_fixing(IndexFixing(
            fixing_date=fixing_date,
            rate=rate,
            index_name=self.name
        ))
    
    def calculate_fixing_date(
        self,
        accrual_start: datetime,
        reset_convention: ResetConvention
    ) -> datetime:
        """
        Calculate the fixing date for an accrual period.
        
        Args:
            accrual_start: Start of accrual period
            reset_convention: In-advance or in-arrears
            
        Returns:
            The fixing date
        """
        if reset_convention == ResetConvention.IN_ADVANCE:
            # Fix at start of period, minus fixing lag
            return accrual_start - timedelta(days=self.fixing_lag_days)
        else:
            # In-arrears: typically fixed near end of period
            # The actual date depends on the specific index and conventions
            return accrual_start - timedelta(days=self.fixing_lag_days)
    
    def __repr__(self):
        tenor_str = "ON" if self.is_overnight else f"{self.tenor_months}M"
        return f"RateIndex({self.name}, {tenor_str}, {self.currency})"
    
    def __eq__(self, other):
        if not isinstance(other, RateIndex):
            return False
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)


# =============================================================================
# Predefined Indices - US Market
# =============================================================================

SOFR = RateIndex(
    name="SOFR",
    tenor_months=0,  # Overnight
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=1,
    calendar_type=CalendarType.US,
    currency="USD",
    description="Secured Overnight Financing Rate"
)

SOFR_1M = RateIndex(
    name="SOFR_1M",
    tenor_months=1,
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=2,
    calendar_type=CalendarType.US,
    currency="USD",
    description="1-Month Term SOFR"
)

SOFR_3M = RateIndex(
    name="SOFR_3M",
    tenor_months=3,
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=2,
    calendar_type=CalendarType.US,
    currency="USD",
    description="3-Month Term SOFR"
)

PRIME = RateIndex(
    name="PRIME",
    tenor_months=0,  # Reference rate, not term-based
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=0,
    calendar_type=CalendarType.US,
    currency="USD",
    description="US Prime Rate"
)


# =============================================================================
# Predefined Indices - Europe
# =============================================================================

EURIBOR_3M = RateIndex(
    name="EURIBOR_3M",
    tenor_months=3,
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=2,
    calendar_type=CalendarType.TARGET,
    currency="EUR",
    description="3-Month EURIBOR"
)

EURIBOR_6M = RateIndex(
    name="EURIBOR_6M",
    tenor_months=6,
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=2,
    calendar_type=CalendarType.TARGET,
    currency="EUR",
    description="6-Month EURIBOR"
)

ESTR = RateIndex(
    name="ESTR",
    tenor_months=0,  # Overnight
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=1,
    calendar_type=CalendarType.TARGET,
    currency="EUR",
    description="Euro Short-Term Rate"
)


# =============================================================================
# Predefined Indices - China
# =============================================================================

SHIBOR_3M = RateIndex(
    name="SHIBOR_3M",
    tenor_months=3,
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=0,
    calendar_type=CalendarType.CHINA,
    currency="CNY",
    description="3-Month Shanghai Interbank Offered Rate"
)

REPO_7D = RateIndex(
    name="REPO_7D",
    tenor_months=0,  # 7-day repo, treated as short-term
    day_count_convention=DayCountConvention.ACT_365,
    fixing_lag_days=0,
    calendar_type=CalendarType.CHINA,
    currency="CNY",
    description="7-Day Repo Rate (DR007) - Fixed Weekly"
)


# =============================================================================
# Predefined Indices - Legacy (LIBOR)
# =============================================================================

LIBOR_3M = RateIndex(
    name="LIBOR_3M",
    tenor_months=3,
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=2,
    calendar_type=CalendarType.UK,
    currency="USD",
    description="3-Month USD LIBOR (Legacy)"
)

LIBOR_6M = RateIndex(
    name="LIBOR_6M",
    tenor_months=6,
    day_count_convention=DayCountConvention.ACT_360,
    fixing_lag_days=2,
    calendar_type=CalendarType.UK,
    currency="USD",
    description="6-Month USD LIBOR (Legacy)"
)


# =============================================================================
# Index Registry and Factory
# =============================================================================

_INDEX_REGISTRY: Dict[str, RateIndex] = {
    "SOFR": SOFR,
    "SOFR_1M": SOFR_1M,
    "SOFR_3M": SOFR_3M,
    "PRIME": PRIME,
    "EURIBOR_3M": EURIBOR_3M,
    "EURIBOR_6M": EURIBOR_6M,
    "ESTR": ESTR,
    "SHIBOR_3M": SHIBOR_3M,
    "REPO_7D": REPO_7D,
    "DR007": REPO_7D,  # Alias
    "LIBOR_3M": LIBOR_3M,
    "LIBOR_6M": LIBOR_6M,
}


def create_index(
    name: str,
    tenor_months: Optional[int] = None,
    day_count_convention: Optional[DayCountConvention] = None,
    fixing_lag_days: Optional[int] = None,
    calendar_type: Optional[CalendarType] = None,
    currency: Optional[str] = None,
    description: Optional[str] = None
) -> RateIndex:
    """
    Create a rate index, either from predefined templates or custom.
    
    If only name is provided and it matches a predefined index, returns
    that index. Otherwise, creates a custom index with the provided parameters.
    
    Args:
        name: Index name (e.g., "SOFR", "EURIBOR_3M", or custom name)
        tenor_months: Tenor in months (required for custom indices)
        day_count_convention: Day count convention
        fixing_lag_days: Business days before fixing
        calendar_type: Calendar for adjustments
        currency: Currency code
        description: Description text
        
    Returns:
        RateIndex instance
        
    Raises:
        ValidationError: If required parameters are missing for custom index
    """
    # Check if it's a predefined index
    if name.upper() in _INDEX_REGISTRY:
        base_index = _INDEX_REGISTRY[name.upper()]
        
        # If no overrides, return the predefined index
        if all(p is None for p in [tenor_months, day_count_convention, 
                                    fixing_lag_days, calendar_type, 
                                    currency, description]):
            return base_index
        
        # Create modified version
        return RateIndex(
            name=name,
            tenor_months=tenor_months if tenor_months is not None else base_index.tenor_months,
            day_count_convention=day_count_convention if day_count_convention is not None else base_index.day_count_convention,
            fixing_lag_days=fixing_lag_days if fixing_lag_days is not None else base_index.fixing_lag_days,
            calendar_type=calendar_type if calendar_type is not None else base_index.calendar_type,
            currency=currency if currency is not None else base_index.currency,
            description=description if description is not None else base_index.description
        )
    
    # Custom index - require all parameters
    if tenor_months is None:
        raise ValidationError(f"tenor_months required for custom index '{name}'")
    if day_count_convention is None:
        raise ValidationError(f"day_count_convention required for custom index '{name}'")
    if calendar_type is None:
        raise ValidationError(f"calendar_type required for custom index '{name}'")
    if currency is None:
        raise ValidationError(f"currency required for custom index '{name}'")
    
    return RateIndex(
        name=name,
        tenor_months=tenor_months,
        day_count_convention=day_count_convention,
        fixing_lag_days=fixing_lag_days if fixing_lag_days is not None else 0,
        calendar_type=calendar_type,
        currency=currency,
        description=description or ""
    )

