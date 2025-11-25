"""
Fluent API for building day paths.

This module provides an intuitive builder pattern for constructing
multi-day market evolution scenarios.
"""

from typing import Optional, List, Union
from datetime import datetime
from dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from stresstest.stress.stress_types import StressType, StressLevel
from util.exceptions import ValidationError


class PathBuilder:
    """
    Fluent builder for creating DayPath objects.
    
    Provides convenient methods for defining multi-day market scenarios
    without verbose boilerplate code.
    
    Example:
        >>> # Create a 5-day rally scenario
        >>> path = (PathBuilder(num_days=5, name="Rally")
        ...     .spot_trend(daily_change=0.02)  # +2% per day
        ...     .vol_decay(daily_change=-0.01, stress_type=StressType.ABSOLUTE)
        ...     .build())
        
        >>> # Create a custom path with different changes per day
        >>> path = (PathBuilder(num_days=3, name="Custom")
        ...     .set_day_change(0, "spot", 0.03)  # Day 0: +3%
        ...     .set_day_change(1, "spot", -0.01)  # Day 1: -1%
        ...     .set_day_change(2, "spot", 0.02)  # Day 2: +2%
        ...     .build())
    """
    
    def __init__(
        self,
        num_days: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        start_date: Optional[datetime] = None
    ):
        """
        Initialize path builder.
        
        Args:
            num_days: Number of days in the path (must be >= 1)
            name: Name for the path (default: auto-generated)
            description: Optional description
            start_date: Optional start date
        """
        if num_days < 1:
            raise ValidationError(f"num_days must be at least 1, got {num_days}")
        
        self._num_days = num_days
        self._name = name
        self._description = description
        self._start_date = start_date
        self._metadata = {}
        
        # Initialize empty steps
        self._steps: List[DayStep] = [
            DayStep(day_index=i) for i in range(num_days)
        ]
    
    def name(self, name: str) -> "PathBuilder":
        """
        Set path name.
        
        Args:
            name: Path name
            
        Returns:
            Self for chaining
        """
        self._name = name
        return self
    
    def description(self, description: str) -> "PathBuilder":
        """
        Set path description.
        
        Args:
            description: Path description
            
        Returns:
            Self for chaining
        """
        self._description = description
        return self
    
    def start_date(self, date: datetime) -> "PathBuilder":
        """
        Set start date.
        
        Args:
            date: Start date
            
        Returns:
            Self for chaining
        """
        self._start_date = date
        return self
    
    def metadata(self, key: str, value) -> "PathBuilder":
        """
        Add metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self
    
    def set_day_label(self, day_index: int, label: str) -> "PathBuilder":
        """
        Set label for a specific day.
        
        Args:
            day_index: Day index (0-based)
            label: Label text
            
        Returns:
            Self for chaining
        """
        self._validate_day_index(day_index)
        self._steps[day_index].label = label
        return self
    
    def set_day_change(
        self,
        day_index: int,
        parameter: str,
        value: float,
        stress_type: StressType = StressType.PERCENTAGE,
        underlying: Optional[str] = None
    ) -> "PathBuilder":
        """
        Set a parameter change for a specific day.
        
        Args:
            day_index: Day index (0-based)
            parameter: Parameter name (spot, volatility, rate, dividend_yield)
            value: Change value
            stress_type: Type of change (default: PERCENTAGE)
            underlying: Optional specific underlying (default: portfolio-wide)
            
        Returns:
            Self for chaining
        """
        self._validate_day_index(day_index)
        
        level, target = self._get_level_target(underlying)
        change = ParameterChange(
            parameter=parameter,
            stress_type=stress_type,
            stress_value=value,
            level=level,
            target=target,
        )
        self._steps[day_index].add_change(change)
        return self
    
    def spot_trend(
        self,
        daily_change: float,
        stress_type: StressType = StressType.PERCENTAGE,
        underlying: Optional[str] = None,
        start_day: int = 0,
        end_day: Optional[int] = None
    ) -> "PathBuilder":
        """
        Add a consistent spot trend across days.
        
        Args:
            daily_change: Daily change amount
            stress_type: Type of change (default: PERCENTAGE)
            underlying: Optional specific underlying
            start_day: First day to apply (default: 0)
            end_day: Last day to apply (default: all days)
            
        Returns:
            Self for chaining
            
        Example:
            >>> builder.spot_trend(daily_change=0.02)  # +2% per day
        """
        return self._apply_trend(
            parameter="spot",
            daily_change=daily_change,
            stress_type=stress_type,
            underlying=underlying,
            start_day=start_day,
            end_day=end_day,
        )
    
    def spot_values(
        self,
        values: List[float],
        underlying: Optional[str] = None
    ) -> "PathBuilder":
        """
        Set specific spot values for each day.
        
        Args:
            values: List of spot values (one per day)
            underlying: Optional specific underlying
            
        Returns:
            Self for chaining
        """
        return self._apply_values("spot", values, underlying)
    
    def vol_trend(
        self,
        daily_change: float,
        stress_type: StressType = StressType.PERCENTAGE,
        underlying: Optional[str] = None,
        start_day: int = 0,
        end_day: Optional[int] = None
    ) -> "PathBuilder":
        """
        Add a consistent volatility trend across days.
        
        Args:
            daily_change: Daily change amount
            stress_type: Type of change (default: PERCENTAGE)
            underlying: Optional specific underlying
            start_day: First day to apply (default: 0)
            end_day: Last day to apply (default: all days)
            
        Returns:
            Self for chaining
        """
        return self._apply_trend(
            parameter="volatility",
            daily_change=daily_change,
            stress_type=stress_type,
            underlying=underlying,
            start_day=start_day,
            end_day=end_day,
        )
    
    def vol_decay(
        self,
        daily_change: float,
        stress_type: StressType = StressType.ABSOLUTE,
        underlying: Optional[str] = None,
        start_day: int = 0,
        end_day: Optional[int] = None
    ) -> "PathBuilder":
        """
        Add volatility decay (typically negative daily change).
        
        Convenience method with ABSOLUTE as default stress type.
        
        Args:
            daily_change: Daily change amount (typically negative)
            stress_type: Type of change (default: ABSOLUTE)
            underlying: Optional specific underlying
            start_day: First day to apply
            end_day: Last day to apply
            
        Returns:
            Self for chaining
        """
        return self.vol_trend(
            daily_change=daily_change,
            stress_type=stress_type,
            underlying=underlying,
            start_day=start_day,
            end_day=end_day,
        )
    
    def vol_values(
        self,
        values: List[float],
        underlying: Optional[str] = None
    ) -> "PathBuilder":
        """
        Set specific volatility values for each day.
        
        Args:
            values: List of volatility values (one per day)
            underlying: Optional specific underlying
            
        Returns:
            Self for chaining
        """
        return self._apply_values("volatility", values, underlying)
    
    def rate_trend(
        self,
        daily_change: float,
        stress_type: StressType = StressType.ABSOLUTE,
        underlying: Optional[str] = None,
        start_day: int = 0,
        end_day: Optional[int] = None
    ) -> "PathBuilder":
        """
        Add a consistent rate trend across days.
        
        Args:
            daily_change: Daily change amount
            stress_type: Type of change (default: ABSOLUTE for rates)
            underlying: Optional specific underlying
            start_day: First day to apply
            end_day: Last day to apply
            
        Returns:
            Self for chaining
        """
        return self._apply_trend(
            parameter="rate",
            daily_change=daily_change,
            stress_type=stress_type,
            underlying=underlying,
            start_day=start_day,
            end_day=end_day,
        )
    
    def rate_values(
        self,
        values: List[float],
        underlying: Optional[str] = None
    ) -> "PathBuilder":
        """
        Set specific rate values for each day.
        
        Args:
            values: List of rate values (one per day)
            underlying: Optional specific underlying
            
        Returns:
            Self for chaining
        """
        return self._apply_values("rate", values, underlying)
    
    def div_yield_trend(
        self,
        daily_change: float,
        stress_type: StressType = StressType.ABSOLUTE,
        underlying: Optional[str] = None,
        start_day: int = 0,
        end_day: Optional[int] = None
    ) -> "PathBuilder":
        """
        Add a consistent dividend yield trend across days.
        
        Args:
            daily_change: Daily change amount
            stress_type: Type of change (default: ABSOLUTE)
            underlying: Optional specific underlying
            start_day: First day to apply
            end_day: Last day to apply
            
        Returns:
            Self for chaining
        """
        return self._apply_trend(
            parameter="dividend_yield",
            daily_change=daily_change,
            stress_type=stress_type,
            underlying=underlying,
            start_day=start_day,
            end_day=end_day,
        )
    
    def div_yield_values(
        self,
        values: List[float],
        underlying: Optional[str] = None
    ) -> "PathBuilder":
        """
        Set specific dividend yield values for each day.
        
        Args:
            values: List of dividend yield values (one per day)
            underlying: Optional specific underlying
            
        Returns:
            Self for chaining
        """
        return self._apply_values("dividend_yield", values, underlying)
    
    def add_change(
        self,
        day_index: int,
        change: ParameterChange
    ) -> "PathBuilder":
        """
        Add a pre-built ParameterChange to a specific day.
        
        Args:
            day_index: Day index
            change: ParameterChange to add
            
        Returns:
            Self for chaining
        """
        self._validate_day_index(day_index)
        self._steps[day_index].add_change(change)
        return self
    
    def _apply_trend(
        self,
        parameter: str,
        daily_change: float,
        stress_type: StressType,
        underlying: Optional[str],
        start_day: int,
        end_day: Optional[int]
    ) -> "PathBuilder":
        """Apply a trend to a parameter across days."""
        if end_day is None:
            end_day = self._num_days - 1
        
        self._validate_day_index(start_day)
        self._validate_day_index(end_day)
        
        level, target = self._get_level_target(underlying)
        
        for day in range(start_day, end_day + 1):
            change = ParameterChange(
                parameter=parameter,
                stress_type=stress_type,
                stress_value=daily_change,
                level=level,
                target=target,
            )
            self._steps[day].add_change(change)
        
        return self
    
    def _apply_values(
        self,
        parameter: str,
        values: List[float],
        underlying: Optional[str]
    ) -> "PathBuilder":
        """Apply specific values to a parameter."""
        if len(values) != self._num_days:
            raise ValidationError(
                f"Number of values ({len(values)}) must match num_days ({self._num_days})"
            )
        
        level, target = self._get_level_target(underlying)
        
        for day, value in enumerate(values):
            change = ParameterChange(
                parameter=parameter,
                stress_type=StressType.VALUE,
                stress_value=value,
                level=level,
                target=target,
            )
            self._steps[day].add_change(change)
        
        return self
    
    def _validate_day_index(self, day_index: int) -> None:
        """Validate day index is in range."""
        if not 0 <= day_index < self._num_days:
            raise ValidationError(
                f"day_index {day_index} out of range [0, {self._num_days - 1}]"
            )
    
    def _get_level_target(self, underlying: Optional[str]) -> tuple:
        """Get level and target from underlying."""
        if underlying:
            return StressLevel.UNDERLYING, underlying
        return StressLevel.PORTFOLIO, None
    
    def build(self) -> DayPath:
        """
        Build the DayPath.
        
        Returns:
            Constructed DayPath object
            
        Raises:
            ValidationError: If required fields are missing
        """
        name = self._name or f"Path_{self._num_days}days"
        
        return DayPath(
            name=name,
            steps=self._steps,
            description=self._description,
            start_date=self._start_date,
            metadata=self._metadata,
        )
    
    def reset(self) -> "PathBuilder":
        """
        Reset builder to initial state (empty steps).
        
        Returns:
            Self for chaining
        """
        self._steps = [DayStep(day_index=i) for i in range(self._num_days)]
        return self
    
    def __repr__(self) -> str:
        return f"PathBuilder(days={self._num_days}, name={self._name})"

