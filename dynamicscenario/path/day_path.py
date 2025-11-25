"""
Day path definitions for dynamic scenario analysis.

This module defines the core data structures for representing multi-day
market evolution scenarios.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime, timedelta
from stresstest.stress.stress_types import StressType, StressLevel
from util.exceptions import ValidationError


@dataclass
class ParameterChange:
    """
    Represents a change to a single market parameter.
    
    This is analogous to the Stress class in static stress testing,
    but designed for day-by-day evolution.
    
    Attributes:
        parameter: Name of parameter to change (spot, volatility, rate, dividend_yield)
        stress_type: How to apply the change (PERCENTAGE, ABSOLUTE, VALUE)
        stress_value: Magnitude of change
        level: Level to apply (PORTFOLIO, UNDERLYING, POSITION)
        target: Target identifier if level is UNDERLYING or POSITION
        
    Examples:
        >>> # Spot up 2% for entire portfolio
        >>> ParameterChange("spot", StressType.PERCENTAGE, 0.02)
        
        >>> # Volatility set to 0.30 for AAPL
        >>> ParameterChange("volatility", StressType.VALUE, 0.30, 
        ...                 level=StressLevel.UNDERLYING, target="AAPL")
    """
    parameter: str
    stress_type: StressType
    stress_value: float
    level: StressLevel = StressLevel.PORTFOLIO
    target: Optional[str] = None
    
    def __post_init__(self):
        """Validate parameter change."""
        if not self.parameter:
            raise ValidationError("Parameter name is required")
        
        # Normalize parameter name
        self.parameter = self.parameter.lower()
        
        valid_params = ['spot', 'volatility', 'vol', 'rate', 'dividend_yield', 'div_yield', 'dividend']
        if self.parameter not in valid_params:
            raise ValidationError(
                f"Unknown parameter '{self.parameter}'. "
                f"Valid parameters: {valid_params}"
            )
        
        if self.level == StressLevel.UNDERLYING and not self.target:
            raise ValidationError("Target underlying required for UNDERLYING level")
        
        if self.level == StressLevel.POSITION and not self.target:
            raise ValidationError("Target position_id required for POSITION level")
        
        if self.level == StressLevel.PORTFOLIO and self.target:
            raise ValidationError("Target should not be specified for PORTFOLIO level")
    
    def apply(self, current_value: float) -> float:
        """
        Apply this change to a current value.
        
        Args:
            current_value: Current value of the parameter
            
        Returns:
            New value after applying change
        """
        return self.stress_type.apply(current_value, self.stress_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'parameter': self.parameter,
            'stress_type': self.stress_type.value,
            'stress_value': self.stress_value,
            'level': self.level.value,
            'target': self.target,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterChange":
        """Create from dictionary."""
        return cls(
            parameter=data['parameter'],
            stress_type=StressType(data['stress_type']),
            stress_value=data['stress_value'],
            level=StressLevel(data.get('level', 'portfolio')),
            target=data.get('target'),
        )
    
    def __repr__(self) -> str:
        target_str = f" on {self.target}" if self.target else ""
        if self.stress_type == StressType.PERCENTAGE:
            return f"ParameterChange({self.parameter} {self.stress_value:+.2%}{target_str})"
        elif self.stress_type == StressType.ABSOLUTE:
            return f"ParameterChange({self.parameter} {self.stress_value:+.4f} abs{target_str})"
        else:
            return f"ParameterChange({self.parameter} -> {self.stress_value:.4f}{target_str})"


@dataclass
class DayStep:
    """
    Represents market state changes for a single day.
    
    A DayStep contains all parameter changes to apply on a given day.
    Multiple parameters can change simultaneously.
    
    Attributes:
        day_index: 0-based index of this day in the path
        changes: List of parameter changes to apply
        label: Optional human-readable label for this day
        metadata: Optional additional data
        
    Example:
        >>> step = DayStep(
        ...     day_index=0,
        ...     changes=[
        ...         ParameterChange("spot", StressType.PERCENTAGE, 0.02),
        ...         ParameterChange("volatility", StressType.ABSOLUTE, 0.01),
        ...     ],
        ...     label="Day 1: Rally begins"
        ... )
    """
    day_index: int
    changes: List[ParameterChange] = field(default_factory=list)
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate day step."""
        if self.day_index < 0:
            raise ValidationError(f"day_index must be non-negative, got {self.day_index}")
    
    def add_change(self, change: ParameterChange) -> "DayStep":
        """
        Add a parameter change to this day.
        
        Args:
            change: Parameter change to add
            
        Returns:
            Self for chaining
        """
        self.changes.append(change)
        return self
    
    def get_changes_for_parameter(self, parameter: str) -> List[ParameterChange]:
        """Get all changes for a specific parameter."""
        param_lower = parameter.lower()
        return [c for c in self.changes if c.parameter == param_lower]
    
    def get_changes_for_underlying(self, underlying: str) -> List[ParameterChange]:
        """Get all changes applicable to a specific underlying."""
        return [
            c for c in self.changes
            if c.level == StressLevel.PORTFOLIO or 
               (c.level == StressLevel.UNDERLYING and c.target == underlying)
        ]
    
    def has_changes(self) -> bool:
        """Check if this step has any changes."""
        return len(self.changes) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'day_index': self.day_index,
            'changes': [c.to_dict() for c in self.changes],
            'label': self.label,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DayStep":
        """Create from dictionary."""
        changes = [ParameterChange.from_dict(c) for c in data.get('changes', [])]
        return cls(
            day_index=data['day_index'],
            changes=changes,
            label=data.get('label'),
            metadata=data.get('metadata', {}),
        )
    
    def __repr__(self) -> str:
        label_str = f" ({self.label})" if self.label else ""
        return f"DayStep(day={self.day_index}, changes={len(self.changes)}{label_str})"


@dataclass
class DayPath:
    """
    Represents a complete multi-day market evolution scenario.
    
    A DayPath is a sequence of DaySteps that defines how market parameters
    evolve over multiple days. It's the core input for dynamic scenario analysis.
    
    Attributes:
        name: Name of the path scenario
        steps: List of day steps (one per day)
        description: Optional description of the scenario
        start_date: Optional start date for the path
        metadata: Optional additional data
        
    Example:
        >>> path = DayPath(
        ...     name="5-Day Rally",
        ...     steps=[
        ...         DayStep(0, [ParameterChange("spot", StressType.PERCENTAGE, 0.02)]),
        ...         DayStep(1, [ParameterChange("spot", StressType.PERCENTAGE, 0.02)]),
        ...         DayStep(2, [ParameterChange("spot", StressType.PERCENTAGE, 0.02)]),
        ...         DayStep(3, [ParameterChange("spot", StressType.PERCENTAGE, 0.02)]),
        ...         DayStep(4, [ParameterChange("spot", StressType.PERCENTAGE, 0.02)]),
        ...     ],
        ...     description="5 consecutive days of 2% price increase"
        ... )
    """
    name: str
    steps: List[DayStep]
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate day path."""
        if not self.name:
            raise ValidationError("Path name is required")
        
        if not self.steps:
            raise ValidationError("At least one step is required")
        
        # Validate steps are in order
        for i, step in enumerate(self.steps):
            if step.day_index != i:
                raise ValidationError(
                    f"Steps must be sequential. Expected day_index={i}, got {step.day_index}"
                )
    
    @property
    def num_days(self) -> int:
        """Get number of days in this path."""
        return len(self.steps)
    
    def get_step(self, day_index: int) -> Optional[DayStep]:
        """
        Get step for a specific day.
        
        Args:
            day_index: 0-based day index
            
        Returns:
            DayStep if exists, None otherwise
        """
        if 0 <= day_index < len(self.steps):
            return self.steps[day_index]
        return None
    
    def get_date_for_day(self, day_index: int) -> Optional[datetime]:
        """
        Get the date for a specific day index.
        
        Args:
            day_index: 0-based day index
            
        Returns:
            Date if start_date is set, None otherwise
        """
        if self.start_date is None:
            return None
        return self.start_date + timedelta(days=day_index)
    
    def get_all_parameters(self) -> set:
        """Get set of all parameters changed in this path."""
        params = set()
        for step in self.steps:
            for change in step.changes:
                params.add(change.parameter)
        return params
    
    def get_all_underlyings(self) -> set:
        """Get set of all underlyings targeted in this path."""
        underlyings = set()
        for step in self.steps:
            for change in step.changes:
                if change.level == StressLevel.UNDERLYING and change.target:
                    underlyings.add(change.target)
        return underlyings
    
    def __iter__(self) -> Iterator[DayStep]:
        """Iterate over day steps."""
        return iter(self.steps)
    
    def __len__(self) -> int:
        """Get number of days."""
        return len(self.steps)
    
    def __getitem__(self, index: int) -> DayStep:
        """Get step by index."""
        return self.steps[index]
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"DayPath: {self.name}",
            f"Number of days: {self.num_days}",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.start_date:
            end_date = self.get_date_for_day(self.num_days - 1)
            lines.append(f"Period: {self.start_date.date()} to {end_date.date()}")
        
        lines.append(f"Parameters changed: {', '.join(sorted(self.get_all_parameters()))}")
        
        lines.append("\nDay-by-day:")
        for step in self.steps:
            date_str = ""
            if self.start_date:
                date = self.get_date_for_day(step.day_index)
                date_str = f" ({date.date()})"
            
            label = step.label or ""
            changes_str = ", ".join(str(c) for c in step.changes) if step.changes else "no changes"
            lines.append(f"  Day {step.day_index}{date_str}: {changes_str}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'steps': [s.to_dict() for s in self.steps],
            'description': self.description,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DayPath":
        """Create from dictionary."""
        steps = [DayStep.from_dict(s) for s in data['steps']]
        start_date = None
        if data.get('start_date'):
            start_date = datetime.fromisoformat(data['start_date'])
        
        return cls(
            name=data['name'],
            steps=steps,
            description=data.get('description'),
            start_date=start_date,
            metadata=data.get('metadata', {}),
        )
    
    def __repr__(self) -> str:
        return f"DayPath(name='{self.name}', days={self.num_days})"

