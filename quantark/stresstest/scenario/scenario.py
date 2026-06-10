"""
Core scenario and stress definitions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from quantark.stresstest.stress.stress_types import StressType, StressLevel
from quantark.util.exceptions import ValidationError


@dataclass
class Stress:
    """
    Represents a single stress to apply to a market parameter.

    A stress defines how to modify a specific parameter in the pricing
    environment, at a specific level (portfolio, underlying, or position).

    Attributes:
        parameter: Name of the parameter to stress (e.g., "spot", "volatility", "rate")
        stress_type: Type of stress (ABSOLUTE, PERCENTAGE, VALUE)
        stress_value: Magnitude of the stress
        level: Level to apply stress (PORTFOLIO, UNDERLYING, POSITION)
        target: Optional target identifier (underlying/position_id) if level is not PORTFOLIO
        description: Optional human-readable description of this stress

    Examples:
        >>> # Stress spot down by 20% at portfolio level
        >>> Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO)

        >>> # Stress volatility up by 10 points (absolute) for AAPL
        >>> Stress("volatility", StressType.ABSOLUTE, 0.10, StressLevel.UNDERLYING, "AAPL")

        >>> # Set rate to 5% for specific position
        >>> Stress("rate", StressType.VALUE, 0.05, StressLevel.POSITION, "pos_123")
    """

    parameter: str
    stress_type: StressType
    stress_value: float
    level: StressLevel
    target: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate stress parameters."""
        if not self.parameter:
            raise ValidationError("Parameter name is required")

        if self.level == StressLevel.UNDERLYING and not self.target:
            raise ValidationError(
                "Target underlying is required for UNDERLYING level stress"
            )

        if self.level == StressLevel.POSITION and not self.target:
            raise ValidationError(
                "Target position_id is required for POSITION level stress"
            )

        if self.level == StressLevel.PORTFOLIO and self.target:
            raise ValidationError(
                "Target should not be specified for PORTFOLIO level stress"
            )

        if not isinstance(self.metadata, dict):
            raise ValidationError("Stress metadata must be a dictionary")

        self._validate_specialized_metadata()

    def _validate_specialized_metadata(self) -> None:
        param = self.parameter.lower()
        if param in {"rate", "key_rate"}:
            bucket = self.metadata.get("tenor_bucket")
            if bucket is not None and (
                not isinstance(bucket, str) or not bucket.strip()
            ):
                raise ValidationError(
                    "tenor_bucket metadata must be a non-empty string"
                )
        if param in {"spread", "credit_spread"}:
            spread_curve = self.metadata.get("spread_curve")
            if spread_curve is not None and (
                not isinstance(spread_curve, str) or not spread_curve.strip()
            ):
                raise ValidationError(
                    "spread_curve metadata must be a non-empty string"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert stress to dictionary for serialization."""
        return {
            "parameter": self.parameter,
            "stress_type": self.stress_type.value,
            "stress_value": self.stress_value,
            "level": self.level.value,
            "target": self.target,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stress":
        """Create stress from dictionary."""
        return cls(
            parameter=data["parameter"],
            stress_type=StressType(data["stress_type"]),
            stress_value=data["stress_value"],
            level=StressLevel(data["level"]),
            target=data.get("target"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        target_str = f" on {self.target}" if self.target else ""
        return (
            f"Stress({self.parameter} {self.stress_type.value} "
            f"{self.stress_value:+.2%} at {self.level.value}{target_str})"
        )


@dataclass
class Scenario:
    """
    Represents a stress testing scenario with one or more stresses.

    A scenario defines a market condition by specifying multiple stresses
    that should be applied simultaneously to the portfolio.

    Attributes:
        name: Unique name for the scenario
        stresses: List of stresses to apply in this scenario
        description: Human-readable description of the scenario
        metadata: Additional metadata (e.g., historical date, probability)
        created_at: When the scenario was created
        scenario_id: Optional unique identifier

    Examples:
        >>> # Market crash scenario
        >>> scenario = Scenario(
        ...     name="Market Crash",
        ...     description="20% equity drop with vol spike",
        ...     stresses=[
        ...         Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO),
        ...         Stress("volatility", StressType.PERCENTAGE, 0.50, StressLevel.PORTFOLIO),
        ...     ]
        ... )
    """

    name: str
    stresses: List[Stress]
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    scenario_id: Optional[str] = None

    def __post_init__(self):
        """Validate scenario."""
        if not self.name:
            raise ValidationError("Scenario name is required")

        if not self.stresses:
            raise ValidationError("At least one stress is required")

        if not isinstance(self.stresses, list):
            raise ValidationError("Stresses must be a list")

        # Validate all stresses
        for stress in self.stresses:
            if not isinstance(stress, Stress):
                raise ValidationError(f"Invalid stress object: {stress}")

    def get_stress_summary(self) -> str:
        """
        Get a summary of stresses in this scenario.

        Returns:
            Human-readable summary string
        """
        summary_lines = [f"Scenario: {self.name}"]
        if self.description:
            summary_lines.append(f"Description: {self.description}")
        summary_lines.append(f"Number of stresses: {len(self.stresses)}")
        summary_lines.append("Stresses:")
        for i, stress in enumerate(self.stresses, 1):
            summary_lines.append(f"  {i}. {stress}")
        return "\n".join(summary_lines)

    def get_portfolio_stresses(self) -> List[Stress]:
        """Get all portfolio-level stresses."""
        return [s for s in self.stresses if s.level == StressLevel.PORTFOLIO]

    def get_underlying_stresses(self, underlying: str) -> List[Stress]:
        """Get all stresses for a specific underlying."""
        return [
            s
            for s in self.stresses
            if s.level == StressLevel.UNDERLYING and s.target == underlying
        ]

    def get_position_stresses(self, position_id: str) -> List[Stress]:
        """Get all stresses for a specific position."""
        return [
            s
            for s in self.stresses
            if s.level == StressLevel.POSITION and s.target == position_id
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "stresses": [s.to_dict() for s in self.stresses],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "scenario_id": self.scenario_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """Create scenario from dictionary."""
        stresses = [Stress.from_dict(s) for s in data["stresses"]]
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now()
        )

        return cls(
            name=data["name"],
            stresses=stresses,
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            scenario_id=data.get("scenario_id"),
        )

    def __repr__(self) -> str:
        return f"Scenario(name='{self.name}', stresses={len(self.stresses)})"
