"""
Fluent API for building stress test scenarios.
"""

from typing import Optional, List, Dict, Any
from stresstest.scenario.scenario import Scenario, Stress
from stresstest.stress.stress_types import StressType, StressLevel


class ScenarioBuilder:
    """
    Fluent builder for creating stress test scenarios.

    Provides a convenient API for constructing scenarios with multiple
    stresses without verbose boilerplate code.

    Example:
        >>> scenario = (ScenarioBuilder()
        ...     .name("Market Crash")
        ...     .description("Severe market downturn")
        ...     .spot_stress(-0.20)  # -20% spot
        ...     .vol_stress(0.50)     # +50% volatility
        ...     .rate_stress(-0.01, stress_type=StressType.ABSOLUTE)  # -100bps
        ...     .build())

        >>> # Target specific underlying
        >>> scenario = (ScenarioBuilder()
        ...     .name("AAPL Specific")
        ...     .spot_stress(-0.10, underlying="AAPL")
        ...     .build())
    """

    def __init__(self):
        """Initialize builder with empty state."""
        self._name: Optional[str] = None
        self._description: Optional[str] = None
        self._stresses: List[Stress] = []
        self._metadata: dict = {}

    def name(self, name: str) -> "ScenarioBuilder":
        """
        Set scenario name.

        Args:
            name: Scenario name

        Returns:
            Self for chaining
        """
        self._name = name
        return self

    def description(self, description: str) -> "ScenarioBuilder":
        """
        Set scenario description.

        Args:
            description: Scenario description

        Returns:
            Self for chaining
        """
        self._description = description
        return self

    def metadata(self, key: str, value) -> "ScenarioBuilder":
        """
        Add metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self

    def add_stress(
        self,
        parameter: str,
        stress_value: float,
        stress_type: StressType = StressType.PERCENTAGE,
        level: StressLevel = StressLevel.PORTFOLIO,
        target: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ScenarioBuilder":
        """
        Add a generic stress to the scenario.

        Args:
            parameter: Parameter name to stress
            stress_value: Magnitude of stress
            stress_type: Type of stress (default: PERCENTAGE)
            level: Level to apply stress (default: PORTFOLIO)
            target: Target identifier if level is UNDERLYING or POSITION
            description: Optional description of this stress

        Returns:
            Self for chaining
        """
        stress = Stress(
            parameter=parameter,
            stress_type=stress_type,
            stress_value=stress_value,
            level=level,
            target=target,
            description=description,
            metadata=metadata or {},
        )
        self._stresses.append(stress)
        return self

    def spot_stress(
        self,
        stress_value: float,
        stress_type: StressType = StressType.PERCENTAGE,
        underlying: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> "ScenarioBuilder":
        """
        Add a spot price stress.

        Args:
            stress_value: Stress magnitude
            stress_type: Type of stress (default: PERCENTAGE)
            underlying: Target underlying (if None, applies to portfolio)
            position_id: Target position (if specified, overrides underlying)

        Returns:
            Self for chaining
        """
        level, target = self._determine_level_target(underlying, position_id)
        return self.add_stress("spot", stress_value, stress_type, level, target)

    def vol_stress(
        self,
        stress_value: float,
        stress_type: StressType = StressType.PERCENTAGE,
        underlying: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> "ScenarioBuilder":
        """
        Add a volatility stress.

        Args:
            stress_value: Stress magnitude
            stress_type: Type of stress (default: PERCENTAGE)
            underlying: Target underlying (if None, applies to portfolio)
            position_id: Target position (if specified, overrides underlying)

        Returns:
            Self for chaining
        """
        level, target = self._determine_level_target(underlying, position_id)
        return self.add_stress("volatility", stress_value, stress_type, level, target)

    def rate_stress(
        self,
        stress_value: float,
        stress_type: StressType = StressType.PERCENTAGE,
        underlying: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> "ScenarioBuilder":
        """
        Add an interest rate stress.

        Args:
            stress_value: Stress magnitude
            stress_type: Type of stress (default: PERCENTAGE)
            underlying: Target underlying (if None, applies to portfolio)
            position_id: Target position (if specified, overrides underlying)

        Returns:
            Self for chaining
        """
        level, target = self._determine_level_target(underlying, position_id)
        return self.add_stress("rate", stress_value, stress_type, level, target)

    def key_rate_stress(
        self,
        stress_value: float,
        tenor_bucket: str,
        stress_type: StressType = StressType.ABSOLUTE,
        curve: Optional[str] = None,
        underlying: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> "ScenarioBuilder":
        """
        Add a key-rate stress targeting a specific tenor bucket.
        """
        level, target = self._determine_level_target(underlying, position_id)
        metadata = {"tenor_bucket": tenor_bucket}
        if curve:
            metadata["curve"] = curve
        return self.add_stress(
            "key_rate", stress_value, stress_type, level, target, metadata=metadata
        )

    def spread_stress(
        self,
        stress_value: float,
        stress_type: StressType = StressType.ABSOLUTE,
        spread_curve: Optional[str] = None,
        underlying: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> "ScenarioBuilder":
        """
        Add a credit spread stress.
        """
        level, target = self._determine_level_target(underlying, position_id)
        metadata = {}
        if spread_curve:
            metadata["spread_curve"] = spread_curve
        return self.add_stress(
            "spread", stress_value, stress_type, level, target, metadata=metadata
        )

    def div_yield_stress(
        self,
        stress_value: float,
        stress_type: StressType = StressType.PERCENTAGE,
        underlying: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> "ScenarioBuilder":
        """
        Add a dividend yield stress.

        Args:
            stress_value: Stress magnitude
            stress_type: Type of stress (default: PERCENTAGE)
            underlying: Target underlying (if None, applies to portfolio)
            position_id: Target position (if specified, overrides underlying)

        Returns:
            Self for chaining
        """
        level, target = self._determine_level_target(underlying, position_id)
        return self.add_stress(
            "dividend_yield", stress_value, stress_type, level, target
        )

    def _determine_level_target(
        self, underlying: Optional[str], position_id: Optional[str]
    ) -> tuple[StressLevel, Optional[str]]:
        """Determine stress level and target from parameters."""
        if position_id:
            return StressLevel.POSITION, position_id
        elif underlying:
            return StressLevel.UNDERLYING, underlying
        else:
            return StressLevel.PORTFOLIO, None

    def build(self) -> Scenario:
        """
        Build the scenario.

        Returns:
            Constructed Scenario object

        Raises:
            ValueError: If required fields are missing
        """
        if not self._name:
            raise ValueError("Scenario name is required")

        if not self._stresses:
            raise ValueError("At least one stress is required")

        return Scenario(
            name=self._name,
            stresses=self._stresses,
            description=self._description,
            metadata=self._metadata,
        )

    def reset(self) -> "ScenarioBuilder":
        """
        Reset builder to initial state.

        Returns:
            Self for chaining
        """
        self._name = None
        self._description = None
        self._stresses = []
        self._metadata = {}
        return self
