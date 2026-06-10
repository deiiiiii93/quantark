"""
Base classes and protocols for SIMM sensitivity engines.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any, Optional, TypeVar, Generic
from typing_extensions import runtime_checkable

from quantark.simm.config import SIMMConfig
from quantark.simm.taxonomy import RiskClass, MarginType
from quantark.simm.sensitivity import SensitivityCollection, AnySensitivity


@runtime_checkable
class SensitivityEngine(Protocol):
    """
    Protocol for all SIMM sensitivity engines.

    This protocol defines the interface that all sensitivity engines must implement.
    It enables duck-typing and flexibility in engine implementations while maintaining
    type safety.
    """

    @property
    def risk_class(self) -> RiskClass:
        """The risk class this engine handles."""
        ...

    def calculate_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
        config: SIMMConfig,
    ) -> SensitivityCollection:
        """
        Calculate sensitivities from positions.

        Args:
            positions: List of positions to calculate sensitivities for
            pricing_environments: Dict mapping position identifiers to pricing environments
            config: SIMM configuration settings

        Returns:
            SensitivityCollection containing all calculated sensitivities
        """
        ...


class BaseSensitivityEngine(ABC):
    """
    Abstract base class implementing common engine functionality.

    This base class provides common logic that can be shared across different
    sensitivity engines, while allowing subclasses to override specific methods
    for their risk class.
    """

    def __init__(self, config: SIMMConfig):
        """
        Initialize the sensitivity engine.

        Args:
            config: SIMM configuration settings
        """
        self.config = config

    @property
    @abstractmethod
    def risk_class(self) -> RiskClass:
        """The risk class this engine handles."""
        ...

    def calculate_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
        config: Optional[SIMMConfig] = None,
    ) -> SensitivityCollection:
        """
        Calculate all sensitivity types based on configuration.

        This method orchestrates the calculation of delta, vega, and curvature
        sensitivities based on the configuration settings.

        Args:
            positions: List of positions to calculate sensitivities for
            pricing_environments: Dict mapping position identifiers to pricing environments
            config: SIMM configuration settings (defaults to self.config if not provided)

        Returns:
            SensitivityCollection containing all calculated sensitivities
        """
        if config is None:
            config = self.config

        collection = SensitivityCollection()

        # Calculate delta sensitivities if enabled in config
        if config.calculate_delta:
            delta_sens = self.calculate_delta_sensitivities(
                positions, pricing_environments
            )
            if delta_sens:
                collection.add_many(delta_sens)

        # Calculate vega sensitivities if enabled in config
        if config.calculate_vega:
            vega_sens = self.calculate_vega_sensitivities(
                positions, pricing_environments
            )
            if vega_sens:
                collection.add_many(vega_sens)

        # Calculate curvature sensitivities if enabled in config
        if config.calculate_curvature:
            curv_sens = self.calculate_curvature_sensitivities(
                positions, pricing_environments
            )
            if curv_sens:
                collection.add_many(curv_sens)

        # Calculate base correlation sensitivities if enabled in config
        if config.calculate_base_corr:
            base_corr_sens = self.calculate_base_corr_sensitivities(
                positions, pricing_environments
            )
            if base_corr_sens:
                collection.add_many(base_corr_sens)

        return collection

    def calculate_delta_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> List[AnySensitivity]:
        """
        Calculate delta sensitivities.

        Subclasses should override this method to implement delta sensitivity
        calculation specific to their risk class.

        Args:
            positions: List of positions
            pricing_environments: Dict mapping position identifiers to pricing environments

        Returns:
            List of delta sensitivities
        """
        return []

    def calculate_vega_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> List[AnySensitivity]:
        """
        Calculate vega sensitivities.

        Subclasses should override this method to implement vega sensitivity
        calculation specific to their risk class.

        Args:
            positions: List of positions
            pricing_environments: Dict mapping position identifiers to pricing environments

        Returns:
            List of vega sensitivities
        """
        return []

    def calculate_curvature_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> List[AnySensitivity]:
        """
        Calculate curvature sensitivities.

        Subclasses should override this method to implement curvature sensitivity
        calculation specific to their risk class.

        Args:
            positions: List of positions
            pricing_environments: Dict mapping position identifiers to pricing environments

        Returns:
            List of curvature sensitivities
        """
        return []

    def calculate_base_corr_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> List[AnySensitivity]:
        """
        Calculate base correlation sensitivities.

        This is primarily used for credit risk classes. Subclasses should override
        this method if base correlation calculations are relevant.

        Args:
            positions: List of positions
            pricing_environments: Dict mapping position identifiers to pricing environments

        Returns:
            List of base correlation sensitivities
        """
        return []

    def classify_to_buckets(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Classify positions to SIMM buckets.

        This method can be used to categorize positions into their appropriate
        SIMM buckets for margin calculation.

        Args:
            positions: List of positions
            pricing_environments: Dict mapping position identifiers to pricing environments

        Returns:
            Dict mapping position identifiers to bucket information
        """
        # Default implementation - subclasses should override
        return {}
