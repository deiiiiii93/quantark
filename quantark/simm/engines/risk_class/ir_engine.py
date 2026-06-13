"""
Interest Rate sensitivity engine.

This module implements the IR sensitivity engine for SIMM calculations, handling
delta (DV01), vega, and curvature sensitivities for fixed income positions.
"""

from typing import List, Dict, Any

from quantark.simm.taxonomy import RiskClass, IRSubCurve, get_currency_volatility
from quantark.simm.sensitivity import IRDeltaSensitivity, IRVegaSensitivity, CurvatureSensitivity
from quantark.simm.engines.base import BaseSensitivityEngine, SIMMSensitivityProvider
from quantark.util.exceptions import ValidationError

from quantark.portfolio.fi.position import FIPosition


class IRSensitivityEngine(BaseSensitivityEngine):
    """
    Sensitivity engine for Interest Rate risk class.

    This engine calculates IR delta (PV01), vega, and curvature sensitivities
    for fixed income positions including bonds, swaps, and swaptions.

    Integration:
    - Requires independently shocked market-vertex sensitivities
    - Supports all IR sub-curves (OIS, LIBOR, etc.)
    - Handles currency volatility classification
    """

    @property
    def risk_class(self) -> RiskClass:
        """Interest Rate risk class."""
        return RiskClass.INTEREST_RATE

    def calculate_delta_sensitivities(
        self,
        positions: List[FIPosition],
        pricing_environments: Dict[str, Any],
    ) -> List[IRDeltaSensitivity]:
        """
        Calculate provider-supplied IR delta sensitivities.

        Args:
            positions: List of fixed income positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            List of IRDeltaSensitivity objects
        """
        return self._provider_sensitivities(positions, pricing_environments, "Delta")

    def calculate_vega_sensitivities(
        self,
        positions: List[FIPosition],
        pricing_environments: Dict[str, Any],
    ) -> List[IRVegaSensitivity]:
        """
        Calculate IR vega sensitivities.

        Vega must be supplied by a compliant position provider.

        Args:
            positions: List of fixed income positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            List of IRVegaSensitivity objects
        """
        return self._provider_sensitivities(positions, pricing_environments, "Vega")

    def calculate_curvature_sensitivities(
        self,
        positions: List[FIPosition],
        pricing_environments: Dict[str, Any],
    ) -> List[CurvatureSensitivity]:
        """
        Calculate IR curvature sensitivities.

        Curvature must be supplied by a compliant position provider.

        Args:
            positions: List of fixed income positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            List of CurvatureSensitivity objects
        """
        return self._provider_sensitivities(positions, pricing_environments, "Curvature")

    def _provider_sensitivities(
        self, positions: List[Any], market_data: Dict[str, Any], margin_type: str
    ) -> List[Any]:
        sensitivities: List[Any] = []
        for position in positions:
            if not hasattr(position, "get_dv01"):
                continue
            if not isinstance(position, SIMMSensitivityProvider):
                raise ValidationError(
                    f"Position {getattr(position, 'position_id', position)!r} must "
                    "provide independently shocked IR SIMM sensitivities; total DV01 "
                    "cannot be allocated across SIMM vertices"
                )
            supplied = position.get_simm_sensitivities(self.config, market_data)
            sensitivities.extend(
                s for s in supplied.by_risk_class(RiskClass.INTEREST_RATE)
                if s.margin_type.value == margin_type
            )
        return sensitivities

    def _determine_sub_curve(self, position: FIPosition) -> IRSubCurve:
        """
        Determine the appropriate IR sub-curve for a position.

        Args:
            position: The fixed income position

        Returns:
            IRSubCurve enum value
        """
        # Default to OIS
        # TODO: Implement proper sub-curve determination based on product
        # This could be based on product type, currency, or other factors

        # For now, map based on underlying currency
        # This is a simplification - production code would be more sophisticated
        underlying = position.underlying.upper()

        if underlying in ["USD", "EUR", "GBP", "JPY"]:
            return IRSubCurve.OIS
        else:
            return IRSubCurve.OIS

    def _calculate_tenor_weights(self, position: FIPosition) -> List[float]:
        """
        Calculate weights for distributing DV01 across tenor buckets.

        This method implements approximate key rate duration bucketing.
        In production, this would use actual key rate durations or
        proper curve bucketing methodology.

        Args:
            position: The fixed income position

        Returns:
            List of weights summing to 1.0
        """
        raise ValidationError(
            "Total DV01 cannot be distributed across SIMM vertices; provide "
            "independently shocked key-rate sensitivities"
        )

    def classify_to_buckets(
        self,
        positions: List[FIPosition],
        pricing_environments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Classify positions to IR buckets.

        For IR risk class, buckets are currencies with volatility classification.
        This method classifies each position by currency and volatility group.

        Args:
            positions: List of fixed income positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            Dict mapping position_id to bucket information
        """
        classification = {}

        for position in positions:
            # Check if position has required FI methods (duck typing)
            if not (hasattr(position, 'get_dv01') and hasattr(position, 'underlying')):
                continue

            env = pricing_environments.get(position.underlying)
            if env is None:
                continue

            # Currency is the bucket for IR
            currency = position.underlying.upper()

            # Determine currency volatility classification
            # (Low, Regular, High volatility currencies per SIMM spec)
            vol_class = self._classify_currency_volatility(currency)

            classification[position.position_id] = {
                "bucket": currency,
                "currency": currency,
                "volatility_class": vol_class,
                "sub_curve": self._determine_sub_curve(position).value,
            }

        return classification

    def _classify_currency_volatility(self, currency: str) -> str:
        """
        Classify currency by volatility group.

        Per SIMM v2.6 specification, only JPY is low volatility.

        Args:
            currency: ISO currency code

        Returns:
            Volatility classification string
        """
        return get_currency_volatility(currency).value
