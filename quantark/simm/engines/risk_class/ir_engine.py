"""
Interest Rate sensitivity engine.

This module implements the IR sensitivity engine for SIMM calculations, handling
delta (DV01), vega, and curvature sensitivities for fixed income positions.
"""

from typing import List, Dict, Any, Optional
from decimal import Decimal

from quantark.simm.config import SIMMConfig
from quantark.simm.taxonomy import RiskClass, IRSubCurve, IR_TENORS
from quantark.simm.sensitivity import IRDeltaSensitivity, IRVegaSensitivity, CurvatureSensitivity
from quantark.simm.engines.base import BaseSensitivityEngine

from quantark.portfolio.fi.position import FIPosition


class IRSensitivityEngine(BaseSensitivityEngine):
    """
    Sensitivity engine for Interest Rate risk class.

    This engine calculates IR delta (PV01), vega, and curvature sensitivities
    for fixed income positions including bonds, swaps, and swaptions.

    Integration:
    - Uses FIPosition.get_dv01() for delta calculation
    - Supports all IR sub-curves (OIS, LIBOR, etc.)
    - Distributes DV01 across tenor buckets
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
        Calculate IR delta sensitivities via DV01 bucketing.

        Distributes the position DV01 across the SIMM tenor bucket vertices
        using approximate key rate duration methodology.

        Args:
            positions: List of fixed income positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            List of IRDeltaSensitivity objects
        """
        sensitivities = []

        for position in positions:
            # Check if position has required FI methods (duck typing)
            if not (hasattr(position, 'get_dv01') and hasattr(position, 'underlying')):
                # Skip non-FI positions for IR engine
                continue

            env = pricing_environments.get(position.underlying)
            if env is None:
                # Skip if no pricing environment available
                continue

            # Get total position DV01
            total_dv01 = position.get_dv01(env)

            if abs(total_dv01) < 1e-10:
                # Skip positions with negligible DV01
                continue

            # Determine sub-curve based on product type
            sub_curve = self._determine_sub_curve(position)

            # Distribute DV01 across tenor buckets
            tenor_weights = self._calculate_tenor_weights(position)

            for tenor_idx, tenor in enumerate(IR_TENORS):
                weight = tenor_weights[tenor_idx] if tenor_idx < len(tenor_weights) else 0.0
                dv01_contribution = total_dv01 * weight

                if abs(dv01_contribution) > 1e-10:
                    sensitivity = IRDeltaSensitivity(
                        trade_id=position.position_id,
                        amount=dv01_contribution,
                        currency=position.underlying,
                        tenor=tenor,
                        sub_curve=sub_curve,
                    )
                    sensitivities.append(sensitivity)

        return sensitivities

    def calculate_vega_sensitivities(
        self,
        positions: List[FIPosition],
        pricing_environments: Dict[str, Any],
    ) -> List[IRVegaSensitivity]:
        """
        Calculate IR vega sensitivities.

        This implementation provides a simplified vega calculation.
        In production, this would integrate with swaption volatility surfaces
        and use proper volatility sensitivities.

        Args:
            positions: List of fixed income positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            List of IRVegaSensitivity objects
        """
        # TODO: Implement full IR vega calculation
        # For now, return empty list as IR vega requires
        # swaption volatility surface integration

        return []

    def calculate_curvature_sensitivities(
        self,
        positions: List[FIPosition],
        pricing_environments: Dict[str, Any],
    ) -> List[CurvatureSensitivity]:
        """
        Calculate IR curvature sensitivities.

        Curvature represents the convexity adjustment for non-linear rate exposures.
        This implementation uses simplified CVR calculation.

        Args:
            positions: List of fixed income positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            List of CurvatureSensitivity objects
        """
        # TODO: Implement full IR curvature calculation
        # For now, return empty list

        return []

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
        # Simplified approach: equal weighting across all tenors
        # In production, this would use:
        # 1. Actual key rate durations
        # 2. Maturity-based weighting
        # 3. Product-specific bucketing (bullet vs amortizing)
        # 4. Instrument-specific factors

        weight = 1.0 / len(IR_TENORS)
        return [weight] * len(IR_TENORS)

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

        Per SIMM v2.6 specification:
        - Low volatility: USD, EUR, GBP, CHF, etc.
        - Regular volatility: CAD, AUD, NZD, etc.
        - High volatility: Emerging market currencies

        Args:
            currency: ISO currency code

        Returns:
            Volatility classification string
        """
        low_vol = {"USD", "EUR", "GBP", "CHF", "JPY", "CNY"}
        regular_vol = {"CAD", "AUD", "NZD", "SEK", "NOK", "DKK"}

        currency_upper = currency.upper()

        if currency_upper in low_vol:
            return "Low"
        elif currency_upper in regular_vol:
            return "Regular"
        else:
            # Default to High for emerging market currencies
            return "High"
