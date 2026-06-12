"""
Equity sensitivity engine.

This module implements the Equity sensitivity engine for SIMM calculations, handling
delta and vega sensitivities for equity positions.
"""

from typing import List, Dict, Any, Optional
from decimal import Decimal

from quantark.simm.config import SIMMConfig
from quantark.simm.taxonomy import RiskClass
from quantark.simm.sensitivity import (
    EquityDeltaSensitivity,
    EquityVegaSensitivity,
    vol_weighted_vega_equity,
)
from quantark.simm.engines.base import BaseSensitivityEngine
from quantark.simm.engines.classification.bucket_mapper import BucketMapper
from quantark.util.numerical import Tolerance, is_zero

from quantark.portfolio.equity.position import EquityPosition
from quantark.asset.equity.riskmeasures import GreeksCalculator


class EquitySensitivityEngine(BaseSensitivityEngine):
    """
    Sensitivity engine for Equity risk class.

    This engine calculates equity delta and vega sensitivities for equity positions
    including stocks, options, and equity derivatives.

    Integration:
    - Uses GreeksCalculator for analytical/numerical Greeks
    - Integrates with EquityPosition.get_greeks() method
    - Supports bucket classification (12 equity buckets)
    - Handles option expiry-based vega tenor assignment
    """

    def __init__(self, config: SIMMConfig, greeks_calculator: Optional[GreeksCalculator] = None):
        """
        Initialize the equity sensitivity engine.

        Args:
            config: SIMM configuration settings
            greeks_calculator: GreeksCalculator instance (creates default if None)
        """
        super().__init__(config)
        self.greeks_calculator = greeks_calculator if greeks_calculator is not None else GreeksCalculator()
        self.bucket_mapper = BucketMapper()

    @property
    def risk_class(self) -> RiskClass:
        """Equity risk class."""
        return RiskClass.EQUITY

    def calculate_delta_sensitivities(
        self,
        positions: List[EquityPosition],
        pricing_environments: Dict[str, Any],
    ) -> List[EquityDeltaSensitivity]:
        """
        Calculate equity delta sensitivities.

        Uses the GreeksCalculator to extract delta from positions.
        Scales by position quantity and classifies to appropriate bucket.

        Args:
            positions: List of equity positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            List of EquityDeltaSensitivity objects
        """
        sensitivities = []

        for position in positions:
            # Check if position has required Equity methods (duck typing)
            if not (hasattr(position, 'get_greeks') and hasattr(position, 'underlying')):
                # Skip non-equity positions
                continue

            env = pricing_environments.get(position.underlying)
            if env is None:
                # Skip if no pricing environment available
                continue

            # Get Greeks from position
            greeks = position.get_greeks(env, self.greeks_calculator)
            delta = greeks.get('delta', 0.0)

            # Scale by position quantity
            delta *= position.quantity

            if is_zero(delta, tol=Tolerance.ZERO):
                # Skip positions with negligible delta
                continue

            # SIMM equity delta is per 1% relative spot move (paragraph 26):
            # s = V(EQ + 1%.EQ) - V(EQ) ~= 0.01 * EQ * dV/dEQ.
            amount = 0.01 * env.spot * delta

            # Classify to bucket
            bucket = self._classify_equity_bucket(position.underlying)

            sensitivity = EquityDeltaSensitivity(
                trade_id=position.position_id,
                amount=amount,
                issuer=position.underlying,
                bucket_number=bucket,
            )
            sensitivities.append(sensitivity)

        return sensitivities

    def calculate_vega_sensitivities(
        self,
        positions: List[EquityPosition],
        pricing_environments: Dict[str, Any],
    ) -> List[EquityVegaSensitivity]:
        """
        Calculate equity vega sensitivities.

        Extracts vega from positions and assigns to vega tenors based on option expiry.

        Args:
            positions: List of equity positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            List of EquityVegaSensitivity objects
        """
        sensitivities = []

        for position in positions:
            # Check if position has required Equity methods (duck typing)
            if not (hasattr(position, 'get_greeks') and hasattr(position, 'underlying')):
                # Skip non-equity positions
                continue

            env = pricing_environments.get(position.underlying)
            if env is None:
                # Skip if no pricing environment available
                continue

            # Get Greeks from position
            greeks = position.get_greeks(env, self.greeks_calculator)
            vega = greeks.get('vega', 0.0)

            # Skip if no vega (e.g., equity spot, futures)
            if is_zero(vega, tol=Tolerance.ZERO):
                continue

            # Scale by position quantity
            vega *= position.quantity

            # Classify to bucket
            bucket = self._classify_equity_bucket(position.underlying)

            # Get option expiry for vega tenor
            option_tenor = self._get_option_expiry_tenor(position)

            # SIMM vega amounts are vol-weighted: sigma_kj * dV/dsigma,
            # with sigma_kj derived from the delta risk weight
            # (paragraph 10(b)). GreeksCalculator vega is per 1 vol point.
            amount = vol_weighted_vega_equity(vega, bucket)

            sensitivity = EquityVegaSensitivity(
                trade_id=position.position_id,
                amount=amount,
                issuer=position.underlying,
                bucket_number=bucket,
                option_tenor=option_tenor,
            )
            sensitivities.append(sensitivity)

        return sensitivities

    def _classify_equity_bucket(self, issuer: str) -> int:
        """
        Classify an equity to a SIMM bucket.

        Uses the BucketMapper to determine the appropriate bucket based on:
        - Region (Developed vs Emerging)
        - Sector (Technology, Financials, etc.)
        - Instrument type (ETF, Index, Stock)

        Args:
            issuer: Issuer identifier (ticker, name, etc.)

        Returns:
            Bucket number (1-12)
        """
        return self.bucket_mapper.classify_equity_bucket(issuer)

    def _get_option_expiry_tenor(self, position: EquityPosition) -> float:
        """
        Get the option expiry tenor for vega bucket assignment.

        Args:
            position: The equity position

        Returns:
            Option expiry in years (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
        """
        # Check if product has maturity/expiry attribute
        if hasattr(position.product, 'get_maturity'):
            # Get maturity from product
            # Note: This is approximate - production code would get from pricing env
            maturity = position.product.get_maturity(None)
            return float(maturity)

        # Default to 1 year if maturity not available
        return 1.0

    def classify_to_buckets(
        self,
        positions: List[EquityPosition],
        pricing_environments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Classify positions to equity buckets.

        Args:
            positions: List of equity positions
            pricing_environments: Dict mapping underlying to pricing environment

        Returns:
            Dict mapping position_id to bucket information
        """
        classification = {}

        for position in positions:
            # Check if position has required Equity methods (duck typing)
            if not (hasattr(position, 'get_greeks') and hasattr(position, 'underlying')):
                continue

            env = pricing_environments.get(position.underlying)
            if env is None:
                continue

            bucket = self._classify_equity_bucket(position.underlying)
            bucket_info = self.bucket_mapper.get_bucket_info("equity", position.underlying)

            classification[position.position_id] = {
                "bucket": bucket,
                "issuer": position.underlying,
                "description": bucket_info.description if bucket_info else None,
                "region": bucket_info.region if bucket_info else None,
                "sector": bucket_info.sector if bucket_info else None,
            }

        return classification

    def _is_equity_spot_or_futures(self, position: EquityPosition) -> bool:
        """
        Determine if a position is equity spot or futures (delta-one instruments).

        Args:
            position: The equity position

        Returns:
            True if delta-one instrument
        """
        # Check product type
        # This is a simplified check - production code would use more sophisticated logic
        product_type = type(position.product).__name__.upper()

        if "SPOT" in product_type or "FUTURE" in product_type or "FUTURES" in product_type:
            return True

        # Check if product has no option-like features
        # (i.e., doesn't have strike, call/put type, etc.)
        if not hasattr(position.product, 'strike'):
            return True

        return False
