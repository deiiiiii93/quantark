"""
Bucket Aggregator for SIMM.

This module implements within-bucket aggregation using the K_b formula.

K_b = sqrt(Σ_k WS_k² + Σ_k Σ_{l≠k} ρ_kl × f_kl × WS_k × WS_l)

Where:
- WS_k is the weighted sensitivity for risk factor k
- ρ_kl is the intra-bucket correlation between k and l
- f_kl = min(CR_k, CR_l) / max(CR_k, CR_l) is the concentration adjustment
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable, Any

from quantark.simm.taxonomy import RiskClass, MarginType
from quantark.simm.engines.aggregation.weighted_sensitivity import WeightedSensitivity
from quantark.simm.calibration import get_intra_bucket_correlation
from quantark.simm.calibration.ir import IR_TENOR_CORRELATIONS, IR_SUB_CURVE_CORRELATION
from quantark.simm.calibration.credit_qualifying import CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS
from quantark.simm.calibration.credit_non_qualifying import CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS
from quantark.simm.calibration.equity import EQUITY_INTRA_BUCKET_CORRELATIONS
from quantark.simm.calibration.commodity import COMMODITY_INTRA_BUCKET_CORRELATIONS
from quantark.simm.calibration.fx import FX_CORRELATIONS


@dataclass
class BucketResult:
    """Result of bucket-level aggregation.
    
    Attributes:
        risk_class: The risk class.
        margin_type: Delta or Vega.
        bucket: The bucket identifier.
        k_b: The aggregated bucket margin (K_b).
        ws_sum: Sum of weighted sensitivities (Σ WS_k).
        weighted_sensitivities: List of weighted sensitivities in this bucket.
        is_residual: True if this is the residual bucket.
    """
    risk_class: RiskClass
    margin_type: MarginType
    bucket: Union[str, int]
    k_b: float
    ws_sum: float
    weighted_sensitivities: List[WeightedSensitivity] = field(default_factory=list)
    is_residual: bool = False


class BucketAggregator:
    """Aggregator for within-bucket margin calculation.
    
    Implements the K_b formula from SIMM paragraph 8(c).
    """
    
    def aggregate(
        self,
        weighted_sensitivities: List[WeightedSensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
        cr_values: Dict[str, float],
    ) -> BucketResult:
        """Aggregate weighted sensitivities within a bucket.
        
        K_b = sqrt(Σ_k WS_k² + Σ_k Σ_{l≠k} ρ_kl × f_kl × WS_k × WS_l)
        
        Args:
            weighted_sensitivities: List of weighted sensitivities.
            risk_class: The risk class.
            margin_type: Delta or Vega.
            bucket: The bucket identifier.
            cr_values: Dict mapping qualifier to concentration risk factor.
            
        Returns:
            BucketResult with aggregated K_b.
        """
        if not weighted_sensitivities:
            return BucketResult(
                risk_class=risk_class,
                margin_type=margin_type,
                bucket=bucket,
                k_b=0.0,
                ws_sum=0.0,
                weighted_sensitivities=[],
                is_residual=self._is_residual_bucket(bucket),
            )
        
        # Check if residual bucket - has no diversification benefit
        if self._is_residual_bucket(bucket):
            return self._aggregate_residual(weighted_sensitivities, risk_class, margin_type, bucket)
        
        # Get correlation function for this bucket
        corr_func = self._get_correlation_function(risk_class, margin_type, bucket)
        
        # Aggregate by qualifier to get net WS per risk factor
        ws_by_qualifier: Dict[str, float] = {}
        for ws in weighted_sensitivities:
            ws_by_qualifier[ws.qualifier] = ws_by_qualifier.get(ws.qualifier, 0.0) + ws.weighted_value
        
        qualifiers = list(ws_by_qualifier.keys())
        n = len(qualifiers)
        
        # Calculate diagonal terms: Σ WS_k²
        sum_sq = sum(ws**2 for ws in ws_by_qualifier.values())
        
        # Calculate cross terms: Σ_k Σ_{l≠k} ρ_kl × f_kl × WS_k × WS_l
        cross_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                q_i, q_j = qualifiers[i], qualifiers[j]
                
                # Get correlation
                rho = corr_func(q_i, q_j)
                
                # Calculate f_kl = min(CR_k, CR_l) / max(CR_k, CR_l)
                cr_i = cr_values.get(q_i, 1.0)
                cr_j = cr_values.get(q_j, 1.0)
                f_ij = min(cr_i, cr_j) / max(cr_i, cr_j) if max(cr_i, cr_j) > 0 else 1.0
                
                ws_i = ws_by_qualifier[q_i]
                ws_j = ws_by_qualifier[q_j]
                
                cross_sum += 2 * rho * f_ij * ws_i * ws_j
        
        # K_b = sqrt(max(0, Σ WS² + cross terms))
        k_b = math.sqrt(max(0, sum_sq + cross_sum))
        
        # Sum of weighted sensitivities
        ws_sum = sum(ws_by_qualifier.values())
        
        return BucketResult(
            risk_class=risk_class,
            margin_type=margin_type,
            bucket=bucket,
            k_b=k_b,
            ws_sum=ws_sum,
            weighted_sensitivities=weighted_sensitivities,
            is_residual=False,
        )
    
    def _aggregate_residual(
        self,
        weighted_sensitivities: List[WeightedSensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> BucketResult:
        """Aggregate residual bucket (simple sum of absolute values).
        
        Residual bucket has no diversification benefit.
        """
        ws_by_qualifier: Dict[str, float] = {}
        for ws in weighted_sensitivities:
            ws_by_qualifier[ws.qualifier] = ws_by_qualifier.get(ws.qualifier, 0.0) + ws.weighted_value
        
        # K_residual = Σ |WS_k|
        k_b = sum(abs(ws) for ws in ws_by_qualifier.values())
        ws_sum = sum(ws_by_qualifier.values())
        
        return BucketResult(
            risk_class=risk_class,
            margin_type=margin_type,
            bucket=bucket,
            k_b=k_b,
            ws_sum=ws_sum,
            weighted_sensitivities=weighted_sensitivities,
            is_residual=True,
        )
    
    def _is_residual_bucket(self, bucket: Union[str, int]) -> bool:
        """Check if bucket is the residual bucket."""
        if isinstance(bucket, str):
            return bucket.lower() in ("residual", "res", "-1")
        return bucket == -1
    
    def _get_correlation_function(
        self,
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> Callable[[str, str], float]:
        """Get the intra-bucket correlation function.
        
        Returns a function that takes two qualifiers and returns their correlation.
        """
        if risk_class == RiskClass.INTEREST_RATE:
            return self._get_ir_correlation_function(margin_type, bucket)
        elif risk_class == RiskClass.CREDIT_QUALIFYING:
            return self._get_credit_q_correlation_function(margin_type, bucket)
        elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
            return self._get_credit_nq_correlation_function(margin_type, bucket)
        elif risk_class == RiskClass.EQUITY:
            return self._get_equity_correlation_function(margin_type, bucket)
        elif risk_class == RiskClass.COMMODITY:
            return self._get_commodity_correlation_function(margin_type, bucket)
        elif risk_class == RiskClass.FX:
            return self._get_fx_correlation_function(margin_type)
        else:
            return lambda q1, q2: 0.0  # No correlation
    
    def _get_ir_correlation_function(
        self,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> Callable[[str, str], float]:
        """Get IR intra-bucket correlation function.
        
        IR correlations depend on tenor and sub-curve.
        """
        def corr_func(q1: str, q2: str) -> float:
            # For IR, correlation is based on tenor
            # This is a simplified version - full implementation would parse tenor from qualifier
            base_corr = IR_TENOR_CORRELATIONS.get("1y", {}).get("1y", 1.0)
            return base_corr
        
        return corr_func
    
    def _get_credit_q_correlation_function(
        self,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> Callable[[str, str], float]:
        """Get Credit Qualifying intra-bucket correlation function."""
        bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 1
        corr = CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS.get(bucket_num, 0.35)
        
        def corr_func(q1: str, q2: str) -> float:
            if q1 == q2:
                return 1.0
            return corr
        
        return corr_func
    
    def _get_credit_nq_correlation_function(
        self,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> Callable[[str, str], float]:
        """Get Credit Non-Qualifying intra-bucket correlation function."""
        bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 1
        corr = CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS.get(bucket_num, 0.41)
        
        def corr_func(q1: str, q2: str) -> float:
            if q1 == q2:
                return 1.0
            return corr
        
        return corr_func
    
    def _get_equity_correlation_function(
        self,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> Callable[[str, str], float]:
        """Get Equity intra-bucket correlation function."""
        bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 1
        corr = EQUITY_INTRA_BUCKET_CORRELATIONS.get(bucket_num, 0.15)
        
        def corr_func(q1: str, q2: str) -> float:
            if q1 == q2:
                return 1.0
            return corr
        
        return corr_func
    
    def _get_commodity_correlation_function(
        self,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> Callable[[str, str], float]:
        """Get Commodity intra-bucket correlation function."""
        bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 1
        corr = COMMODITY_INTRA_BUCKET_CORRELATIONS.get(bucket_num, 0.16)
        
        def corr_func(q1: str, q2: str) -> float:
            if q1 == q2:
                return 1.0
            return corr
        
        return corr_func
    
    def _get_fx_correlation_function(
        self,
        margin_type: MarginType,
    ) -> Callable[[str, str], float]:
        """Get FX correlation function."""
        # FX correlation between different currency pairs
        # FX_CORRELATIONS is a numpy array with rows/cols by volatility group
        base_corr = float(FX_CORRELATIONS[0, 0])  # Use regular volatility correlation
        
        def corr_func(q1: str, q2: str) -> float:
            if q1 == q2:
                return 1.0
            return base_corr
        
        return corr_func
