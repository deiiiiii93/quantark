"""
Risk Class Aggregator for SIMM.

This module implements cross-bucket aggregation within a risk class.

Delta Margin = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc × S_b × S_c) + K_residual

Where:
- K_b is the bucket-level margin
- γ_bc is the inter-bucket correlation
- S_b = max(min(Σ WS_k, K_b), -K_b) is the capped bucket sum
- K_residual is added without diversification

For IR, the formula includes g_bc factor:
Delta Margin = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc × g_bc × S_b × S_c)

For Curvature:
- Uses ρ² and γ² (squared correlations)
- Includes θ and λ factors
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable, Any

from simm.taxonomy import RiskClass, MarginType
from simm.engines.aggregation.bucket_aggregator import BucketResult
from simm.engines.aggregation.concentration import ConcentrationCalculator
from simm.calibration import get_inter_bucket_correlation
from simm.calibration.ir import IR_INTER_CURRENCY_CORRELATION, IR_HVR
from simm.calibration.credit_qualifying import CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS
from simm.calibration.credit_non_qualifying import CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION
from simm.calibration.equity import EQUITY_INTER_BUCKET_CORRELATIONS
from simm.calibration.commodity import COMMODITY_INTER_BUCKET_CORRELATIONS


# Inverse of standard normal CDF at 99.5% (Φ^-1(0.995))
PHI_INV_995 = 2.5758293035489


@dataclass
class RiskClassResult:
    """Result of risk class aggregation.
    
    Attributes:
        risk_class: The risk class.
        margin_type: Delta, Vega, or Curvature.
        margin: The aggregated margin for this risk class and margin type.
        bucket_results: Dict mapping bucket to BucketResult.
        residual_margin: The residual bucket margin (added without diversification).
    """
    risk_class: RiskClass
    margin_type: MarginType
    margin: float
    bucket_results: Dict[Union[str, int], BucketResult] = field(default_factory=dict)
    residual_margin: float = 0.0


class RiskClassAggregator:
    """Aggregator for risk class margin calculation.
    
    Implements the cross-bucket aggregation formulas from SIMM spec.
    """
    
    def aggregate_delta(
        self,
        bucket_results: List[BucketResult],
        risk_class: RiskClass,
        bucket_cr_values: Optional[Dict[Union[str, int], float]] = None,
    ) -> RiskClassResult:
        """Aggregate delta margin across buckets.
        
        DeltaMargin = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc × S_b × S_c) + K_residual
        
        For IR, includes g_bc factor.
        
        Args:
            bucket_results: List of bucket results.
            risk_class: The risk class.
            bucket_cr_values: Dict mapping bucket to bucket-level CR (for IR g_bc).
            
        Returns:
            RiskClassResult with aggregated delta margin.
        """
        if not bucket_results:
            return RiskClassResult(
                risk_class=risk_class,
                margin_type=MarginType.DELTA,
                margin=0.0,
            )
        
        # Separate residual and non-residual buckets
        residual = [b for b in bucket_results if b.is_residual]
        non_residual = [b for b in bucket_results if not b.is_residual]
        
        if not non_residual:
            # Only residual bucket
            residual_margin = sum(b.k_b for b in residual)
            return RiskClassResult(
                risk_class=risk_class,
                margin_type=MarginType.DELTA,
                margin=residual_margin,
                bucket_results={b.bucket: b for b in bucket_results},
                residual_margin=residual_margin,
            )
        
        # Get inter-bucket correlation function
        gamma_func = self._get_inter_bucket_correlation_function(risk_class, MarginType.DELTA)
        
        # Calculate S_b for each bucket: S_b = max(min(Σ WS_k, K_b), -K_b)
        s_values = {}
        for br in non_residual:
            s_values[br.bucket] = max(min(br.ws_sum, br.k_b), -br.k_b)
        
        buckets = [br.bucket for br in non_residual]
        n = len(buckets)
        
        # Diagonal terms: Σ K_b²
        sum_k_sq = sum(br.k_b ** 2 for br in non_residual)
        
        # Cross terms with gamma
        cross_sum = 0.0
        bucket_cr = bucket_cr_values or {}
        
        for i in range(n):
            for j in range(i + 1, n):
                b_i, b_j = buckets[i], buckets[j]
                
                gamma = gamma_func(b_i, b_j)
                
                # For IR, apply g_bc factor
                if risk_class == RiskClass.INTEREST_RATE and bucket_cr:
                    cr_i = bucket_cr.get(b_i, 1.0)
                    cr_j = bucket_cr.get(b_j, 1.0)
                    g_bc = ConcentrationCalculator.calculate_g_bc(cr_i, cr_j)
                    gamma *= g_bc
                
                cross_sum += 2 * gamma * s_values[b_i] * s_values[b_j]
        
        # DeltaMargin (excluding residual)
        margin = math.sqrt(max(0, sum_k_sq + cross_sum))
        
        # Add residual bucket margin
        residual_margin = sum(b.k_b for b in residual)
        margin += residual_margin
        
        return RiskClassResult(
            risk_class=risk_class,
            margin_type=MarginType.DELTA,
            margin=margin,
            bucket_results={b.bucket: b for b in bucket_results},
            residual_margin=residual_margin,
        )
    
    def aggregate_vega(
        self,
        bucket_results: List[BucketResult],
        risk_class: RiskClass,
        bucket_cr_values: Optional[Dict[Union[str, int], float]] = None,
    ) -> RiskClassResult:
        """Aggregate vega margin across buckets.
        
        Uses same formula as delta.
        
        Args:
            bucket_results: List of bucket results.
            risk_class: The risk class.
            bucket_cr_values: Dict mapping bucket to bucket-level CR (for IR g_bc).
            
        Returns:
            RiskClassResult with aggregated vega margin.
        """
        result = self.aggregate_delta(bucket_results, risk_class, bucket_cr_values)
        result.margin_type = MarginType.VEGA
        return result
    
    def aggregate_curvature(
        self,
        bucket_cvr_values: Dict[Union[str, int], Dict[str, float]],
        risk_class: RiskClass,
    ) -> RiskClassResult:
        """Aggregate curvature margin across buckets.
        
        Uses squared correlations (ρ² and γ²) and θ/λ factors.
        
        CurvatureMargin = max(Σ CVR + λ × sqrt(Σ K_b² + Σ γ_bc² × S_b × S_c), 0)
        
        Where:
        θ = min(Σ CVR / Σ|CVR|, 0)
        λ = (Φ^-1(99.5%)² - 1)(1 + θ) - θ
        
        Args:
            bucket_cvr_values: Dict mapping bucket -> {qualifier: CVR}.
            risk_class: The risk class.
            
        Returns:
            RiskClassResult with aggregated curvature margin.
        """
        if not bucket_cvr_values:
            return RiskClassResult(
                risk_class=risk_class,
                margin_type=MarginType.CURVATURE,
                margin=0.0,
            )
        
        # Separate residual and non-residual
        residual_cvrs = bucket_cvr_values.get("Residual", bucket_cvr_values.get(-1, {}))
        non_residual = {b: cvrs for b, cvrs in bucket_cvr_values.items() 
                       if b not in ("Residual", -1)}
        
        if not non_residual:
            # Only residual
            residual_margin = sum(abs(cvr) for cvr in residual_cvrs.values())
            return RiskClassResult(
                risk_class=risk_class,
                margin_type=MarginType.CURVATURE,
                margin=residual_margin,
                residual_margin=residual_margin,
            )
        
        # Get correlation functions (use squared correlations)
        intra_corr_func = self._get_intra_bucket_corr_squared(risk_class)
        gamma_func = self._get_inter_bucket_correlation_function(risk_class, MarginType.CURVATURE)
        
        # Calculate K_b for each bucket using ρ²
        bucket_k = {}
        for bucket, cvrs in non_residual.items():
            qualifiers = list(cvrs.keys())
            n_q = len(qualifiers)
            
            # Diagonal: Σ CVR²
            sum_sq = sum(cvr**2 for cvr in cvrs.values())
            
            # Cross with ρ²
            cross = 0.0
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    q_i, q_j = qualifiers[i], qualifiers[j]
                    rho_sq = intra_corr_func(bucket, q_i, q_j) ** 2
                    cross += 2 * rho_sq * cvrs[q_i] * cvrs[q_j]
            
            bucket_k[bucket] = math.sqrt(max(0, sum_sq + cross))
        
        # Calculate S_b for curvature
        s_values = {}
        for bucket, cvrs in non_residual.items():
            cvr_sum = sum(cvrs.values())
            k_b = bucket_k[bucket]
            s_values[bucket] = max(min(cvr_sum, k_b), -k_b)
        
        # Calculate total CVR and |CVR|
        total_cvr = sum(sum(cvrs.values()) for cvrs in bucket_cvr_values.values())
        total_abs_cvr = sum(sum(abs(v) for v in cvrs.values()) for cvrs in bucket_cvr_values.values())
        
        # Calculate θ and λ
        theta = min(total_cvr / total_abs_cvr, 0) if total_abs_cvr > 0 else 0
        lambda_val = (PHI_INV_995**2 - 1) * (1 + theta) - theta
        
        # Cross-bucket with γ²
        buckets = list(bucket_k.keys())
        n = len(buckets)
        sum_k_sq = sum(k**2 for k in bucket_k.values())
        
        cross_gamma = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                b_i, b_j = buckets[i], buckets[j]
                gamma_sq = gamma_func(b_i, b_j) ** 2
                cross_gamma += 2 * gamma_sq * s_values[b_i] * s_values[b_j]
        
        # CurvatureMargin (non-residual)
        margin = max(total_cvr + lambda_val * math.sqrt(max(0, sum_k_sq + cross_gamma)), 0)
        
        # Apply HVR^(-2) scaling for IR curvature
        if risk_class == RiskClass.INTEREST_RATE:
            hvr = IR_HVR
            margin *= hvr ** (-2)
        
        # Add residual curvature margin
        residual_margin = sum(abs(cvr) for cvr in residual_cvrs.values())
        margin += residual_margin
        
        return RiskClassResult(
            risk_class=risk_class,
            margin_type=MarginType.CURVATURE,
            margin=margin,
            residual_margin=residual_margin,
        )
    
    def aggregate_base_corr(
        self,
        sensitivities_weighted: Dict[str, float],
    ) -> RiskClassResult:
        """Aggregate base correlation margin (Credit Qualifying only).
        
        BaseCorrMargin uses simple sum with correlation.
        
        Args:
            sensitivities_weighted: Dict mapping index name to weighted sensitivity.
            
        Returns:
            RiskClassResult with base correlation margin.
        """
        if not sensitivities_weighted:
            return RiskClassResult(
                risk_class=RiskClass.CREDIT_QUALIFYING,
                margin_type=MarginType.BASE_CORR,
                margin=0.0,
            )
        
        indices = list(sensitivities_weighted.keys())
        n = len(indices)
        
        # For base corr, CR = 1 always
        # BaseCorrMargin = sqrt(Σ WS² + Σ ρ × WS_i × WS_j)
        sum_sq = sum(ws**2 for ws in sensitivities_weighted.values())
        
        # Inter-index correlation (from calibration)
        from simm.calibration.credit_qualifying import CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION
        rho = CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION
        
        cross = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                ws_i = sensitivities_weighted[indices[i]]
                ws_j = sensitivities_weighted[indices[j]]
                cross += 2 * rho * ws_i * ws_j
        
        margin = math.sqrt(max(0, sum_sq + cross))
        
        return RiskClassResult(
            risk_class=RiskClass.CREDIT_QUALIFYING,
            margin_type=MarginType.BASE_CORR,
            margin=margin,
        )
    
    def _get_inter_bucket_correlation_function(
        self,
        risk_class: RiskClass,
        margin_type: MarginType,
    ) -> Callable[[Union[str, int], Union[str, int]], float]:
        """Get inter-bucket correlation function."""
        if risk_class == RiskClass.INTEREST_RATE:
            gamma = IR_INTER_CURRENCY_CORRELATION
            return lambda b1, b2: gamma if b1 != b2 else 1.0
        
        elif risk_class == RiskClass.CREDIT_QUALIFYING:
            corr_matrix = CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS
            def get_corr(b1, b2):
                if b1 == b2:
                    return 1.0
                b1_int = int(b1) if isinstance(b1, (int, str)) and str(b1).isdigit() else 1
                b2_int = int(b2) if isinstance(b2, (int, str)) and str(b2).isdigit() else 1
                return corr_matrix.get((b1_int, b2_int), corr_matrix.get((b2_int, b1_int), 0.36))
            return get_corr
        
        elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
            gamma = CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION
            return lambda b1, b2: gamma if b1 != b2 else 1.0
        
        elif risk_class == RiskClass.EQUITY:
            corr_matrix = EQUITY_INTER_BUCKET_CORRELATIONS
            def get_corr(b1, b2):
                if b1 == b2:
                    return 1.0
                b1_int = int(b1) if isinstance(b1, (int, str)) and str(b1).isdigit() else 1
                b2_int = int(b2) if isinstance(b2, (int, str)) and str(b2).isdigit() else 1
                return corr_matrix.get((b1_int, b2_int), corr_matrix.get((b2_int, b1_int), 0.15))
            return get_corr
        
        elif risk_class == RiskClass.COMMODITY:
            corr_matrix = COMMODITY_INTER_BUCKET_CORRELATIONS
            def get_corr(b1, b2):
                if b1 == b2:
                    return 1.0
                b1_int = int(b1) if isinstance(b1, (int, str)) and str(b1).isdigit() else 1
                b2_int = int(b2) if isinstance(b2, (int, str)) and str(b2).isdigit() else 1
                return corr_matrix.get((b1_int, b2_int), corr_matrix.get((b2_int, b1_int), 0.0))
            return get_corr
        
        elif risk_class == RiskClass.FX:
            # FX has single bucket
            return lambda b1, b2: 1.0
        
        else:
            return lambda b1, b2: 0.0
    
    def _get_intra_bucket_corr_squared(
        self,
        risk_class: RiskClass,
    ) -> Callable[[Union[str, int], str, str], float]:
        """Get intra-bucket correlation function for curvature (returns non-squared value).
        
        Curvature aggregation squares this value.
        """
        # Return base correlation; caller squares it
        def get_corr(bucket, q1, q2):
            if q1 == q2:
                return 1.0
            # Default correlation
            return 0.5
        
        return get_corr
