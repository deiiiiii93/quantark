"""
Product Class Aggregator for SIMM.

This module implements cross-risk-class aggregation within a product class.

SIMM_product = sqrt(Σ_r IM_r² + Σ_r Σ_{s≠r} ψ_rs × IM_r × IM_s)

Where:
- IM_r is the total margin for risk class r (Delta + Vega + Curvature + BaseCorr)
- ψ_rs is the inter-risk-class correlation
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

from simm.taxonomy import RiskClass, ProductClass, MarginType
from simm.engines.aggregation.risk_class_aggregator import RiskClassResult
from simm.calibration import get_inter_risk_class_correlation
from simm.calibration.cross_risk import INTER_RISK_CLASS_CORRELATIONS


# Mapping from risk class to product class
RISK_CLASS_TO_PRODUCT_CLASS = {
    RiskClass.INTEREST_RATE: ProductClass.RATES_FX,
    RiskClass.FX: ProductClass.RATES_FX,
    RiskClass.CREDIT_QUALIFYING: ProductClass.CREDIT,
    RiskClass.CREDIT_NON_QUALIFYING: ProductClass.CREDIT,
    RiskClass.EQUITY: ProductClass.EQUITY,
    RiskClass.COMMODITY: ProductClass.COMMODITY,
}


@dataclass
class ProductClassResult:
    """Result of product class aggregation.
    
    Attributes:
        product_class: The product class.
        margin: The aggregated SIMM for this product class.
        risk_class_margins: Dict mapping risk class to total margin.
        by_margin_type: Breakdown by risk class and margin type.
    """
    product_class: ProductClass
    margin: float
    risk_class_margins: Dict[RiskClass, float] = field(default_factory=dict)
    by_margin_type: Dict[RiskClass, Dict[MarginType, float]] = field(default_factory=dict)


class ProductClassAggregator:
    """Aggregator for product class SIMM calculation.
    
    Implements cross-risk-class aggregation using inter-risk-class correlations.
    """
    
    def aggregate(
        self,
        risk_class_results: Dict[RiskClass, Dict[MarginType, RiskClassResult]],
        product_class: ProductClass,
    ) -> ProductClassResult:
        """Aggregate margins across risk classes within a product class.
        
        SIMM_product = sqrt(Σ_r IM_r² + Σ_r Σ_{s≠r} ψ_rs × IM_r × IM_s)
        
        Args:
            risk_class_results: Dict mapping risk class -> margin type -> RiskClassResult.
            product_class: The product class.
            
        Returns:
            ProductClassResult with aggregated SIMM.
        """
        if not risk_class_results:
            return ProductClassResult(
                product_class=product_class,
                margin=0.0,
            )
        
        # Calculate total margin per risk class (sum of all margin types)
        risk_class_margins: Dict[RiskClass, float] = {}
        by_margin_type: Dict[RiskClass, Dict[MarginType, float]] = {}
        
        for rc, mt_results in risk_class_results.items():
            by_margin_type[rc] = {}
            total_rc_margin = 0.0
            
            for mt, result in mt_results.items():
                by_margin_type[rc][mt] = result.margin
                total_rc_margin += result.margin
            
            risk_class_margins[rc] = total_rc_margin
        
        # Filter to risk classes in this product class
        relevant_rc = [rc for rc in risk_class_margins.keys() 
                      if RISK_CLASS_TO_PRODUCT_CLASS.get(rc) == product_class]
        
        if not relevant_rc:
            return ProductClassResult(
                product_class=product_class,
                margin=0.0,
                risk_class_margins=risk_class_margins,
                by_margin_type=by_margin_type,
            )
        
        # Diagonal terms: Σ IM_r²
        sum_sq = sum(risk_class_margins[rc]**2 for rc in relevant_rc)
        
        # Cross terms: Σ_r Σ_{s≠r} ψ_rs × IM_r × IM_s
        cross_sum = 0.0
        n = len(relevant_rc)
        
        for i in range(n):
            for j in range(i + 1, n):
                rc_i, rc_j = relevant_rc[i], relevant_rc[j]
                
                # Get inter-risk-class correlation
                psi = self._get_inter_risk_class_correlation(rc_i, rc_j)
                
                im_i = risk_class_margins[rc_i]
                im_j = risk_class_margins[rc_j]
                
                cross_sum += 2 * psi * im_i * im_j
        
        # SIMM_product
        margin = math.sqrt(max(0, sum_sq + cross_sum))
        
        return ProductClassResult(
            product_class=product_class,
            margin=margin,
            risk_class_margins={rc: risk_class_margins[rc] for rc in relevant_rc},
            by_margin_type={rc: by_margin_type.get(rc, {}) for rc in relevant_rc},
        )
    
    def _get_inter_risk_class_correlation(
        self,
        rc1: RiskClass,
        rc2: RiskClass,
    ) -> float:
        """Get inter-risk-class correlation ψ.
        
        Args:
            rc1: First risk class.
            rc2: Second risk class.
            
        Returns:
            The inter-risk-class correlation.
        """
        if rc1 == rc2:
            return 1.0
        
        # Map risk class to index in correlation matrix
        from simm.calibration.cross_risk import INTER_RISK_CLASS_CORRELATION_LABELS
        
        key1 = rc1.value
        key2 = rc2.value
        
        try:
            idx1 = INTER_RISK_CLASS_CORRELATION_LABELS.index(key1)
            idx2 = INTER_RISK_CLASS_CORRELATION_LABELS.index(key2)
            psi = float(INTER_RISK_CLASS_CORRELATIONS[idx1, idx2])
        except (ValueError, IndexError):
            # Default correlation if not found
            psi = 0.27
        
        return psi
    
    def aggregate_all_product_classes(
        self,
        risk_class_results: Dict[RiskClass, Dict[MarginType, RiskClassResult]],
    ) -> Dict[ProductClass, ProductClassResult]:
        """Aggregate margins for all product classes.
        
        Args:
            risk_class_results: Dict mapping risk class -> margin type -> RiskClassResult.
            
        Returns:
            Dict mapping product class to ProductClassResult.
        """
        results = {}
        
        for pc in ProductClass:
            results[pc] = self.aggregate(risk_class_results, pc)
        
        return results
