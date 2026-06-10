"""
Main SIMM Calculator.

This module provides the SIMMCalculator class that orchestrates
the full SIMM margin calculation pipeline.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from quantark.simm.config import SIMMConfig
from quantark.simm.taxonomy import RiskClass, ProductClass, MarginType
from quantark.simm.sensitivity import SensitivityCollection, AnySensitivity, CurvatureSensitivity
from quantark.simm.engines.aggregation.concentration import ConcentrationCalculator, ConcentrationResult
from quantark.simm.engines.aggregation.weighted_sensitivity import WeightedSensitivityCalculator, WeightedSensitivity
from quantark.simm.engines.aggregation.bucket_aggregator import BucketAggregator, BucketResult
from quantark.simm.engines.aggregation.risk_class_aggregator import RiskClassAggregator, RiskClassResult
from quantark.simm.engines.aggregation.product_class_aggregator import (
    ProductClassAggregator,
    ProductClassResult,
    RISK_CLASS_TO_PRODUCT_CLASS,
)
from quantark.simm.engines.aggregation.addon import AddOnCalculator, AddOnResult
from quantark.simm.calibration import get_risk_weight
from quantark.simm.calibration.credit_qualifying import CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT


@dataclass
class SIMMAggregationResult:
    """Complete SIMM calculation result with full attribution.
    
    Attributes:
        total_margin: Total SIMM margin (sum across product classes + add-ons).
        by_product_class: Margin breakdown by product class.
        by_risk_class: Margin breakdown by risk class.
        by_margin_type: Full breakdown by risk class and margin type.
        addon: Add-on calculation result.
        calculation_currency: Currency of the margin amounts.
        calculation_timestamp: When the calculation was performed.
        simm_version: SIMM version used.
    """
    total_margin: float
    by_product_class: Dict[ProductClass, float] = field(default_factory=dict)
    by_risk_class: Dict[RiskClass, float] = field(default_factory=dict)
    by_margin_type: Dict[RiskClass, Dict[MarginType, float]] = field(default_factory=dict)
    addon: Optional[AddOnResult] = None
    calculation_currency: str = "USD"
    calculation_timestamp: Optional[str] = None
    simm_version: str = "2.6"
    
    # Detailed attribution (optional)
    bucket_details: Dict[RiskClass, Dict[MarginType, Dict[Any, BucketResult]]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "total_margin": self.total_margin,
            "by_product_class": {pc.value: m for pc, m in self.by_product_class.items()},
            "by_risk_class": {rc.value: m for rc, m in self.by_risk_class.items()},
            "by_margin_type": {
                rc.value: {mt.value: m for mt, m in mt_dict.items()}
                for rc, mt_dict in self.by_margin_type.items()
            },
            "calculation_currency": self.calculation_currency,
            "calculation_timestamp": self.calculation_timestamp,
            "simm_version": self.simm_version,
        }
        
        if self.addon:
            result["addon"] = {
                "fixed": self.addon.fixed_addon,
                "factor": self.addon.factor_addon,
                "total": self.addon.total_addon,
            }
        
        return result
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"SIMM Result ({self.simm_version})",
            f"  Total Margin: {self.total_margin:,.2f} {self.calculation_currency}",
            "",
            "  By Product Class:",
        ]
        for pc in ProductClass:
            margin = self.by_product_class.get(pc, 0.0)
            if margin > 0:
                lines.append(f"    {pc.value}: {margin:,.2f}")
        
        lines.append("")
        lines.append("  By Risk Class:")
        for rc in RiskClass:
            margin = self.by_risk_class.get(rc, 0.0)
            if margin > 0:
                lines.append(f"    {rc.value}: {margin:,.2f}")
                mt_dict = self.by_margin_type.get(rc, {})
                for mt, m in mt_dict.items():
                    if m > 0:
                        lines.append(f"      {mt.value}: {m:,.2f}")
        
        return "\n".join(lines)


class SIMMCalculator:
    """Main SIMM calculation engine.
    
    Orchestrates the full SIMM margin calculation pipeline:
    1. Group sensitivities by risk class and bucket
    2. Calculate concentration risk factors
    3. Calculate weighted sensitivities
    4. Aggregate within buckets (K_b)
    5. Aggregate across buckets within risk classes
    6. Aggregate across risk classes within product classes
    7. Sum across product classes
    8. Apply add-ons
    
    Example:
        >>> config = SIMMConfig()
        >>> calculator = SIMMCalculator(config)
        >>> sensitivities = SensitivityCollection()
        >>> # ... add sensitivities ...
        >>> result = calculator.calculate(sensitivities)
        >>> print(result.total_margin)
    """
    
    def __init__(self, config: Optional[SIMMConfig] = None):
        """Initialize the calculator.
        
        Args:
            config: SIMM configuration. Defaults to SIMMConfig().
        """
        self.config = config or SIMMConfig()
        
        # Initialize sub-calculators
        self.concentration_calc = ConcentrationCalculator()
        self.weighted_sens_calc = WeightedSensitivityCalculator()
        self.bucket_agg = BucketAggregator()
        self.risk_class_agg = RiskClassAggregator()
        self.product_class_agg = ProductClassAggregator()
        self.addon_calc = AddOnCalculator(self.config)
    
    def calculate(
        self,
        sensitivities: SensitivityCollection,
        notionals: Optional[Dict[str, float]] = None,
    ) -> SIMMAggregationResult:
        """Calculate total SIMM margin.
        
        Args:
            sensitivities: Collection of sensitivities.
            notionals: Optional dict of notionals for factor-based add-ons.
            
        Returns:
            SIMMAggregationResult with full attribution.
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Initialize result containers
        risk_class_results: Dict[RiskClass, Dict[MarginType, RiskClassResult]] = {}
        bucket_details: Dict[RiskClass, Dict[MarginType, Dict[Any, BucketResult]]] = {}
        
        # Process each risk class
        for risk_class in RiskClass:
            risk_class_results[risk_class] = {}
            bucket_details[risk_class] = {}
            
            # Delta margin
            if self.config.calculate_delta:
                delta_result, delta_buckets = self._calculate_delta_margin(
                    sensitivities, risk_class
                )
                risk_class_results[risk_class][MarginType.DELTA] = delta_result
                bucket_details[risk_class][MarginType.DELTA] = delta_buckets
            
            # Vega margin
            if self.config.calculate_vega:
                vega_result, vega_buckets = self._calculate_vega_margin(
                    sensitivities, risk_class
                )
                risk_class_results[risk_class][MarginType.VEGA] = vega_result
                bucket_details[risk_class][MarginType.VEGA] = vega_buckets
            
            # Curvature margin
            if self.config.calculate_curvature:
                curv_result = self._calculate_curvature_margin(
                    sensitivities, risk_class
                )
                risk_class_results[risk_class][MarginType.CURVATURE] = curv_result
            
            # Base correlation (Credit Qualifying only)
            if self.config.calculate_base_corr and risk_class == RiskClass.CREDIT_QUALIFYING:
                base_corr_result = self._calculate_base_corr_margin(sensitivities)
                risk_class_results[risk_class][MarginType.BASE_CORR] = base_corr_result
        
        # Aggregate by product class
        product_class_results = self.product_class_agg.aggregate_all_product_classes(
            risk_class_results
        )
        
        # Apply multiplicative scales
        by_product_class = {
            pc: result.margin * self._get_multiplier(pc)
            for pc, result in product_class_results.items()
        }
        
        # Calculate add-ons
        addon_result = self.addon_calc.calculate(
            notionals=notionals,
            product_class_margins={pc: r.margin for pc, r in product_class_results.items()},
        )
        
        # Total SIMM = sum of product class margins + add-ons
        total_margin = sum(by_product_class.values()) + addon_result.total_addon
        
        # Build breakdown by risk class
        by_risk_class = {}
        by_margin_type = {}
        for rc, mt_results in risk_class_results.items():
            by_margin_type[rc] = {mt: r.margin for mt, r in mt_results.items()}
            by_risk_class[rc] = sum(r.margin for r in mt_results.values())
        
        return SIMMAggregationResult(
            total_margin=total_margin,
            by_product_class=by_product_class,
            by_risk_class=by_risk_class,
            by_margin_type=by_margin_type,
            addon=addon_result,
            calculation_currency=self.config.calculation_currency,
            calculation_timestamp=timestamp,
            simm_version=str(self.config.version),
            bucket_details=bucket_details if self.config.include_bucket_detail else {},
        )
    
    def _calculate_delta_margin(
        self,
        sensitivities: SensitivityCollection,
        risk_class: RiskClass,
    ) -> tuple[RiskClassResult, Dict[Any, BucketResult]]:
        """Calculate delta margin for a risk class.
        
        Returns:
            Tuple of (RiskClassResult, bucket_details).
        """
        # Get delta sensitivities for this risk class
        delta_sens = sensitivities.by_risk_class_and_margin_type(risk_class, MarginType.DELTA)
        
        if not delta_sens:
            return (
                RiskClassResult(risk_class=risk_class, margin_type=MarginType.DELTA, margin=0.0),
                {},
            )
        
        # Group by bucket
        by_bucket = sensitivities.group_by_bucket(risk_class, MarginType.DELTA)
        
        bucket_results = []
        bucket_cr_values = {}  # For IR g_bc factor
        bucket_details = {}
        
        for bucket, bucket_sens in by_bucket.items():
            # Calculate concentration risk
            cr_result = self.concentration_calc.calculate(
                bucket_sens, risk_class, MarginType.DELTA, bucket
            )
            
            # Store bucket-level CR for IR
            if risk_class == RiskClass.INTEREST_RATE:
                bucket_cr_values[bucket] = cr_result.bucket_cr
            
            # Calculate weighted sensitivities
            ws_list = self.weighted_sens_calc.calculate(
                bucket_sens, risk_class, MarginType.DELTA, bucket, cr_result.cr_values
            )
            
            # Aggregate bucket
            bucket_result = self.bucket_agg.aggregate(
                ws_list, risk_class, MarginType.DELTA, bucket, cr_result.cr_values
            )
            
            bucket_results.append(bucket_result)
            bucket_details[bucket] = bucket_result
        
        # Aggregate across buckets
        result = self.risk_class_agg.aggregate_delta(
            bucket_results, risk_class, bucket_cr_values
        )
        
        return result, bucket_details
    
    def _calculate_vega_margin(
        self,
        sensitivities: SensitivityCollection,
        risk_class: RiskClass,
    ) -> tuple[RiskClassResult, Dict[Any, BucketResult]]:
        """Calculate vega margin for a risk class.
        
        Same logic as delta but for vega sensitivities.
        """
        vega_sens = sensitivities.by_risk_class_and_margin_type(risk_class, MarginType.VEGA)
        
        if not vega_sens:
            return (
                RiskClassResult(risk_class=risk_class, margin_type=MarginType.VEGA, margin=0.0),
                {},
            )
        
        by_bucket = sensitivities.group_by_bucket(risk_class, MarginType.VEGA)
        
        bucket_results = []
        bucket_cr_values = {}
        bucket_details = {}
        
        for bucket, bucket_sens in by_bucket.items():
            cr_result = self.concentration_calc.calculate(
                bucket_sens, risk_class, MarginType.VEGA, bucket
            )
            
            if risk_class == RiskClass.INTEREST_RATE:
                bucket_cr_values[bucket] = cr_result.bucket_cr
            
            ws_list = self.weighted_sens_calc.calculate(
                bucket_sens, risk_class, MarginType.VEGA, bucket, cr_result.cr_values
            )
            
            bucket_result = self.bucket_agg.aggregate(
                ws_list, risk_class, MarginType.VEGA, bucket, cr_result.cr_values
            )
            
            bucket_results.append(bucket_result)
            bucket_details[bucket] = bucket_result
        
        result = self.risk_class_agg.aggregate_vega(
            bucket_results, risk_class, bucket_cr_values
        )
        
        return result, bucket_details
    
    def _calculate_curvature_margin(
        self,
        sensitivities: SensitivityCollection,
        risk_class: RiskClass,
    ) -> RiskClassResult:
        """Calculate curvature margin for a risk class.
        
        Curvature uses CVR values and squared correlations.
        """
        curv_sens = sensitivities.by_risk_class_and_margin_type(risk_class, MarginType.CURVATURE)
        
        if not curv_sens:
            return RiskClassResult(
                risk_class=risk_class, margin_type=MarginType.CURVATURE, margin=0.0
            )
        
        # Group by bucket and calculate CVR
        by_bucket = sensitivities.group_by_bucket(risk_class, MarginType.CURVATURE)
        
        bucket_cvr_values: Dict[Any, Dict[str, float]] = {}
        
        for bucket, bucket_sens in by_bucket.items():
            cvrs = {}
            for sens in bucket_sens:
                if isinstance(sens, CurvatureSensitivity):
                    # CVR = max(CVR_up, CVR_down) - CVR_delta
                    # For simplicity, use the amount as CVR
                    cvr = sens.amount
                    cvrs[sens.qualifier] = cvrs.get(sens.qualifier, 0.0) + cvr
                else:
                    # Fallback for generic sensitivities
                    cvrs[sens.qualifier] = cvrs.get(sens.qualifier, 0.0) + sens.amount
            
            bucket_cvr_values[bucket] = cvrs
        
        return self.risk_class_agg.aggregate_curvature(bucket_cvr_values, risk_class)
    
    def _calculate_base_corr_margin(
        self,
        sensitivities: SensitivityCollection,
    ) -> RiskClassResult:
        """Calculate base correlation margin (Credit Qualifying only)."""
        base_corr_sens = sensitivities.by_risk_class_and_margin_type(
            RiskClass.CREDIT_QUALIFYING, MarginType.BASE_CORR
        )
        
        if not base_corr_sens:
            return RiskClassResult(
                risk_class=RiskClass.CREDIT_QUALIFYING,
                margin_type=MarginType.BASE_CORR,
                margin=0.0,
            )
        
        # Calculate weighted sensitivities (CR = 1 for base corr)
        rw = CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT
        
        ws_by_index = {}
        for sens in base_corr_sens:
            ws = rw * sens.amount  # CR = 1
            ws_by_index[sens.qualifier] = ws_by_index.get(sens.qualifier, 0.0) + ws
        
        return self.risk_class_agg.aggregate_base_corr(ws_by_index)
    
    def _get_multiplier(self, product_class: ProductClass) -> float:
        """Get multiplicative scale for a product class."""
        return self.config.get_product_class_multiplier(product_class.value)
    
    def calculate_from_crif(
        self,
        crif_records: List[Dict[str, Any]],
        notionals: Optional[Dict[str, float]] = None,
    ) -> SIMMAggregationResult:
        """Calculate SIMM from CRIF records.
        
        Convenience method to convert CRIF records to sensitivities
        and calculate SIMM.
        
        Args:
            crif_records: List of CRIF record dicts.
            notionals: Optional notionals for add-ons.
            
        Returns:
            SIMMAggregationResult.
        """
        from quantark.simm.crif.parser import CRIFParser
        
        parser = CRIFParser()
        collection = parser.parse_records(crif_records)
        
        return self.calculate(collection, notionals)
