"""
Concentration Risk Calculator for SIMM.

This module implements concentration risk factor calculations per SIMM paragraphs 7-8.
Concentration risk factors (CR) are applied to sensitivities to account for
concentration in large positions.

Formulas:
- IR: CR_b = max(1, sqrt(|Σs|/T))
- Credit/Equity/Commodity/FX: CR_k = max(1, sqrt(|s_k|/T)) or grouped by issuer
- Vega: VCR similar to delta CR
- g_bc = min(CR_b, CR_c) / max(CR_b, CR_c) for cross-bucket aggregation
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

from quantark.simm.taxonomy import RiskClass, MarginType
from quantark.simm.sensitivity import AnySensitivity
from quantark.simm.calibration import (
    get_concentration_threshold,
    IR_DELTA_CONCENTRATION_THRESHOLDS,
    IR_VEGA_CONCENTRATION_THRESHOLDS,
    CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
    CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
    EQUITY_DELTA_CONCENTRATION_THRESHOLDS,
    EQUITY_VEGA_CONCENTRATION_THRESHOLDS,
    COMMODITY_DELTA_CONCENTRATION_THRESHOLDS,
    COMMODITY_VEGA_CONCENTRATION_THRESHOLDS,
    FX_DELTA_CONCENTRATION_THRESHOLDS,
    FX_VEGA_CONCENTRATION_THRESHOLDS,
)
from quantark.simm.taxonomy import get_currency_volatility, CurrencyVolatility


@dataclass
class ConcentrationResult:
    """Result of concentration risk factor calculation.
    
    Attributes:
        risk_class: The risk class.
        margin_type: Delta or Vega.
        bucket: The bucket identifier.
        cr_values: Dict mapping risk factor qualifier to CR value.
        bucket_cr: The bucket-level CR (for IR).
    """
    risk_class: RiskClass
    margin_type: MarginType
    bucket: Union[str, int]
    cr_values: Dict[str, float] = field(default_factory=dict)
    bucket_cr: float = 1.0


class ConcentrationCalculator:
    """Calculator for concentration risk factors.
    
    Implements concentration risk calculation per SIMM spec paragraphs 7-8.
    Different risk classes have different concentration formulas.
    """
    
    def calculate(
        self,
        sensitivities: List[AnySensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Calculate concentration risk factors for sensitivities in a bucket.
        
        Args:
            sensitivities: List of sensitivities in the bucket.
            risk_class: The risk class.
            margin_type: Delta or Vega.
            bucket: The bucket identifier.
            
        Returns:
            ConcentrationResult with CR values for each risk factor.
        """
        if not sensitivities:
            return ConcentrationResult(
                risk_class=risk_class,
                margin_type=margin_type,
                bucket=bucket,
            )
        
        if risk_class == RiskClass.INTEREST_RATE:
            return self._calculate_ir_concentration(sensitivities, margin_type, bucket)
        elif risk_class in (RiskClass.CREDIT_QUALIFYING, RiskClass.CREDIT_NON_QUALIFYING):
            return self._calculate_credit_concentration(sensitivities, risk_class, margin_type, bucket)
        elif risk_class == RiskClass.EQUITY:
            return self._calculate_equity_concentration(sensitivities, margin_type, bucket)
        elif risk_class == RiskClass.COMMODITY:
            return self._calculate_commodity_concentration(sensitivities, margin_type, bucket)
        elif risk_class == RiskClass.FX:
            return self._calculate_fx_concentration(sensitivities, margin_type, bucket)
        else:
            # Default: CR = 1 for all
            return ConcentrationResult(
                risk_class=risk_class,
                margin_type=margin_type,
                bucket=bucket,
                cr_values={s.qualifier: 1.0 for s in sensitivities},
                bucket_cr=1.0,
            )
    
    def _calculate_ir_concentration(
        self,
        sensitivities: List[AnySensitivity],
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Calculate IR concentration risk.
        
        For IR, concentration is at the currency (bucket) level.
        CR_b = max(1, sqrt(|Σ s_k,i| / T_b))
        
        Note: Thresholds in calibration are in USD millions, convert to base units.
        """
        currency = str(bucket).upper()
        
        # Get threshold based on currency volatility
        vol_class = get_currency_volatility(currency)
        
        if margin_type == MarginType.DELTA:
            thresholds = IR_DELTA_CONCENTRATION_THRESHOLDS
        else:  # Vega
            thresholds = IR_VEGA_CONCENTRATION_THRESHOLDS
        
        # Map volatility class to threshold key
        # IR thresholds use specific keys for different currency groups
        if vol_class == CurrencyVolatility.HIGH:
            threshold_mm = thresholds.get("high", 30)
        elif currency in ("USD", "EUR", "GBP"):
            threshold_mm = thresholds.get("regular_well_traded", 330)
        elif currency == "JPY":
            threshold_mm = thresholds.get("low", 61)
        else:
            threshold_mm = thresholds.get("regular_less_traded", 130)
        
        # Convert from millions to base currency units
        threshold = threshold_mm * 1_000_000
        
        # Sum all sensitivities in the currency bucket
        total_sens = sum(s.amount for s in sensitivities)
        
        # Calculate bucket-level CR
        bucket_cr = max(1.0, math.sqrt(abs(total_sens) / threshold))
        
        # For IR, all risk factors in the bucket share the same CR
        cr_values = {s.qualifier: bucket_cr for s in sensitivities}
        
        return ConcentrationResult(
            risk_class=RiskClass.INTEREST_RATE,
            margin_type=margin_type,
            bucket=bucket,
            cr_values=cr_values,
            bucket_cr=bucket_cr,
        )
    
    def _calculate_credit_concentration(
        self,
        sensitivities: List[AnySensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Calculate Credit concentration risk.
        
        For Credit, concentration is per issuer (grouped by issuer/seniority).
        CR_k = max(1, sqrt(|Σ_j s_j| / T_b)) where j sums over same issuer.
        
        Note: Thresholds in calibration are in USD millions, convert to base units.
        """
        # Get threshold for this bucket (in USD millions, convert to base units)
        if risk_class == RiskClass.CREDIT_QUALIFYING:
            if margin_type == MarginType.DELTA:
                thresholds = CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS
                threshold_mm = thresholds.get(bucket, thresholds.get(1, 1e6))
            else:
                threshold_mm = CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD
        else:  # Non-Qualifying
            if margin_type == MarginType.DELTA:
                thresholds = CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS
                threshold_mm = thresholds.get(bucket, thresholds.get(1, 1e6))
            else:
                threshold_mm = CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD
        
        # Convert from millions to base currency units
        threshold = threshold_mm * 1_000_000
        
        # Group by issuer (qualifier)
        issuer_sums: Dict[str, float] = {}
        for s in sensitivities:
            issuer = s.qualifier
            issuer_sums[issuer] = issuer_sums.get(issuer, 0.0) + s.amount
        
        # Calculate CR per issuer
        cr_values = {}
        for issuer, total in issuer_sums.items():
            cr_values[issuer] = max(1.0, math.sqrt(abs(total) / threshold))
        
        return ConcentrationResult(
            risk_class=risk_class,
            margin_type=margin_type,
            bucket=bucket,
            cr_values=cr_values,
            bucket_cr=1.0,  # Credit doesn't use bucket-level CR in same way as IR
        )
    
    def _calculate_equity_concentration(
        self,
        sensitivities: List[AnySensitivity],
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Calculate Equity concentration risk.
        
        For Equity, CR is per individual risk factor.
        CR_k = max(1, sqrt(|s_k| / T_b))
        
        Note: Thresholds in calibration are in USD millions, convert to base units.
        """
        if margin_type == MarginType.DELTA:
            thresholds = EQUITY_DELTA_CONCENTRATION_THRESHOLDS
            threshold_mm = thresholds.get(bucket, thresholds.get(1, 1e6))
        else:
            thresholds = EQUITY_VEGA_CONCENTRATION_THRESHOLDS
            threshold_mm = thresholds.get(bucket, thresholds.get(1, 1e6))
        
        # Convert from millions to base currency units
        threshold = threshold_mm * 1_000_000
        
        # Aggregate by qualifier first
        qualifier_sums: Dict[str, float] = {}
        for s in sensitivities:
            qualifier_sums[s.qualifier] = qualifier_sums.get(s.qualifier, 0.0) + s.amount
        
        # CR per risk factor
        cr_values = {}
        for qualifier, amount in qualifier_sums.items():
            cr_values[qualifier] = max(1.0, math.sqrt(abs(amount) / threshold))
        
        return ConcentrationResult(
            risk_class=RiskClass.EQUITY,
            margin_type=margin_type,
            bucket=bucket,
            cr_values=cr_values,
            bucket_cr=1.0,
        )
    
    def _calculate_commodity_concentration(
        self,
        sensitivities: List[AnySensitivity],
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Calculate Commodity concentration risk.
        
        Similar to Equity, CR is per risk factor.
        
        Note: Thresholds in calibration are in USD millions, convert to base units.
        """
        if margin_type == MarginType.DELTA:
            thresholds = COMMODITY_DELTA_CONCENTRATION_THRESHOLDS
            threshold_mm = thresholds.get(bucket, thresholds.get(1, 1e6))
        else:
            thresholds = COMMODITY_VEGA_CONCENTRATION_THRESHOLDS
            threshold_mm = thresholds.get(bucket, thresholds.get(1, 1e6))
        
        # Convert from millions to base currency units
        threshold = threshold_mm * 1_000_000
        
        # Aggregate by qualifier
        qualifier_sums: Dict[str, float] = {}
        for s in sensitivities:
            qualifier_sums[s.qualifier] = qualifier_sums.get(s.qualifier, 0.0) + s.amount
        
        cr_values = {}
        for qualifier, amount in qualifier_sums.items():
            cr_values[qualifier] = max(1.0, math.sqrt(abs(amount) / threshold))
        
        return ConcentrationResult(
            risk_class=RiskClass.COMMODITY,
            margin_type=margin_type,
            bucket=bucket,
            cr_values=cr_values,
            bucket_cr=1.0,
        )
    
    def _calculate_fx_concentration(
        self,
        sensitivities: List[AnySensitivity],
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Calculate FX concentration risk.
        
        FX concentration is per currency pair.
        
        Note: Thresholds in calibration are in USD millions, convert to base units.
        """
        if margin_type == MarginType.DELTA:
            # FX delta thresholds depend on currency volatility category
            # For simplicity, use the regular category threshold
            threshold_mm = FX_DELTA_CONCENTRATION_THRESHOLDS.get("regular", 8400)
        else:
            threshold_mm = FX_VEGA_CONCENTRATION_THRESHOLDS.get("regular", 2900)
        
        # Convert from millions to base currency units
        threshold = threshold_mm * 1_000_000
        
        # Aggregate by qualifier (currency pair)
        qualifier_sums: Dict[str, float] = {}
        for s in sensitivities:
            qualifier_sums[s.qualifier] = qualifier_sums.get(s.qualifier, 0.0) + s.amount
        
        cr_values = {}
        for qualifier, amount in qualifier_sums.items():
            cr_values[qualifier] = max(1.0, math.sqrt(abs(amount) / threshold))
        
        return ConcentrationResult(
            risk_class=RiskClass.FX,
            margin_type=margin_type,
            bucket=bucket,
            cr_values=cr_values,
            bucket_cr=1.0,
        )
    
    @staticmethod
    def calculate_g_bc(cr_b: float, cr_c: float) -> float:
        """Calculate g_bc factor for cross-bucket aggregation.
        
        g_bc = min(CR_b, CR_c) / max(CR_b, CR_c)
        
        This factor is used in IR aggregation to scale cross-currency correlations.
        
        Args:
            cr_b: Concentration risk factor for bucket b.
            cr_c: Concentration risk factor for bucket c.
            
        Returns:
            The g_bc factor (between 0 and 1).
        """
        if cr_b <= 0 or cr_c <= 0:
            return 1.0
        return min(cr_b, cr_c) / max(cr_b, cr_c)
