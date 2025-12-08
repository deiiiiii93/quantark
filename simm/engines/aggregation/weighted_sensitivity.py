"""
Weighted Sensitivity Calculator for SIMM.

This module implements the weighted sensitivity calculation:
WS = RW × s × CR

Where:
- RW is the risk weight from calibration
- s is the raw sensitivity amount
- CR is the concentration risk factor
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from simm.taxonomy import RiskClass, MarginType
from simm.sensitivity import AnySensitivity, IRDeltaSensitivity, IRVegaSensitivity
from simm.calibration import get_risk_weight, get_vrw
from simm.calibration.ir import IR_RISK_WEIGHTS, IR_VRW, IR_TENOR_LABELS
from simm.calibration.credit_qualifying import CREDIT_QUALIFYING_RISK_WEIGHTS, CREDIT_QUALIFYING_VRW
from simm.calibration.credit_non_qualifying import CREDIT_NON_QUALIFYING_RISK_WEIGHTS, CREDIT_NON_QUALIFYING_VRW
from simm.calibration.equity import EQUITY_RISK_WEIGHTS, EQUITY_VRW
from simm.calibration.commodity import COMMODITY_RISK_WEIGHTS, COMMODITY_VRW
from simm.calibration.fx import FX_RISK_WEIGHTS, FX_VRW
from simm.taxonomy import get_currency_volatility


@dataclass
class WeightedSensitivity:
    """A sensitivity with its weighted value calculated.
    
    Attributes:
        original: The original sensitivity.
        qualifier: Risk factor identifier.
        bucket: The bucket.
        risk_weight: The applied risk weight.
        concentration_factor: The CR factor applied.
        weighted_value: The final WS = RW × s × CR.
    """
    original: AnySensitivity
    qualifier: str
    bucket: Union[str, int]
    risk_weight: float
    concentration_factor: float
    weighted_value: float


class WeightedSensitivityCalculator:
    """Calculator for weighted sensitivities.
    
    Calculates WS = RW × s × CR for each sensitivity.
    """
    
    def calculate(
        self,
        sensitivities: List[AnySensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
        cr_values: Dict[str, float],
    ) -> List[WeightedSensitivity]:
        """Calculate weighted sensitivities.
        
        Args:
            sensitivities: List of raw sensitivities.
            risk_class: The risk class.
            margin_type: Delta or Vega.
            bucket: The bucket identifier.
            cr_values: Dict mapping qualifier to concentration risk factor.
            
        Returns:
            List of WeightedSensitivity objects.
        """
        results = []
        
        for sens in sensitivities:
            # Get risk weight
            rw = self._get_risk_weight(sens, risk_class, margin_type, bucket)
            
            # Get concentration factor (default to 1.0 if not found)
            cr = cr_values.get(sens.qualifier, 1.0)
            
            # Special case: IR cross-currency basis does not get CR scaling
            if self._is_xccy_basis(sens):
                cr = 1.0
            
            # Calculate weighted sensitivity: WS = RW × s × CR
            ws = rw * sens.amount * cr
            
            results.append(WeightedSensitivity(
                original=sens,
                qualifier=sens.qualifier,
                bucket=bucket,
                risk_weight=rw,
                concentration_factor=cr,
                weighted_value=ws,
            ))
        
        return results
    
    def _get_risk_weight(
        self,
        sens: AnySensitivity,
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> float:
        """Get the risk weight for a sensitivity.
        
        Args:
            sens: The sensitivity.
            risk_class: The risk class.
            margin_type: Delta or Vega.
            bucket: The bucket.
            
        Returns:
            The risk weight.
        """
        if margin_type == MarginType.VEGA:
            return self._get_vega_risk_weight(sens, risk_class, bucket)
        else:
            return self._get_delta_risk_weight(sens, risk_class, bucket)
    
    def _get_delta_risk_weight(
        self,
        sens: AnySensitivity,
        risk_class: RiskClass,
        bucket: Union[str, int],
    ) -> float:
        """Get delta risk weight."""
        if risk_class == RiskClass.INTEREST_RATE:
            return self._get_ir_delta_risk_weight(sens)
        elif risk_class == RiskClass.CREDIT_QUALIFYING:
            return self._get_credit_q_delta_risk_weight(bucket)
        elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
            return self._get_credit_nq_delta_risk_weight(bucket)
        elif risk_class == RiskClass.EQUITY:
            return self._get_equity_delta_risk_weight(bucket)
        elif risk_class == RiskClass.COMMODITY:
            return self._get_commodity_delta_risk_weight(bucket)
        elif risk_class == RiskClass.FX:
            return self._get_fx_delta_risk_weight(sens)
        else:
            return 1.0
    
    def _get_ir_delta_risk_weight(self, sens: AnySensitivity) -> float:
        """Get IR delta risk weight based on currency volatility and tenor."""
        if not isinstance(sens, IRDeltaSensitivity):
            # Default for non-IR delta sensitivities
            return 1.0
        
        currency = sens.currency
        tenor = sens.tenor
        
        # Get volatility class
        vol_class = get_currency_volatility(currency)
        vol_key = vol_class.value.lower()  # "regular", "low", "high"
        
        # Map tenor to label
        tenor_label = self._tenor_to_label(tenor)
        
        # IR_RISK_WEIGHTS uses tuple keys like ("5yr", "regular")
        # Risk weights are in basis points, convert to decimal (divide by 10000)
        rw_bp = IR_RISK_WEIGHTS.get((tenor_label, vol_key))
        if rw_bp is None:
            # Try with "regular" as fallback
            rw_bp = IR_RISK_WEIGHTS.get((tenor_label, "regular"), 66)  # Default to 1yr regular
        
        # Convert from basis points to decimal
        return float(rw_bp) / 10000.0
    
    def _tenor_to_label(self, tenor: float) -> str:
        """Convert tenor in years to label."""
        tenor_map = {
            0.0384: "2w",
            0.0833: "1m",
            0.25: "3m",
            0.5: "6m",
            1.0: "1y",
            2.0: "2y",
            3.0: "3y",
            5.0: "5y",
            10.0: "10y",
            15.0: "15y",
            20.0: "20y",
            30.0: "30y",
        }
        # Find closest tenor
        closest = min(tenor_map.keys(), key=lambda t: abs(t - tenor))
        return tenor_map[closest]
    
    def _get_credit_q_delta_risk_weight(self, bucket: Union[str, int]) -> float:
        """Get Credit Qualifying delta risk weight.
        
        Risk weights in calibration are in percentage, convert to decimal.
        """
        bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 0
        rw_pct = CREDIT_QUALIFYING_RISK_WEIGHTS.get(bucket_num, CREDIT_QUALIFYING_RISK_WEIGHTS.get(1, 75))
        return rw_pct / 100.0
    
    def _get_credit_nq_delta_risk_weight(self, bucket: Union[str, int]) -> float:
        """Get Credit Non-Qualifying delta risk weight.
        
        Risk weights in calibration are in percentage, convert to decimal.
        """
        bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 0
        rw_pct = CREDIT_NON_QUALIFYING_RISK_WEIGHTS.get(bucket_num, CREDIT_NON_QUALIFYING_RISK_WEIGHTS.get(1, 280))
        return rw_pct / 100.0
    
    def _get_equity_delta_risk_weight(self, bucket: Union[str, int]) -> float:
        """Get Equity delta risk weight.
        
        Risk weights in calibration are in percentage, convert to decimal.
        """
        bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 0
        rw_pct = EQUITY_RISK_WEIGHTS.get(bucket_num, EQUITY_RISK_WEIGHTS.get(1, 26))
        return rw_pct / 100.0
    
    def _get_commodity_delta_risk_weight(self, bucket: Union[str, int]) -> float:
        """Get Commodity delta risk weight.
        
        Risk weights in calibration are in percentage, convert to decimal.
        """
        bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 0
        rw_pct = COMMODITY_RISK_WEIGHTS.get(bucket_num, COMMODITY_RISK_WEIGHTS.get(1, 19))
        return rw_pct / 100.0
    
    def _get_fx_delta_risk_weight(self, sens: AnySensitivity) -> float:
        """Get FX delta risk weight.
        
        Risk weights in calibration are in percentage, convert to decimal.
        """
        # FX risk weight from calibration (regular/regular = index [0,0])
        # FX_RISK_WEIGHTS is a numpy array with rows/cols by volatility group
        rw_pct = float(FX_RISK_WEIGHTS[0, 0])  # Use regular volatility weight
        return rw_pct / 100.0
    
    def _get_vega_risk_weight(
        self,
        sens: AnySensitivity,
        risk_class: RiskClass,
        bucket: Union[str, int],
    ) -> float:
        """Get vega risk weight (VRW)."""
        if risk_class == RiskClass.INTEREST_RATE:
            return IR_VRW
        elif risk_class == RiskClass.CREDIT_QUALIFYING:
            return CREDIT_QUALIFYING_VRW
        elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
            return CREDIT_NON_QUALIFYING_VRW
        elif risk_class == RiskClass.EQUITY:
            bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 1
            return EQUITY_VRW.get(bucket_num, EQUITY_VRW.get(1, 0.21))
        elif risk_class == RiskClass.COMMODITY:
            bucket_num = int(bucket) if isinstance(bucket, int) or (isinstance(bucket, str) and bucket.isdigit()) else 1
            return COMMODITY_VRW.get(bucket_num, COMMODITY_VRW.get(1, 0.36))
        elif risk_class == RiskClass.FX:
            return FX_VRW
        else:
            return 1.0
    
    def _is_xccy_basis(self, sens: AnySensitivity) -> bool:
        """Check if sensitivity is IR cross-currency basis.
        
        Cross-currency basis sensitivities do not get CR scaling.
        """
        # Check if sensitivity has sub_curve attribute indicating xccy basis
        if hasattr(sens, 'sub_curve'):
            sub_curve = getattr(sens, 'sub_curve', None)
            if sub_curve and 'xccy' in str(sub_curve).lower():
                return True
        return False
