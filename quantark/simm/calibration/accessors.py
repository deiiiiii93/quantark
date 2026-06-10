"""
Unified Accessor Functions for SIMM Calibration Parameters

This module provides a unified API for accessing calibration parameters across
all risk classes. It abstracts the complexity of per-risk-class parameter storage
and provides convenient accessor functions.

References:
    - ISDA SIMM Methodology, Version 2.6, effective December 2, 2023
"""

from enum import Enum
from typing import Union, Optional, Any, Dict, Tuple

from quantark.simm.calibration.ir import (
    IR_RISK_WEIGHTS,
    IR_TENOR_CORRELATIONS,
    IR_SUB_CURVE_CORRELATION,
    IR_INFLATION_CORRELATION,
    IR_CROSS_CURRENCY_BASIS_CORRELATION,
    IR_INTER_CURRENCY_CORRELATION,
    IR_INFLATION_RISK_WEIGHT,
    IR_CROSS_CURRENCY_BASIS_RISK_WEIGHT,
    IR_HVR,
    IR_VRW,
    IR_DELTA_CONCENTRATION_THRESHOLDS,
    IR_VEGA_CONCENTRATION_THRESHOLDS,
    IR_TENOR_LABELS,
)

from quantark.simm.calibration.credit_qualifying import (
    CREDIT_QUALIFYING_RISK_WEIGHTS,
    CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS,
    CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS,
    CREDIT_QUALIFYING_VRW,
    CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT,
    CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION,
    CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
)

from quantark.simm.calibration.credit_non_qualifying import (
    CREDIT_NON_QUALIFYING_RISK_WEIGHTS,
    CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS,
    CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION,
    CREDIT_NON_QUALIFYING_VRW,
    CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
)

from quantark.simm.calibration.equity import (
    EQUITY_RISK_WEIGHTS,
    EQUITY_INTRA_BUCKET_CORRELATIONS,
    EQUITY_INTER_BUCKET_CORRELATIONS,
    EQUITY_HVR,
    EQUITY_VRW,
    EQUITY_DELTA_CONCENTRATION_THRESHOLDS,
    EQUITY_VEGA_CONCENTRATION_THRESHOLDS,
    EQUITY_BUCKET_LABELS,
)

from quantark.simm.calibration.commodity import (
    COMMODITY_RISK_WEIGHTS,
    COMMODITY_INTRA_BUCKET_CORRELATIONS,
    COMMODITY_INTER_BUCKET_CORRELATIONS,
    COMMODITY_HVR,
    COMMODITY_VRW,
    COMMODITY_DELTA_CONCENTRATION_THRESHOLDS,
    COMMODITY_VEGA_CONCENTRATION_THRESHOLDS,
    COMMODITY_BUCKET_LABELS,
)

from quantark.simm.calibration.fx import (
    FX_RISK_WEIGHTS,
    FX_CORRELATIONS,
    FX_VEGA_CURVATURE_CORRELATION,
    FX_HVR,
    FX_VRW,
    FX_DELTA_CONCENTRATION_THRESHOLDS,
    FX_VEGA_CONCENTRATION_THRESHOLDS,
    FX_VOLATILITY_GROUPS,
    FX_VOLATILITY_GROUP_LABELS,
)

from quantark.simm.calibration.cross_risk import (
    INTER_RISK_CLASS_CORRELATIONS,
    INTER_RISK_CLASS_CORRELATION_LABELS,
)


# ============================================================================
# ENUMS
# ============================================================================

class RiskClass(Enum):
    """Enumeration of SIMM risk classes."""
    IR = "IR"
    CREDIT_QUALIFYING = "CreditQ"
    CREDIT_NON_QUALIFYING = "CreditNQ"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
    FX = "FX"


class MarginType(Enum):
    """Enumeration of margin types."""
    DELTA = "delta"
    VEGA = "vega"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_bucket_key(bucket: Union[int, str, Tuple]) -> Union[int, str]:
    """Normalize bucket key for dictionary lookup."""
    if isinstance(bucket, tuple):
        return bucket
    return bucket


# ============================================================================
# UNIFIED ACCESSOR FUNCTIONS
# ============================================================================

def get_risk_weight(
    risk_class: Union[RiskClass, str],
    bucket: Union[int, str, Tuple],
    tenor: Optional[str] = None,
    currency_group: Optional[str] = None,
    volatility_group: Optional[str] = None
) -> float:
    """Get risk weight for a given risk class and dimensions.

    Args:
        risk_class: The risk class (IR, CreditQ, CreditNQ, Equity, Commodity, FX)
        bucket: Bucket number or identifier
        tenor: Tenor label (for IR only)
        currency_group: Currency volatility group (for IR only: 'regular', 'low', 'high')
        volatility_group: Volatility group (for FX only: 'regular', 'high')

    Returns:
        The risk weight value

    Raises:
        ValueError: If parameters are invalid for the risk class
    """
    rc = risk_class.value if isinstance(risk_class, RiskClass) else risk_class

    if rc == "IR":
        if tenor is None or currency_group is None:
            raise ValueError("IR risk weights require tenor and currency_group")
        key = (tenor, currency_group)
        return IR_RISK_WEIGHTS.get(key, 0.0)

    elif rc == "CreditQ":
        key = _get_bucket_key(bucket)
        return CREDIT_QUALIFYING_RISK_WEIGHTS.get(key, 0.0)

    elif rc == "CreditNQ":
        key = _get_bucket_key(bucket)
        return CREDIT_NON_QUALIFYING_RISK_WEIGHTS.get(key, 0.0)

    elif rc == "Equity":
        key = _get_bucket_key(bucket)
        return EQUITY_RISK_WEIGHTS.get(key, 0.0)

    elif rc == "Commodity":
        key = _get_bucket_key(bucket)
        return COMMODITY_RISK_WEIGHTS.get(key, 0.0)

    elif rc == "FX":
        if volatility_group is None:
            raise ValueError("FX risk weights require volatility_group")
        # Map volatility group to matrix indices
        row = FX_VOLATILITY_GROUPS.get(volatility_group, 1) - 1
        col = FX_VOLATILITY_GROUPS.get(volatility_group, 1) - 1
        return FX_RISK_WEIGHTS[row, col]

    else:
        raise ValueError(f"Unknown risk class: {risk_class}")


def get_intra_bucket_correlation(
    risk_class: Union[RiskClass, str],
    bucket: Union[int, str, Tuple],
    risk_factor_1: Any = None,
    risk_factor_2: Any = None
) -> float:
    """Get correlation between two risk factors in the same bucket.

    Args:
        risk_class: The risk class
        bucket: Bucket number or identifier
        risk_factor_1: First risk factor (for same issuer correlation check)
        risk_factor_2: Second risk factor

    Returns:
        The intra-bucket correlation value
    """
    rc = risk_class.value if isinstance(risk_class, RiskClass) else risk_class

    if rc == "IR":
        # IR doesn't have bucket-based intra-bucket correlations
        return IR_SUB_CURVE_CORRELATION

    elif rc == "CreditQ":
        key = _get_bucket_key(bucket)
        # Check for same issuer
        if risk_factor_1 == risk_factor_2:
            return CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS["same_issuer"]
        return CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS.get(key, 0.0)

    elif rc == "CreditNQ":
        key = _get_bucket_key(bucket)
        # Check for same issuer
        if risk_factor_1 == risk_factor_2:
            return CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS["same_issuer"]
        return CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS.get(key, 0.0)

    elif rc == "Equity":
        key = _get_bucket_key(bucket)
        return EQUITY_INTRA_BUCKET_CORRELATIONS.get(key, 0.0)

    elif rc == "Commodity":
        key = _get_bucket_key(bucket)
        return COMMODITY_INTRA_BUCKET_CORRELATIONS.get(key, 0.0)

    elif rc == "FX":
        # FX doesn't have bucket-based intra-bucket correlations
        return FX_VEGA_CURVATURE_CORRELATION

    else:
        raise ValueError(f"Unknown risk class: {risk_class}")


def get_inter_bucket_correlation(
    risk_class: Union[RiskClass, str],
    bucket_1: Union[int, str, Tuple],
    bucket_2: Union[int, str, Tuple]
) -> float:
    """Get correlation between two different buckets.

    Args:
        risk_class: The risk class
        bucket_1: First bucket
        bucket_2: Second bucket

    Returns:
        The inter-bucket correlation value
    """
    rc = risk_class.value if isinstance(risk_class, RiskClass) else risk_class

    if rc == "IR":
        # IR tenor correlations
        idx1 = IR_TENOR_LABELS.index(bucket_1) if isinstance(bucket_1, str) else bucket_1
        idx2 = IR_TENOR_LABELS.index(bucket_2) if isinstance(bucket_2, str) else bucket_2
        return IR_TENOR_CORRELATIONS[idx1, idx2]

    elif rc == "CreditQ":
        # Convert bucket labels to indices
        idx1 = bucket_1 - 1  # Buckets are 1-indexed
        idx2 = bucket_2 - 1
        return CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS[idx1, idx2]

    elif rc == "CreditNQ":
        # Only two buckets, return the single correlation value
        return CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION

    elif rc == "Equity":
        # Convert bucket labels to indices
        idx1 = bucket_1 - 1  # Buckets are 1-indexed
        idx2 = bucket_2 - 1
        return EQUITY_INTER_BUCKET_CORRELATIONS[idx1, idx2]

    elif rc == "Commodity":
        # Convert bucket labels to indices
        idx1 = bucket_1 - 1  # Buckets are 1-indexed
        idx2 = bucket_2 - 1
        return COMMODITY_INTER_BUCKET_CORRELATIONS[idx1, idx2]

    elif rc == "FX":
        # FX bucket correlations
        vol_group_1 = FX_VOLATILITY_GROUPS.get(bucket_1, 1) - 1
        vol_group_2 = FX_VOLATILITY_GROUPS.get(bucket_2, 1) - 1
        return FX_CORRELATIONS[vol_group_1, vol_group_2]

    else:
        raise ValueError(f"Unknown risk class: {risk_class}")


def get_inter_risk_class_correlation(
    risk_class_1: Union[RiskClass, str],
    risk_class_2: Union[RiskClass, str]
) -> float:
    """Get correlation between two different risk classes (ψ).

    Args:
        risk_class_1: First risk class
        risk_class_2: Second risk class

    Returns:
        The inter-risk-class correlation value
    """
    rc1 = risk_class_1.value if isinstance(risk_class_1, RiskClass) else risk_class_1
    rc2 = risk_class_2.value if isinstance(risk_class_2, RiskClass) else risk_class_2

    idx1 = INTER_RISK_CLASS_CORRELATION_LABELS.index(rc1)
    idx2 = INTER_RISK_CLASS_CORRELATION_LABELS.index(rc2)

    return INTER_RISK_CLASS_CORRELATIONS[idx1, idx2]


def get_concentration_threshold(
    risk_class: Union[RiskClass, str],
    bucket: Union[int, str, Tuple],
    margin_type: MarginType = MarginType.DELTA
) -> float:
    """Get concentration threshold for a bucket.

    Args:
        risk_class: The risk class
        bucket: Bucket number or identifier
        margin_type: Type of margin (delta or vega)

    Returns:
        The concentration threshold value
    """
    rc = risk_class.value if isinstance(risk_class, RiskClass) else risk_class
    mt = margin_type.value if isinstance(margin_type, MarginType) else margin_type

    if rc == "IR":
        if mt == "delta":
            # IR uses currency group, not bucket
            key = bucket
            return IR_DELTA_CONCENTRATION_THRESHOLDS.get(key, 0.0)
        else:
            key = bucket
            return IR_VEGA_CONCENTRATION_THRESHOLDS.get(key, 0.0)

    elif rc == "CreditQ":
        key = _get_bucket_key(bucket)
        if mt == "delta":
            return CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS.get(key, 0.0)
        else:
            # CreditQ vega uses a single threshold
            return CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD

    elif rc == "CreditNQ":
        key = _get_bucket_key(bucket)
        if mt == "delta":
            return CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS.get(key, 0.0)
        else:
            return CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD

    elif rc == "Equity":
        key = _get_bucket_key(bucket)
        if mt == "delta":
            return EQUITY_DELTA_CONCENTRATION_THRESHOLDS.get(key, 0.0)
        else:
            return EQUITY_VEGA_CONCENTRATION_THRESHOLDS.get(key, 0.0)

    elif rc == "Commodity":
        key = _get_bucket_key(bucket)
        if mt == "delta":
            return COMMODITY_DELTA_CONCENTRATION_THRESHOLDS.get(key, 0.0)
        else:
            return COMMODITY_VEGA_CONCENTRATION_THRESHOLDS.get(key, 0.0)

    elif rc == "FX":
        key = _get_bucket_key(bucket)
        if mt == "delta":
            return FX_DELTA_CONCENTRATION_THRESHOLDS.get(key, 0.0)
        else:
            # FX vega uses tuple key
            return FX_VEGA_CONCENTRATION_THRESHOLDS.get(key, 0.0)

    else:
        raise ValueError(f"Unknown risk class: {risk_class}")


def get_hvr(risk_class: Union[RiskClass, str]) -> float:
    """Get Historical Volatility Ratio for a risk class.

    Args:
        risk_class: The risk class

    Returns:
        The HVR value
    """
    rc = risk_class.value if isinstance(risk_class, RiskClass) else risk_class

    if rc == "IR":
        return IR_HVR
    elif rc == "Equity":
        return EQUITY_HVR
    elif rc == "Commodity":
        return COMMODITY_HVR
    elif rc == "FX":
        return FX_HVR
    else:
        # Credit classes don't have HVR
        return 0.0


def get_vrw(
    risk_class: Union[RiskClass, str],
    bucket: Optional[Union[int, str]] = None
) -> float:
    """Get Vega Risk Weight for a risk class.

    Args:
        risk_class: The risk class
        bucket: Bucket number (for special cases like Equity bucket 12)

    Returns:
        The VRW value
    """
    rc = risk_class.value if isinstance(risk_class, RiskClass) else risk_class

    if rc == "IR":
        return IR_VRW
    elif rc == "CreditQ":
        return CREDIT_QUALIFYING_VRW
    elif rc == "CreditNQ":
        return CREDIT_NON_QUALIFYING_VRW
    elif rc == "Equity":
        if isinstance(EQUITY_VRW, dict):
            if bucket == 12:
                return EQUITY_VRW[12]
            else:
                return EQUITY_VRW["default"]
        return EQUITY_VRW
    elif rc == "Commodity":
        return COMMODITY_VRW
    elif rc == "FX":
        return FX_VRW
    else:
        raise ValueError(f"Unknown risk class: {risk_class}")
