"""
SIMM Calibration Data Module

This module provides access to all calibration parameters for ISDA SIMM v2.6.
Calibration parameters include risk weights, correlations, concentration thresholds,
and other factors needed for initial margin calculations.

The module is organized by risk class:
- Interest Rate (IR)
- Credit Qualifying (CreditQ)
- Credit Non-Qualifying (CreditNQ)
- Equity
- Commodity
- Foreign Exchange (FX)

Each submodule contains risk-specific parameters following the SIMM v2.6 specification
published by ISDA.

Example usage:
    >>> from simm.calibration import get_risk_weight, get_intra_bucket_correlation
    >>> # Get Interest Rate risk weight
    >>> risk_weight = get_risk_weight("IR", tenor="1yr", currency="USD")
    >>> # Get Equity intra-bucket correlation
    >>> corr = get_intra_bucket_correlation("EQUITY", 1, "Equity", "Equity")

References:
    - ISDA SIMM Methodology, Version 2.6, effective December 2, 2023
"""

from simm.calibration.version import CURRENT_VERSION

# Import all parameter accessors
from simm.calibration.ir import (
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

from simm.calibration.credit_qualifying import (
    CREDIT_QUALIFYING_RISK_WEIGHTS,
    CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS,
    CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS,
    CREDIT_QUALIFYING_VRW,
    CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT,
    CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION,
    CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
)

from simm.calibration.credit_non_qualifying import (
    CREDIT_NON_QUALIFYING_RISK_WEIGHTS,
    CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS,
    CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION,
    CREDIT_NON_QUALIFYING_VRW,
    CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
)

from simm.calibration.equity import (
    EQUITY_RISK_WEIGHTS,
    EQUITY_INTRA_BUCKET_CORRELATIONS,
    EQUITY_INTER_BUCKET_CORRELATIONS,
    EQUITY_HVR,
    EQUITY_VRW,
    EQUITY_DELTA_CONCENTRATION_THRESHOLDS,
    EQUITY_VEGA_CONCENTRATION_THRESHOLDS,
    EQUITY_BUCKET_LABELS,
)

from simm.calibration.commodity import (
    COMMODITY_RISK_WEIGHTS,
    COMMODITY_INTRA_BUCKET_CORRELATIONS,
    COMMODITY_INTER_BUCKET_CORRELATIONS,
    COMMODITY_HVR,
    COMMODITY_VRW,
    COMMODITY_DELTA_CONCENTRATION_THRESHOLDS,
    COMMODITY_VEGA_CONCENTRATION_THRESHOLDS,
    COMMODITY_BUCKET_LABELS,
)

from simm.calibration.fx import (
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

from simm.calibration.cross_risk import (
    INTER_RISK_CLASS_CORRELATIONS,
    INTER_RISK_CLASS_CORRELATION_LABELS,
)

# Unified accessor functions
from simm.calibration.accessors import (
    get_risk_weight,
    get_intra_bucket_correlation,
    get_inter_bucket_correlation,
    get_inter_risk_class_correlation,
    get_concentration_threshold,
    get_hvr,
    get_vrw,
    MarginType,
    RiskClass,
)


__all__ = [
    # Version
    "CURRENT_VERSION",

    # IR Parameters
    "IR_RISK_WEIGHTS",
    "IR_TENOR_CORRELATIONS",
    "IR_SUB_CURVE_CORRELATION",
    "IR_INFLATION_CORRELATION",
    "IR_CROSS_CURRENCY_BASIS_CORRELATION",
    "IR_INTER_CURRENCY_CORRELATION",
    "IR_INFLATION_RISK_WEIGHT",
    "IR_CROSS_CURRENCY_BASIS_RISK_WEIGHT",
    "IR_HVR",
    "IR_VRW",
    "IR_DELTA_CONCENTRATION_THRESHOLDS",
    "IR_VEGA_CONCENTRATION_THRESHOLDS",
    "IR_TENOR_LABELS",

    # Credit Qualifying Parameters
    "CREDIT_QUALIFYING_RISK_WEIGHTS",
    "CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS",
    "CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS",
    "CREDIT_QUALIFYING_VRW",
    "CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT",
    "CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION",
    "CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS",
    "CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD",

    # Credit Non-Qualifying Parameters
    "CREDIT_NON_QUALIFYING_RISK_WEIGHTS",
    "CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS",
    "CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION",
    "CREDIT_NON_QUALIFYING_VRW",
    "CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS",
    "CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD",

    # Equity Parameters
    "EQUITY_RISK_WEIGHTS",
    "EQUITY_INTRA_BUCKET_CORRELATIONS",
    "EQUITY_INTER_BUCKET_CORRELATIONS",
    "EQUITY_HVR",
    "EQUITY_VRW",
    "EQUITY_DELTA_CONCENTRATION_THRESHOLDS",
    "EQUITY_VEGA_CONCENTRATION_THRESHOLDS",
    "EQUITY_BUCKET_LABELS",

    # Commodity Parameters
    "COMMODITY_RISK_WEIGHTS",
    "COMMODITY_INTRA_BUCKET_CORRELATIONS",
    "COMMODITY_INTER_BUCKET_CORRELATIONS",
    "COMMODITY_HVR",
    "COMMODITY_VRW",
    "COMMODITY_DELTA_CONCENTRATION_THRESHOLDS",
    "COMMODITY_VEGA_CONCENTRATION_THRESHOLDS",
    "COMMODITY_BUCKET_LABELS",

    # FX Parameters
    "FX_RISK_WEIGHTS",
    "FX_CORRELATIONS",
    "FX_VEGA_CURVATURE_CORRELATION",
    "FX_HVR",
    "FX_VRW",
    "FX_DELTA_CONCENTRATION_THRESHOLDS",
    "FX_VEGA_CONCENTRATION_THRESHOLDS",
    "FX_VOLATILITY_GROUPS",
    "FX_VOLATILITY_GROUP_LABELS",

    # Cross-Risk Parameters
    "INTER_RISK_CLASS_CORRELATIONS",
    "INTER_RISK_CLASS_CORRELATION_LABELS",

    # Accessor Functions
    "get_risk_weight",
    "get_intra_bucket_correlation",
    "get_inter_bucket_correlation",
    "get_inter_risk_class_correlation",
    "get_concentration_threshold",
    "get_hvr",
    "get_vrw",

    # Enums
    "MarginType",
    "RiskClass",
]
