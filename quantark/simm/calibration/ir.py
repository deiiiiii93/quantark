"""
Interest Rate Calibration Parameters for SIMM v2.6

This module contains all Interest Rate (IR) calibration parameters as specified
in ISDA SIMM v2.6, effective December 2, 2023.

References:
    - ISDA SIMM Methodology, Version 2.6, Section 2.1 (Interest Rate Risk)
"""

import numpy as np


# ============================================================================
# RISK WEIGHTS
# ============================================================================

# IR risk weights by (tenor_label, currency_group)
# Currency groups: 'regular', 'low', 'high'
IR_RISK_WEIGHTS = {
    # Regular volatility currencies (USD, EUR, GBP)
    ("2w", "regular"): 109,
    ("1m", "regular"): 105,
    ("3m", "regular"): 90,
    ("6m", "regular"): 71,
    ("1yr", "regular"): 66,
    ("2yr", "regular"): 66,
    ("3yr", "regular"): 64,
    ("5yr", "regular"): 60,
    ("10yr", "regular"): 60,
    ("15yr", "regular"): 61,
    ("20yr", "regular"): 61,
    ("30yr", "regular"): 67,

    # Low volatility currencies (JPY)
    ("2w", "low"): 15,
    ("1m", "low"): 18,
    ("3m", "low"): 9,
    ("6m", "low"): 11,
    ("1yr", "low"): 13,
    ("2yr", "low"): 15,
    ("3yr", "low"): 19,
    ("5yr", "low"): 23,
    ("10yr", "low"): 23,
    ("15yr", "low"): 22,
    ("20yr", "low"): 22,
    ("30yr", "low"): 23,

    # High volatility currencies
    ("2w", "high"): 163,
    ("1m", "high"): 109,
    ("3m", "high"): 87,
    ("6m", "high"): 89,
    ("1yr", "high"): 102,
    ("2yr", "high"): 96,
    ("3yr", "high"): 101,
    ("5yr", "high"): 97,
    ("10yr", "high"): 97,
    ("15yr", "high"): 102,
    ("20yr", "high"): 106,
    ("30yr", "high"): 101,
}

# Labels for IR tenors
IR_TENOR_LABELS = [
    "2w", "1m", "3m", "6m", "1yr", "2yr",
    "3yr", "5yr", "10yr", "15yr", "20yr", "30yr"
]


# ============================================================================
# CORRELATIONS
# ============================================================================

# IR tenor correlation matrix (12x12)
# Source: ISDA SIMM v2.6, Table IR-1
IR_TENOR_CORRELATIONS = np.array([
    #    2w   1m   3m   6m   1yr  2yr  3yr  5yr  10yr 15yr 20yr 30yr
    [1.00, 0.77, 0.67, 0.59, 0.48, 0.39, 0.34, 0.30, 0.25, 0.23, 0.21, 0.20],  # 2w
    [0.77, 1.00, 0.84, 0.74, 0.56, 0.43, 0.36, 0.31, 0.26, 0.21, 0.19, 0.19],  # 1m
    [0.67, 0.84, 1.00, 0.87, 0.62, 0.47, 0.39, 0.33, 0.27, 0.22, 0.20, 0.19],  # 3m
    [0.59, 0.74, 0.87, 1.00, 0.72, 0.54, 0.44, 0.36, 0.29, 0.23, 0.21, 0.20],  # 6m
    [0.48, 0.56, 0.62, 0.72, 1.00, 0.79, 0.65, 0.50, 0.38, 0.30, 0.27, 0.25],  # 1yr
    [0.39, 0.43, 0.47, 0.54, 0.79, 1.00, 0.87, 0.69, 0.50, 0.39, 0.34, 0.31],  # 2yr
    [0.34, 0.36, 0.39, 0.44, 0.65, 0.87, 1.00, 0.85, 0.62, 0.48, 0.42, 0.38],  # 3yr
    [0.30, 0.31, 0.33, 0.36, 0.50, 0.69, 0.85, 1.00, 0.79, 0.62, 0.54, 0.48],  # 5yr
    [0.25, 0.26, 0.27, 0.29, 0.38, 0.50, 0.62, 0.79, 1.00, 0.86, 0.75, 0.66],  # 10yr
    [0.23, 0.21, 0.22, 0.23, 0.30, 0.39, 0.48, 0.62, 0.86, 1.00, 0.92, 0.81],  # 15yr
    [0.21, 0.19, 0.20, 0.21, 0.27, 0.34, 0.42, 0.54, 0.75, 0.92, 1.00, 0.89],  # 20yr
    [0.20, 0.19, 0.19, 0.20, 0.25, 0.31, 0.38, 0.48, 0.66, 0.81, 0.89, 1.00],  # 30yr
])

# IR sub-curve correlation (99.3%)
# Correlation between tenor curves on same currency
IR_SUB_CURVE_CORRELATION = 0.993

# IR inflation correlation (24%)
# Correlation between inflation and nominal curves
IR_INFLATION_CORRELATION = 0.24

# IR cross-currency basis correlation (4%)
# Correlation between basis swaps across currencies
IR_CROSS_CURRENCY_BASIS_CORRELATION = 0.04

# IR inter-currency correlation (32%)
# Correlation between interest rates across different currencies
IR_INTER_CURRENCY_CORRELATION = 0.32


# ============================================================================
# ADDITIONAL RISK WEIGHTS
# ============================================================================

# IR inflation risk weight
IR_INFLATION_RISK_WEIGHT = 61

# IR cross-currency basis risk weight
IR_CROSS_CURRENCY_BASIS_RISK_WEIGHT = 21


# ============================================================================
# HISTORICAL VOLATILITY RATIOS AND VEGA RISK WEIGHTS
# ============================================================================

# IR Historical Volatility Ratio (HVR)
IR_HVR = 0.47

# IR Vega Risk Weight (VRW)
IR_VRW = 0.23


# ============================================================================
# CONCENTRATION THRESHOLDS
# ============================================================================

# IR delta concentration thresholds (USD million / basis point)
# Source: ISDA SIMM v2.6, Section 2.1.3
IR_DELTA_CONCENTRATION_THRESHOLDS = {
    "high": 30,  # High volatility currencies
    "regular_well_traded": 330,  # USD, EUR, GBP
    "regular_less_traded": 130,  # AUD, CAD, CHF, NZD, SEK, NOK, DKK
    "low": 61,  # JPY
}

# IR vega concentration thresholds (USD million / bp)
# Source: ISDA SIMM v2.6, Section 2.1.4
IR_VEGA_CONCENTRATION_THRESHOLDS = {
    "high": 30,  # High volatility currencies
    "regular_well_traded": 330,  # USD, EUR, GBP
    "regular_less_traded": 130,  # AUD, CAD, CHF, NZD, SEK, NOK, DKK
    "low": 61,  # JPY
}
