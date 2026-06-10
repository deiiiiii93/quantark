"""
Foreign Exchange Calibration Parameters for SIMM v2.6

This module contains all Foreign Exchange (FX) calibration parameters as
specified in ISDA SIMM v2.6, effective December 2, 2023.

References:
    - ISDA SIMM Methodology, Version 2.6, Section 2.6 (Foreign Exchange Risk)
"""

import numpy as np


# ============================================================================
# VOLATILITY GROUPS
# ============================================================================

# FX volatility groups for calculation currency
# Group 1: Regular volatility calculation currencies
# Group 2: High volatility calculation currencies
FX_VOLATILITY_GROUPS = {
    "regular": 1,  # Regular volatility calculation currencies (AUD, BRL, CAD, CHF, CNY, CZK, DKK, EUR, GBP, HKD, HUF, ILS, ISK, JPY, KRW, MXN, NOK, NZD, PLN, SEK, SGD, THB, TRY, TWD, USD, ZAR)
    "high": 2      # High volatility calculation currencies (all others)
}


# ============================================================================
# RISK WEIGHTS
# ============================================================================

# FX risk weights by volatility group pair (2x2 matrix)
# Rows: Calculation currency volatility group
# Columns: Underlying currency volatility group
# Source: ISDA SIMM v2.6, Table FX-1
FX_RISK_WEIGHTS = np.array([
    #  Regular Ccy  High Vol Ccy
    [15,            18],             # Regular calc currency
    [18,            21]              # High vol calc currency
])

# Labels for FX volatility groups
FX_VOLATILITY_GROUP_LABELS = ["regular", "high"]


# ============================================================================
# CORRELATIONS
# ============================================================================

# FX correlations by volatility group (2x2 matrix)
# Correlation between currency pairs by volatility group
# Source: ISDA SIMM v2.6, Table FX-2
FX_CORRELATIONS = np.array([
    #  Regular Ccy  High Vol Ccy
    [0.46,          0.38],           # Regular calc currency
    [0.38,          0.38]            # High vol calc currency
])

# FX vega/curvature correlation (50%)
# Correlation between vega and curvature risk
FX_VEGA_CURVATURE_CORRELATION = 0.50


# ============================================================================
# HISTORICAL VOLATILITY RATIOS AND VEGA RISK WEIGHTS
# ============================================================================

# FX Historical Volatility Ratio (HVR)
FX_HVR = 0.57

# FX Vega Risk Weight (VRW)
FX_VRW = 0.48


# ============================================================================
# CONCENTRATION THRESHOLDS
# ============================================================================

# FX delta concentration thresholds by category (USD million)
# Source: ISDA SIMM v2.6, Section 2.6.3
FX_DELTA_CONCENTRATION_THRESHOLDS = {
    "regular": 100,  # Regular volatility currencies
    "high": 50       # High volatility currencies
}

# FX vega concentration thresholds by category pair (USD million)
# Source: ISDA SIMM v2.6, Section 2.6.4
FX_VEGA_CONCENTRATION_THRESHOLDS = {
    ("regular", "regular"): 100,
    ("regular", "high"): 100,
    ("high", "regular"): 100,
    ("high", "high"): 50
}
