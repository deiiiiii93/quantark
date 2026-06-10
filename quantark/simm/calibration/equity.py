"""
Equity Calibration Parameters for SIMM v2.6

This module contains all Equity calibration parameters as specified in
ISDA SIMM v2.6, effective December 2, 2023.

References:
    - ISDA SIMM Methodology, Version 2.6, Section 2.4 (Equity Risk)
"""

import numpy as np


# ============================================================================
# RISK WEIGHTS
# ============================================================================

# Equity risk weights by bucket (12 buckets + residual)
# Source: ISDA SIMM v2.6, Table EQ-1
EQUITY_RISK_WEIGHTS = {
    1: 30,   # Emerging Markets Large Cap
    2: 33,   # Emerging Markets Large Cap
    3: 36,   # Emerging Markets Large Cap
    4: 29,   # Emerging Markets Large Cap
    5: 26,   # Developed Markets Large Cap
    6: 25,   # Developed Markets Large Cap
    7: 34,   # Developed Markets Large Cap
    8: 28,   # Developed Markets Large Cap
    9: 36,   # Emerging Markets Small Cap
    10: 50,  # Developed Markets Small Cap
    11: 19,  # Indexes - Emerging Markets
    12: 19,  # Indexes - Developed Markets
    "Residual": 50
}


# ============================================================================
# INTRA-BUCKET CORRELATIONS
# ============================================================================

# Equity intra-bucket correlations by bucket
# Source: ISDA SIMM v2.6, Section 2.4.1
EQUITY_INTRA_BUCKET_CORRELATIONS = {
    1: 0.18,  # Emerging Markets Large Cap
    2: 0.20,  # Emerging Markets Large Cap
    3: 0.28,  # Emerging Markets Large Cap
    4: 0.24,  # Emerging Markets Large Cap
    5: 0.25,  # Developed Markets Large Cap
    6: 0.36,  # Developed Markets Large Cap
    7: 0.35,  # Developed Markets Large Cap
    8: 0.37,  # Developed Markets Large Cap
    9: 0.23,  # Emerging Markets Small Cap
    10: 0.27,  # Developed Markets Small Cap
    11: 0.45,  # Indexes - Emerging Markets
    12: 0.45,  # Indexes - Developed Markets
    "Residual": 0.00
}


# ============================================================================
# INTER-BUCKET CORRELATIONS
# ============================================================================

# Equity inter-bucket correlation matrix (12x12)
# Source: ISDA SIMM v2.6, Table EQ-2
EQUITY_INTER_BUCKET_CORRELATIONS = np.array([
    #    1     2     3     4     5     6     7     8     9    10    11    12
    [0.00, 0.18, 0.19, 0.19, 0.14, 0.16, 0.15, 0.16, 0.18, 0.12, 0.19, 0.19],  # 1 EM Large
    [0.18, 0.00, 0.22, 0.21, 0.15, 0.18, 0.17, 0.19, 0.20, 0.14, 0.21, 0.21],  # 2 EM Large
    [0.19, 0.22, 0.00, 0.23, 0.16, 0.19, 0.18, 0.20, 0.21, 0.15, 0.23, 0.23],  # 3 EM Large
    [0.19, 0.21, 0.23, 0.00, 0.16, 0.18, 0.18, 0.19, 0.20, 0.14, 0.23, 0.23],  # 4 EM Large
    [0.14, 0.15, 0.16, 0.16, 0.00, 0.32, 0.30, 0.33, 0.11, 0.24, 0.16, 0.16],  # 5 DM Large
    [0.16, 0.18, 0.19, 0.18, 0.32, 0.00, 0.35, 0.37, 0.13, 0.27, 0.18, 0.18],  # 6 DM Large
    [0.15, 0.17, 0.18, 0.18, 0.30, 0.35, 0.00, 0.36, 0.12, 0.26, 0.18, 0.18],  # 7 DM Large
    [0.16, 0.19, 0.20, 0.19, 0.33, 0.37, 0.36, 0.00, 0.13, 0.28, 0.19, 0.19],  # 8 DM Large
    [0.18, 0.20, 0.21, 0.20, 0.11, 0.13, 0.12, 0.13, 0.00, 0.12, 0.21, 0.21],  # 9 EM Small
    [0.12, 0.14, 0.15, 0.14, 0.24, 0.27, 0.26, 0.28, 0.12, 0.00, 0.14, 0.14],  # 10 DM Small
    [0.19, 0.21, 0.23, 0.23, 0.16, 0.18, 0.18, 0.19, 0.21, 0.14, 0.00, 0.45],  # 11 Index EM
    [0.19, 0.21, 0.23, 0.23, 0.16, 0.18, 0.18, 0.19, 0.21, 0.14, 0.45, 0.00],  # 12 Index DM
])

# Labels for Equity buckets
EQUITY_BUCKET_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ============================================================================
# HISTORICAL VOLATILITY RATIOS AND VEGA RISK WEIGHTS
# ============================================================================

# Equity Historical Volatility Ratio (HVR)
EQUITY_HVR = 0.60

# Equity Vega Risk Weight (VRW)
# Special case for bucket 12 (indexes)
EQUITY_VRW = {
    12: 0.96,  # Indexes - Developed Markets
    "default": 0.45
}


# ============================================================================
# CONCENTRATION THRESHOLDS
# ============================================================================

# Equity delta concentration thresholds (USD million / %)
# Source: ISDA SIMM v2.6, Section 2.4.3
EQUITY_DELTA_CONCENTRATION_THRESHOLDS = {
    (1, 2, 3, 4): 3,      # EM Large Cap
    (5, 6, 7, 8): 12,     # DM Large Cap
    9: 0.64,              # EM Small Cap
    10: 0.37,             # DM Small Cap
    (11, 12): 810,        # Indexes
    "Residual": 0.37
}

# Equity vega concentration thresholds (USD million / %)
# Source: ISDA SIMM v2.6, Section 2.4.4
EQUITY_VEGA_CONCENTRATION_THRESHOLDS = {
    (1, 2, 3, 4): 3,      # EM Large Cap
    (5, 6, 7, 8): 12,     # DM Large Cap
    9: 0.64,              # EM Small Cap
    10: 0.37,             # DM Small Cap
    (11, 12): 810,        # Indexes
    "Residual": 0.37
}
