"""
Equity calibration parameters for ISDA SIMM v2.6.

All values are transcribed from the ISDA SIMM Methodology, version 2.6,
Section G (Equity risk) and Sections J.3/J.8 (concentration thresholds).
Paragraph references are given on each table.
"""

import numpy as np


# ============================================================================
# DELTA RISK WEIGHTS (paragraph 56)
# ============================================================================

EQUITY_RISK_WEIGHTS = {
    1: 30.0,
    2: 33.0,
    3: 36.0,
    4: 29.0,
    5: 26.0,
    6: 25.0,
    7: 34.0,
    8: 28.0,
    9: 36.0,
    10: 50.0,
    11: 19.0,   # Indexes, Funds, ETFs
    12: 19.0,   # Volatility Indexes
    "Residual": 50.0,
}

# Historical volatility ratio for the equity risk class (paragraph 57).
EQUITY_HVR = 0.60

# Vega risk weight (paragraph 58): 0.45 for all buckets except bucket 12
# (Volatility Indexes), for which it is 0.96.
EQUITY_VRW_DEFAULT = 0.45
EQUITY_VRW_VOLATILITY_INDEX = 0.96


def get_equity_vrw(bucket) -> float:
    """Vega risk weight for an equity bucket (paragraph 58)."""
    if bucket == 12:
        return EQUITY_VRW_VOLATILITY_INDEX
    return EQUITY_VRW_DEFAULT


# ============================================================================
# CORRELATIONS (paragraphs 59-60)
# ============================================================================

# Intra-bucket correlations rho_kl (paragraph 59).
EQUITY_INTRA_BUCKET_CORRELATIONS = {
    1: 0.18,
    2: 0.20,
    3: 0.28,
    4: 0.24,
    5: 0.25,
    6: 0.36,
    7: 0.35,
    8: 0.37,
    9: 0.23,
    10: 0.27,
    11: 0.45,
    12: 0.45,
    "Residual": 0.00,
}

# Inter-bucket correlations gamma_bc across non-residual buckets
# (paragraph 60), 12x12 indexed by bucket-1.
EQUITY_INTER_BUCKET_CORRELATIONS = np.array([
    #   1     2     3     4     5     6     7     8     9     10    11    12
    [1.00, 0.18, 0.19, 0.19, 0.14, 0.16, 0.15, 0.16, 0.18, 0.12, 0.19, 0.19],  # 1
    [0.18, 1.00, 0.22, 0.21, 0.15, 0.18, 0.17, 0.19, 0.20, 0.14, 0.21, 0.21],  # 2
    [0.19, 0.22, 1.00, 0.22, 0.13, 0.16, 0.18, 0.17, 0.22, 0.13, 0.20, 0.20],  # 3
    [0.19, 0.21, 0.22, 1.00, 0.17, 0.22, 0.22, 0.23, 0.22, 0.17, 0.26, 0.26],  # 4
    [0.14, 0.15, 0.13, 0.17, 1.00, 0.29, 0.26, 0.29, 0.14, 0.24, 0.32, 0.32],  # 5
    [0.16, 0.18, 0.16, 0.22, 0.29, 1.00, 0.34, 0.36, 0.17, 0.30, 0.39, 0.39],  # 6
    [0.15, 0.17, 0.18, 0.22, 0.26, 0.34, 1.00, 0.33, 0.16, 0.28, 0.36, 0.36],  # 7
    [0.16, 0.19, 0.17, 0.23, 0.29, 0.36, 0.33, 1.00, 0.17, 0.29, 0.40, 0.40],  # 8
    [0.18, 0.20, 0.22, 0.22, 0.14, 0.17, 0.16, 0.17, 1.00, 0.13, 0.21, 0.21],  # 9
    [0.12, 0.14, 0.13, 0.17, 0.24, 0.30, 0.28, 0.29, 0.13, 1.00, 0.30, 0.30],  # 10
    [0.19, 0.21, 0.20, 0.26, 0.32, 0.39, 0.36, 0.40, 0.21, 0.30, 1.00, 0.45],  # 11
    [0.19, 0.21, 0.20, 0.26, 0.32, 0.39, 0.36, 0.40, 0.21, 0.30, 0.45, 1.00],  # 12
])


# ============================================================================
# CONCENTRATION THRESHOLDS (Sections J.3 and J.8)
# ============================================================================

# Delta concentration thresholds, USD mm/% (paragraph 77).
EQUITY_DELTA_CONCENTRATION_THRESHOLDS = {
    1: 3.0,
    2: 3.0,
    3: 3.0,
    4: 3.0,
    5: 12.0,
    6: 12.0,
    7: 12.0,
    8: 12.0,
    9: 0.64,
    10: 0.37,
    11: 810.0,
    12: 810.0,
    "Residual": 0.37,
}

# Vega concentration thresholds, USD mm (paragraph 84).
EQUITY_VEGA_CONCENTRATION_THRESHOLDS = {
    1: 210.0,
    2: 210.0,
    3: 210.0,
    4: 210.0,
    5: 1300.0,
    6: 1300.0,
    7: 1300.0,
    8: 1300.0,
    9: 39.0,
    10: 190.0,
    11: 6400.0,
    12: 6400.0,
    "Residual": 39.0,
}
