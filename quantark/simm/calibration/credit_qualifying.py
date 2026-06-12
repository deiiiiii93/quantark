"""
Credit Qualifying calibration parameters for ISDA SIMM v2.6.

All values are transcribed from the ISDA SIMM Methodology, version 2.6,
Section E (Credit Qualifying risk) and Sections J.2/J.7 (concentration
thresholds). Paragraph references are given on each table.
"""

import numpy as np


# ============================================================================
# DELTA RISK WEIGHTS (paragraph 39)
# ============================================================================

# The same risk weight is used for all vertices (1y, 2y, 3y, 5y, 10y).
# Key "Residual" denotes the residual bucket.
CREDIT_QUALIFYING_RISK_WEIGHTS = {
    1: 75.0,
    2: 90.0,
    3: 84.0,
    4: 54.0,
    5: 62.0,
    6: 48.0,
    7: 185.0,
    8: 343.0,
    9: 255.0,
    10: 250.0,
    11: 214.0,
    12: 173.0,
    "Residual": 343.0,
}

# Vega risk weight for the Credit risk class (paragraph 40).
CREDIT_QUALIFYING_VRW = 0.76

# Base Correlation risk weight, all index families (paragraph 41).
CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT = 10.0


# ============================================================================
# CORRELATIONS (paragraphs 42-43)
# ============================================================================

# Intra-bucket correlations rho_kl (paragraph 42).
# Same issuer/seniority, different vertex or currency: 93%.
# Different issuer/seniority: 46%.
# Residual bucket: 50% for both cases.
CREDIT_QUALIFYING_SAME_ISSUER_CORRELATION = 0.93
CREDIT_QUALIFYING_DIFFERENT_ISSUER_CORRELATION = 0.46
CREDIT_QUALIFYING_RESIDUAL_SAME_ISSUER_CORRELATION = 0.50
CREDIT_QUALIFYING_RESIDUAL_DIFFERENT_ISSUER_CORRELATION = 0.50

# Base Correlation correlation across different index families (paragraph 42).
CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION = 0.29

# Inter-bucket correlations gamma_bc across non-residual buckets
# (paragraph 43), 12x12 indexed by bucket-1.
CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS = np.array([
    #   1     2     3     4     5     6     7     8     9     10    11    12
    [1.00, 0.38, 0.38, 0.35, 0.37, 0.34, 0.42, 0.32, 0.34, 0.33, 0.34, 0.33],  # 1
    [0.38, 1.00, 0.48, 0.46, 0.48, 0.46, 0.39, 0.40, 0.41, 0.41, 0.43, 0.40],  # 2
    [0.38, 0.48, 1.00, 0.50, 0.51, 0.50, 0.40, 0.39, 0.45, 0.44, 0.47, 0.42],  # 3
    [0.35, 0.46, 0.50, 1.00, 0.50, 0.50, 0.37, 0.37, 0.41, 0.43, 0.45, 0.40],  # 4
    [0.37, 0.48, 0.51, 0.50, 1.00, 0.50, 0.39, 0.38, 0.43, 0.43, 0.46, 0.42],  # 5
    [0.34, 0.46, 0.50, 0.50, 0.50, 1.00, 0.37, 0.35, 0.39, 0.41, 0.44, 0.41],  # 6
    [0.42, 0.39, 0.40, 0.37, 0.39, 0.37, 1.00, 0.33, 0.37, 0.37, 0.35, 0.35],  # 7
    [0.32, 0.40, 0.39, 0.37, 0.38, 0.35, 0.33, 1.00, 0.36, 0.37, 0.37, 0.36],  # 8
    [0.34, 0.41, 0.45, 0.41, 0.43, 0.39, 0.37, 0.36, 1.00, 0.41, 0.40, 0.38],  # 9
    [0.33, 0.41, 0.44, 0.43, 0.43, 0.41, 0.37, 0.37, 0.41, 1.00, 0.41, 0.39],  # 10
    [0.34, 0.43, 0.47, 0.45, 0.46, 0.44, 0.35, 0.37, 0.40, 0.41, 1.00, 0.40],  # 11
    [0.33, 0.40, 0.42, 0.40, 0.42, 0.41, 0.35, 0.36, 0.38, 0.39, 0.40, 1.00],  # 12
])


# ============================================================================
# CONCENTRATION THRESHOLDS (Sections J.2 and J.7)
# ============================================================================

# Delta concentration thresholds, USD mm/bp (paragraph 76).
# Buckets 1 and 7 (sovereigns including central banks): 1.0.
# Buckets 2-6, 8-12 (corporate entities) and Residual: 0.17.
CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS = {
    1: 1.0,
    2: 0.17,
    3: 0.17,
    4: 0.17,
    5: 0.17,
    6: 0.17,
    7: 1.0,
    8: 0.17,
    9: 0.17,
    10: 0.17,
    11: 0.17,
    12: 0.17,
    "Residual": 0.17,
}

# Vega concentration threshold, USD mm (paragraph 83), all buckets
# including residual.
CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD = 360.0
