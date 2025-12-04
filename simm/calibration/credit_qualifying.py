"""
Credit Qualifying Calibration Parameters for SIMM v2.6

This module contains all Credit Qualifying (CreditQ) calibration parameters as
specified in ISDA SIMM v2.6, effective December 2, 2023.

References:
    - ISDA SIMM Methodology, Version 2.6, Section 2.2 (Credit Qualifying Risk)
"""

import numpy as np


# ============================================================================
# RISK WEIGHTS
# ============================================================================

# Credit Qualifying risk weights by bucket (12 buckets + residual)
# Source: ISDA SIMM v2.6, Table CQ-1
CREDIT_QUALIFYING_RISK_WEIGHTS = {
    1: 45,   # Sovereign
    2: 47,   # Local authority
    3: 69,   # Financial institution - Senior Unsecured
    4: 66,   # Covered bond
    5: 64,   # Financial institution - Senior Structured Finance
    6: 75,   # Corporate - Senior Unsecured
    7: 59,   # Sovereign entity
    8: 68,   # Corporate - Senior Secured
    9: 60,   # Asset backed
    10: 78,  # Residential mortgage backed
    11: 70,  # Commercial mortgage backed
    12: 62,  # Other structured finance
    "Residual": 100
}


# ============================================================================
# INTRA-BUCKET CORRELATIONS
# ============================================================================

# Credit Qualifying intra-bucket correlations
# Same issuer correlation vs different issuer correlation
# Source: ISDA SIMM v2.6, Section 2.2.1
CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS = {
    # Same issuer, different risk factor
    "same_issuer": 0.99,
    # Different issuer, same bucket
    1: 0.10,   # Sovereign
    2: 0.10,   # Local authority
    3: 0.13,   # Financial institution - Senior Unsecured
    4: 0.13,   # Covered bond
    5: 0.13,   # Financial institution - Senior Structured Finance
    6: 0.14,   # Corporate - Senior Unsecured
    7: 0.10,   # Sovereign entity
    8: 0.14,   # Corporate - Senior Secured
    9: 0.21,   # Asset backed
    10: 0.18,  # Residential mortgage backed
    11: 0.18,  # Commercial mortgage backed
    12: 0.21,  # Other structured finance
    "Residual": 0.21
}


# ============================================================================
# INTER-BUCKET CORRELATIONS
# ============================================================================

# Credit Qualifying inter-bucket correlation matrix (12x12)
# Source: ISDA SIMM v2.6, Table CQ-2
CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS = np.array([
    #    1     2     3     4     5     6     7     8     9    10    11    12
    [0.00, 0.40, 0.25, 0.25, 0.25, 0.20, 0.40, 0.20, 0.08, 0.12, 0.12, 0.08],  # 1 Sovereign
    [0.40, 0.00, 0.25, 0.25, 0.25, 0.20, 0.40, 0.20, 0.08, 0.12, 0.12, 0.08],  # 2 Local authority
    [0.25, 0.25, 0.00, 0.31, 0.31, 0.34, 0.25, 0.34, 0.14, 0.15, 0.15, 0.14],  # 3 FI Senior Unsecured
    [0.25, 0.25, 0.31, 0.00, 0.31, 0.25, 0.25, 0.25, 0.12, 0.12, 0.12, 0.12],  # 4 Covered bond
    [0.25, 0.25, 0.31, 0.31, 0.00, 0.25, 0.25, 0.25, 0.12, 0.12, 0.12, 0.12],  # 5 FI Senior SF
    [0.20, 0.20, 0.34, 0.25, 0.25, 0.00, 0.20, 0.49, 0.16, 0.15, 0.15, 0.16],  # 6 Corporate Senior Unsec
    [0.40, 0.40, 0.25, 0.25, 0.25, 0.20, 0.00, 0.20, 0.08, 0.12, 0.12, 0.08],  # 7 Sovereign entity
    [0.20, 0.20, 0.34, 0.25, 0.25, 0.49, 0.20, 0.00, 0.16, 0.15, 0.15, 0.16],  # 8 Corporate Senior Sec
    [0.08, 0.08, 0.14, 0.12, 0.12, 0.16, 0.08, 0.16, 0.00, 0.43, 0.43, 0.43],  # 9 Asset backed
    [0.12, 0.12, 0.15, 0.12, 0.12, 0.15, 0.12, 0.15, 0.43, 0.00, 0.58, 0.43],  # 10 RMBS
    [0.12, 0.12, 0.15, 0.12, 0.12, 0.15, 0.12, 0.15, 0.43, 0.58, 0.00, 0.43],  # 11 CMBS
    [0.08, 0.08, 0.14, 0.12, 0.12, 0.16, 0.08, 0.16, 0.43, 0.43, 0.43, 0.00],  # 12 Other SF
])

# Labels for Credit Qualifying buckets
CREDIT_QUALIFYING_BUCKET_LABELS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
]


# ============================================================================
# VEGA RISK WEIGHTS AND BASE CORRELATION
# ============================================================================

# Credit Qualifying Vega Risk Weight (VRW)
CREDIT_QUALIFYING_VRW = 0.76

# Credit Qualifying base correlation risk weight
CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT = 10

# Credit Qualifying base correlation inter-index correlation (29%)
CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION = 0.29


# ============================================================================
# CONCENTRATION THRESHOLDS
# ============================================================================

# Credit Qualifying delta concentration thresholds (USD million)
# Source: ISDA SIMM v2.6, Section 2.2.3
CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS = {
    1: 300,   # Sovereign
    2: 300,   # Local authority
    3: 150,   # Financial institution - Senior Unsecured
    4: 300,   # Covered bond
    5: 150,   # Financial institution - Senior Structured Finance
    6: 200,   # Corporate - Senior Unsecured
    7: 300,   # Sovereign entity
    8: 200,   # Corporate - Senior Secured
    9: 40,    # Asset backed
    10: 40,   # Residential mortgage backed
    11: 40,   # Commercial mortgage backed
    12: 40,   # Other structured finance
    "Residual": 20
}

# Credit Qualifying vega concentration threshold (USD million)
# Source: ISDA SIMM v2.6, Section 2.2.4
CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD = 360
