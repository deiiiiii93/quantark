"""
Credit Non-Qualifying Calibration Parameters for SIMM v2.6

This module contains all Credit Non-Qualifying (CreditNQ) calibration parameters
as specified in ISDA SIMM v2.6, effective December 2, 2023.

References:
    - ISDA SIMM Methodology, Version 2.6, Section 2.3 (Credit Non-Qualifying Risk)
"""

import numpy as np


# ============================================================================
# RISK WEIGHTS
# ============================================================================

# Credit Non-Qualifying risk weights by bucket (2 buckets + residual)
# Source: ISDA SIMM v2.6, Table CN-1
CREDIT_NON_QUALIFYING_RISK_WEIGHTS = {
    1: 135,  # Sovereign
    2: 135,  # Corporate
    "Residual": 200
}


# ============================================================================
# INTRA-BUCKET CORRELATIONS
# ============================================================================

# Credit Non-Qualifying intra-bucket correlations
# Same issuer correlation vs different issuer correlation
# Source: ISDA SIMM v2.6, Section 2.3.1
CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS = {
    # Same issuer, different risk factor
    "same_issuer": 0.99,
    # Different issuer, same bucket
    1: 0.17,  # Sovereign
    2: 0.28,  # Corporate
    "Residual": 0.28
}


# ============================================================================
# INTER-BUCKET CORRELATIONS
# ============================================================================

# Credit Non-Qualifying inter-bucket correlation (43%)
# Correlation between bucket 1 and bucket 2
# Source: ISDA SIMM v2.6, Table CN-2
CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION = 0.43

# Credit Non-Qualifying bucket labels
CREDIT_NON_QUALIFYING_BUCKET_LABELS = [1, 2]


# ============================================================================
# VEGA RISK WEIGHTS
# ============================================================================

# Credit Non-Qualifying Vega Risk Weight (VRW)
CREDIT_NON_QUALIFYING_VRW = 0.76


# ============================================================================
# CONCENTRATION THRESHOLDS
# ============================================================================

# Credit Non-Qualifying delta concentration thresholds (USD million)
# Source: ISDA SIMM v2.6, Section 2.3.3
CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS = {
    1: 300,  # Sovereign
    2: 200,  # Corporate
    "Residual": 20
}

# Credit Non-Qualifying vega concentration threshold (USD million)
# Source: ISDA SIMM v2.6, Section 2.3.4
CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD = 70
