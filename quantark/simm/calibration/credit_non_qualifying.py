"""
Credit Non-Qualifying calibration parameters for ISDA SIMM v2.6.

All values are transcribed from the ISDA SIMM Methodology, version 2.6,
Section F (Credit Non-Qualifying risk) and Sections J.2/J.7 (concentration
thresholds). Paragraph references are given on each table.
"""


# ============================================================================
# DELTA RISK WEIGHTS (paragraph 46)
# ============================================================================

CREDIT_NON_QUALIFYING_RISK_WEIGHTS = {
    1: 280.0,
    2: 1300.0,
    "Residual": 1300.0,
}

# Vega risk weight for Credit Non-Qualifying (paragraph 47).
CREDIT_NON_QUALIFYING_VRW = 0.76


# ============================================================================
# CORRELATIONS (paragraphs 48-49)
# ============================================================================

# Intra-bucket correlations rho_kl (paragraph 48).
# Same group name (such as CMBX, ABX): 83%. Different group name: 32%.
# Residual bucket: 50% for both cases.
CREDIT_NON_QUALIFYING_SAME_GROUP_CORRELATION = 0.83
CREDIT_NON_QUALIFYING_DIFFERENT_GROUP_CORRELATION = 0.32
CREDIT_NON_QUALIFYING_RESIDUAL_SAME_GROUP_CORRELATION = 0.50
CREDIT_NON_QUALIFYING_RESIDUAL_DIFFERENT_GROUP_CORRELATION = 0.50

# Inter-bucket correlation gamma_bc, non-residual to non-residual
# (paragraph 49).
CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION = 0.43


# ============================================================================
# CONCENTRATION THRESHOLDS (Sections J.2 and J.7)
# ============================================================================

# Delta concentration thresholds, USD mm/bp (paragraph 76).
CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS = {
    1: 9.5,
    2: 0.5,
    "Residual": 0.5,
}

# Vega concentration threshold, USD mm (paragraph 83), all buckets
# including residual.
CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD = 70.0
