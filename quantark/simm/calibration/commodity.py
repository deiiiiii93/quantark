"""
Commodity Calibration Parameters for SIMM v2.6

This module contains all Commodity calibration parameters as specified in
ISDA SIMM v2.6, effective December 2, 2023.

References:
    - ISDA SIMM Methodology, Version 2.6, Section 2.5 (Commodity Risk)
"""

import numpy as np


# ============================================================================
# RISK WEIGHTS
# ============================================================================

# Commodity risk weights by bucket (17 buckets)
# Source: ISDA SIMM v2.6, Table CO-1
COMMODITY_RISK_WEIGHTS = {
    1: 70,   # Energy - Oil
    2: 70,   # Energy - Gas
    3: 70,   # Energy - Power
    4: 70,   # Energy - Coal
    5: 84,   # Metals - Base
    6: 84,   # Metals - Precious
    7: 84,   # Metals - Other
    8: 58,   # Agriculture - Grains
    9: 58,   # Agriculture - Softs
    10: 58,  # Agriculture - Livestock
    11: 58,  # Agriculture - Other
    12: 60,  # Freight
    13: 68,  # Weather
    14: 68,  # Emission
    15: 68,  # Nuclear
    16: 68,  # Alternative Energy
    17: 68   # Other
}


# ============================================================================
# INTRA-BUCKET CORRELATIONS
# ============================================================================

# Commodity intra-bucket correlations by bucket
# Source: ISDA SIMM v2.6, Section 2.5.1
COMMODITY_INTRA_BUCKET_CORRELATIONS = {
    1: 0.34,  # Energy - Oil
    2: 0.36,  # Energy - Gas
    3: 0.26,  # Energy - Power
    4: 0.31,  # Energy - Coal
    5: 0.25,  # Metals - Base
    6: 0.25,  # Metals - Precious
    7: 0.25,  # Metals - Other
    8: 0.24,  # Agriculture - Grains
    9: 0.25,  # Agriculture - Softs
    10: 0.19, # Agriculture - Livestock
    11: 0.25, # Agriculture - Other
    12: 0.31, # Freight
    13: 0.15, # Weather
    14: 0.18, # Emission
    15: 0.18, # Nuclear
    16: 0.18, # Alternative Energy
    17: 0.25  # Other
}


# ============================================================================
# INTER-BUCKET CORRELATIONS
# ============================================================================

# Commodity inter-bucket correlation matrix (17x17)
# Source: ISDA SIMM v2.6, Table CO-2
COMMODITY_INTER_BUCKET_CORRELATIONS = np.array([
    #    1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17
    [0.00, 0.72, 0.63, 0.63, 0.17, 0.17, 0.17, 0.16, 0.16, 0.14, 0.16, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 1 Oil
    [0.72, 0.00, 0.57, 0.57, 0.17, 0.17, 0.17, 0.16, 0.16, 0.14, 0.16, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 2 Gas
    [0.63, 0.57, 0.00, 0.54, 0.17, 0.17, 0.17, 0.16, 0.16, 0.14, 0.16, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 3 Power
    [0.63, 0.57, 0.54, 0.00, 0.17, 0.17, 0.17, 0.16, 0.16, 0.14, 0.16, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 4 Coal
    [0.17, 0.17, 0.17, 0.17, 0.00, 0.53, 0.53, 0.16, 0.16, 0.14, 0.16, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 5 Base Metals
    [0.17, 0.17, 0.17, 0.17, 0.53, 0.00, 0.53, 0.16, 0.16, 0.14, 0.16, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 6 Precious Metals
    [0.17, 0.17, 0.17, 0.17, 0.53, 0.53, 0.00, 0.16, 0.16, 0.14, 0.16, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 7 Other Metals
    [0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.00, 0.44, 0.38, 0.44, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 8 Grains
    [0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.44, 0.00, 0.38, 0.44, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 9 Softs
    [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.38, 0.38, 0.00, 0.38, 0.18, 0.11, 0.11, 0.11, 0.11, 0.14],  # 10 Livestock
    [0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.44, 0.44, 0.38, 0.00, 0.18, 0.11, 0.11, 0.11, 0.11, 0.16],  # 11 Other Agriculture
    [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.00, 0.11, 0.11, 0.11, 0.11, 0.18],  # 12 Freight
    [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.00, 0.11, 0.11, 0.11, 0.11],  # 13 Weather
    [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.00, 0.11, 0.11, 0.11],  # 14 Emission
    [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.00, 0.11, 0.11],  # 15 Nuclear
    [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.00, 0.11],  # 16 Alt Energy
    [0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.14, 0.16, 0.18, 0.11, 0.11, 0.11, 0.11, 0.00],  # 17 Other
])

# Labels for Commodity buckets
COMMODITY_BUCKET_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


# ============================================================================
# HISTORICAL VOLATILITY RATIOS AND VEGA RISK WEIGHTS
# ============================================================================

# Commodity Historical Volatility Ratio (HVR)
COMMODITY_HVR = 0.74

# Commodity Vega Risk Weight (VRW)
COMMODITY_VRW = 0.55


# ============================================================================
# CONCENTRATION THRESHOLDS
# ============================================================================

# Commodity delta concentration thresholds (USD million / %)
# Source: ISDA SIMM v2.6, Section 2.5.3
COMMODITY_DELTA_CONCENTRATION_THRESHOLDS = {
    1: 100,  # Energy - Oil
    2: 100,  # Energy - Gas
    3: 100,  # Energy - Power
    4: 100,  # Energy - Coal
    5: 80,   # Metals - Base
    6: 80,   # Metals - Precious
    7: 80,   # Metals - Other
    8: 20,   # Agriculture - Grains
    9: 20,   # Agriculture - Softs
    10: 20,  # Agriculture - Livestock
    11: 20,  # Agriculture - Other
    12: 30,  # Freight
    13: 30,  # Weather
    14: 30,  # Emission
    15: 30,  # Nuclear
    16: 30,  # Alternative Energy
    17: 30   # Other
}

# Commodity vega concentration thresholds (USD million / %)
# Source: ISDA SIMM v2.6, Section 2.5.4
COMMODITY_VEGA_CONCENTRATION_THRESHOLDS = {
    1: 100,  # Energy - Oil
    2: 100,  # Energy - Gas
    3: 100,  # Energy - Power
    4: 100,  # Energy - Coal
    5: 80,   # Metals - Base
    6: 80,   # Metals - Precious
    7: 80,   # Metals - Other
    8: 20,   # Agriculture - Grains
    9: 20,   # Agriculture - Softs
    10: 20,  # Agriculture - Livestock
    11: 20,  # Agriculture - Other
    12: 30,  # Freight
    13: 30,  # Weather
    14: 30,  # Emission
    15: 30,  # Nuclear
    16: 30,  # Alternative Energy
    17: 30   # Other
}
