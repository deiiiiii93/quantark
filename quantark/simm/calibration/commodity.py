"""
Commodity calibration parameters for ISDA SIMM v2.6.

All values are transcribed from the ISDA SIMM Methodology, version 2.6,
Section H (Commodity risk) and Sections J.4/J.9 (concentration thresholds).
Paragraph references are given on each table.
"""

import numpy as np


# ============================================================================
# DELTA RISK WEIGHTS (paragraph 61)
# ============================================================================

COMMODITY_RISK_WEIGHTS = {
    1: 48.0,    # Coal
    2: 29.0,    # Crude
    3: 33.0,    # Light Ends
    4: 25.0,    # Middle Distillates
    5: 35.0,    # Heavy Distillates
    6: 30.0,    # North America Natural Gas
    7: 60.0,    # European Natural Gas
    8: 52.0,    # North American Power
    9: 68.0,    # European Power and Carbon
    10: 63.0,   # Freight
    11: 21.0,   # Base Metals
    12: 21.0,   # Precious Metals
    13: 15.0,   # Grains and Oilseed
    14: 16.0,   # Softs and Other Agriculturals
    15: 13.0,   # Livestock and Dairy
    16: 68.0,   # Other
    17: 17.0,   # Indexes
}

# Historical volatility ratio for the commodity risk class (paragraph 62).
COMMODITY_HVR = 0.74

# Vega risk weight for the commodity risk class (paragraph 63).
COMMODITY_VRW = 0.55


# ============================================================================
# CORRELATIONS (paragraphs 64-65)
# ============================================================================

# Intra-bucket correlations rho_kl (paragraph 64).
COMMODITY_INTRA_BUCKET_CORRELATIONS = {
    1: 0.83,
    2: 0.97,
    3: 0.93,
    4: 0.97,
    5: 0.98,
    6: 0.90,
    7: 0.98,
    8: 0.49,
    9: 0.80,
    10: 0.46,
    11: 0.58,
    12: 0.53,
    13: 0.62,
    14: 0.16,
    15: 0.18,
    16: 0.00,
    17: 0.38,
}

# Inter-bucket correlations gamma_bc (paragraph 65), 17x17 indexed by
# bucket-1.
COMMODITY_INTER_BUCKET_CORRELATIONS = np.array([
    #   1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17
    [1.00,  0.22,  0.18,  0.21,  0.20,  0.24,  0.49,  0.16,  0.38,  0.14,  0.10,  0.02,  0.12,  0.11,  0.02,  0.00,  0.17],  # 1
    [0.22,  1.00,  0.92,  0.90,  0.88,  0.25,  0.08,  0.19,  0.17,  0.17,  0.42,  0.28,  0.36,  0.27,  0.20,  0.00,  0.64],  # 2
    [0.18,  0.92,  1.00,  0.87,  0.84,  0.16,  0.07,  0.15,  0.10,  0.18,  0.33,  0.22,  0.27,  0.23,  0.16,  0.00,  0.54],  # 3
    [0.21,  0.90,  0.87,  1.00,  0.77,  0.19,  0.11,  0.18,  0.16,  0.14,  0.32,  0.22,  0.28,  0.22,  0.11,  0.00,  0.58],  # 4
    [0.20,  0.88,  0.84,  0.77,  1.00,  0.19,  0.09,  0.12,  0.13,  0.18,  0.42,  0.34,  0.32,  0.29,  0.13,  0.00,  0.59],  # 5
    [0.24,  0.25,  0.16,  0.19,  0.19,  1.00,  0.31,  0.62,  0.23,  0.10,  0.21,  0.05,  0.18,  0.10,  0.08,  0.00,  0.28],  # 6
    [0.49,  0.08,  0.07,  0.11,  0.09,  0.31,  1.00,  0.21,  0.79,  0.17,  0.10, -0.08,  0.10,  0.07, -0.02,  0.00,  0.13],  # 7
    [0.16,  0.19,  0.15,  0.18,  0.12,  0.62,  0.21,  1.00,  0.16,  0.08,  0.13, -0.07,  0.07,  0.05,  0.02,  0.00,  0.19],  # 8
    [0.38,  0.17,  0.10,  0.16,  0.13,  0.23,  0.79,  0.16,  1.00,  0.15,  0.09, -0.06,  0.06,  0.06,  0.01,  0.00,  0.16],  # 9
    [0.14,  0.17,  0.18,  0.14,  0.18,  0.10,  0.17,  0.08,  0.15,  1.00,  0.16,  0.09,  0.14,  0.09,  0.03,  0.00,  0.11],  # 10
    [0.10,  0.42,  0.33,  0.32,  0.42,  0.21,  0.10,  0.13,  0.09,  0.16,  1.00,  0.36,  0.30,  0.25,  0.18,  0.00,  0.37],  # 11
    [0.02,  0.28,  0.22,  0.22,  0.34,  0.05, -0.08, -0.07, -0.06,  0.09,  0.36,  1.00,  0.20,  0.18,  0.11,  0.00,  0.26],  # 12
    [0.12,  0.36,  0.27,  0.28,  0.32,  0.18,  0.10,  0.07,  0.06,  0.14,  0.30,  0.20,  1.00,  0.28,  0.19,  0.00,  0.39],  # 13
    [0.11,  0.27,  0.23,  0.22,  0.29,  0.10,  0.07,  0.05,  0.06,  0.09,  0.25,  0.18,  0.28,  1.00,  0.13,  0.00,  0.26],  # 14
    [0.02,  0.20,  0.16,  0.11,  0.13,  0.08, -0.02,  0.02,  0.01,  0.03,  0.18,  0.11,  0.19,  0.13,  1.00,  0.00,  0.21],  # 15
    [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00],  # 16
    [0.17,  0.64,  0.54,  0.58,  0.59,  0.28,  0.13,  0.19,  0.16,  0.11,  0.37,  0.26,  0.39,  0.26,  0.21,  0.00,  1.00],  # 17
])


# ============================================================================
# CONCENTRATION THRESHOLDS (Sections J.4 and J.9)
# ============================================================================

# Delta concentration thresholds, USD mm/% (paragraph 78).
COMMODITY_DELTA_CONCENTRATION_THRESHOLDS = {
    1: 310.0,    # Coal
    2: 2100.0,   # Crude Oil
    3: 1700.0,   # Oil Fractions
    4: 1700.0,
    5: 1700.0,
    6: 2800.0,   # Natural gas
    7: 2800.0,
    8: 2700.0,   # Power
    9: 2700.0,
    10: 52.0,    # Freight, Dry or Wet
    11: 530.0,   # Base metals
    12: 1300.0,  # Precious Metals
    13: 100.0,   # Agricultural
    14: 100.0,
    15: 100.0,
    16: 52.0,    # Other
    17: 4000.0,  # Indices
}

# Vega concentration thresholds, USD mm (paragraph 85).
COMMODITY_VEGA_CONCENTRATION_THRESHOLDS = {
    1: 390.0,    # Coal
    2: 2900.0,   # Crude Oil
    3: 310.0,    # Oil fractions
    4: 310.0,
    5: 310.0,
    6: 6300.0,   # Natural gas
    7: 6300.0,
    8: 1200.0,   # Power
    9: 1200.0,
    10: 120.0,   # Freight, Dry or Wet
    11: 390.0,   # Base metals
    12: 1300.0,  # Precious Metals
    13: 590.0,   # Agricultural
    14: 590.0,
    15: 590.0,
    16: 69.0,    # Other
    17: 69.0,    # Indices
}
