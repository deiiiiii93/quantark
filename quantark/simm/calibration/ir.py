"""
Interest Rate calibration parameters for ISDA SIMM v2.6.

All values are transcribed from the ISDA SIMM Methodology, version 2.6
(effective December 2, 2023), Section D (Interest Rate risk) and
Sections J.1/J.6 (concentration thresholds). Paragraph references are
given on each table.

Conventions:
- Delta risk weights multiply PV01-style sensitivities directly
  (s = V(r + 1bp) - V(r), paragraph 22): WS = RW * s * CR.
- Concentration thresholds are stored in USD millions (per bp for delta);
  the engine converts to base currency units.
"""

import numpy as np

from quantark.simm.taxonomy import (
    CurrencyVolatility,
    IRConcentrationGroup,
    IR_TENOR_LABELS,
)


# ============================================================================
# DELTA RISK WEIGHTS (paragraph 33)
# ============================================================================

# Risk weights per vertex, by currency volatility group. Keys follow
# IR_TENOR_LABELS ordering: 2w 1m 3m 6m 1y 2y 3y 5y 10y 15y 20y 30y.
IR_DELTA_RISK_WEIGHTS = {
    CurrencyVolatility.REGULAR: dict(zip(IR_TENOR_LABELS, (
        109, 105, 90, 71, 66, 66, 64, 60, 60, 61, 61, 67))),
    CurrencyVolatility.LOW: dict(zip(IR_TENOR_LABELS, (
        15, 18, 9, 11, 13, 15, 19, 23, 23, 22, 22, 23))),
    CurrencyVolatility.HIGH: dict(zip(IR_TENOR_LABELS, (
        163, 109, 87, 89, 102, 96, 101, 97, 97, 102, 106, 101))),
}

# The risk weight for any currency's inflation rate (paragraph 33).
IR_INFLATION_RISK_WEIGHT = 61

# The risk weight for any currency's cross-currency basis swap spread
# (paragraph 33).
IR_XCCY_BASIS_RISK_WEIGHT = 21


# ============================================================================
# VEGA RISK WEIGHT AND HISTORICAL VOLATILITY RATIO (paragraphs 34-35)
# ============================================================================

# Historical volatility ratio for the interest-rate risk class (paragraph 34).
IR_HVR = 0.47

# Vega risk weight for the Interest Rate risk class (paragraph 35).
IR_VRW = 0.23


# ============================================================================
# CORRELATIONS (paragraphs 36-37)
# ============================================================================

# 12x12 tenor correlation matrix rho_kl (paragraph 36), indexed by
# IR_TENOR_LABELS order.
IR_TENOR_CORRELATIONS = np.array([
    #  2w     1m     3m     6m     1y     2y     3y     5y     10y    15y    20y    30y
    [1.00, 0.77, 0.67, 0.59, 0.48, 0.39, 0.34, 0.30, 0.25, 0.23, 0.21, 0.20],  # 2w
    [0.77, 1.00, 0.84, 0.74, 0.56, 0.43, 0.36, 0.31, 0.26, 0.21, 0.19, 0.19],  # 1m
    [0.67, 0.84, 1.00, 0.88, 0.69, 0.55, 0.47, 0.40, 0.34, 0.27, 0.25, 0.25],  # 3m
    [0.59, 0.74, 0.88, 1.00, 0.86, 0.73, 0.65, 0.57, 0.49, 0.40, 0.38, 0.37],  # 6m
    [0.48, 0.56, 0.69, 0.86, 1.00, 0.94, 0.87, 0.79, 0.68, 0.60, 0.57, 0.55],  # 1y
    [0.39, 0.43, 0.55, 0.73, 0.94, 1.00, 0.96, 0.91, 0.80, 0.74, 0.70, 0.69],  # 2y
    [0.34, 0.36, 0.47, 0.65, 0.87, 0.96, 1.00, 0.97, 0.88, 0.81, 0.77, 0.76],  # 3y
    [0.30, 0.31, 0.40, 0.57, 0.79, 0.91, 0.97, 1.00, 0.95, 0.90, 0.86, 0.85],  # 5y
    [0.25, 0.26, 0.34, 0.49, 0.68, 0.80, 0.88, 0.95, 1.00, 0.97, 0.94, 0.94],  # 10y
    [0.23, 0.21, 0.27, 0.40, 0.60, 0.74, 0.81, 0.90, 0.97, 1.00, 0.98, 0.97],  # 15y
    [0.21, 0.19, 0.25, 0.38, 0.57, 0.70, 0.77, 0.86, 0.94, 0.98, 1.00, 0.99],  # 20y
    [0.20, 0.19, 0.25, 0.37, 0.55, 0.69, 0.76, 0.85, 0.94, 0.97, 0.99, 1.00],  # 30y
])

# Index of each tenor label within the correlation matrix.
IR_TENOR_INDEX = {label: i for i, label in enumerate(IR_TENOR_LABELS)}

# Correlation phi_ij between any two sub-curves of the same currency
# (paragraph 36).
IR_SUB_CURVE_CORRELATION = 0.993

# Correlation between the inflation rate (or inflation volatility) and any
# yield (or interest-rate volatility) for the same currency (paragraph 36).
IR_INFLATION_CORRELATION = 0.24

# Correlation between the cross-currency basis swap spread and any yield or
# inflation rate for the same currency (paragraph 36).
IR_XCCY_BASIS_CORRELATION = 0.04

# gamma_bc for aggregating across different currencies (paragraph 37).
IR_INTER_CURRENCY_CORRELATION = 0.32


# ============================================================================
# CONCENTRATION THRESHOLDS (Sections J.1 and J.6)
# ============================================================================

# Delta concentration thresholds, USD mm/bp (paragraph 74).
IR_DELTA_CONCENTRATION_THRESHOLDS = {
    IRConcentrationGroup.HIGH_VOLATILITY: 30.0,
    IRConcentrationGroup.REGULAR_WELL_TRADED: 330.0,
    IRConcentrationGroup.REGULAR_LESS_WELL_TRADED: 130.0,
    IRConcentrationGroup.LOW_VOLATILITY: 61.0,
}

# Vega concentration thresholds, USD mm (paragraph 81).
IR_VEGA_CONCENTRATION_THRESHOLDS = {
    IRConcentrationGroup.HIGH_VOLATILITY: 74.0,
    IRConcentrationGroup.REGULAR_WELL_TRADED: 4900.0,
    IRConcentrationGroup.REGULAR_LESS_WELL_TRADED: 520.0,
    IRConcentrationGroup.LOW_VOLATILITY: 970.0,
}
