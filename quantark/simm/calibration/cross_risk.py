"""
Cross-Risk-Class Calibration Parameters for SIMM v2.6

This module contains inter-risk-class correlation parameters (ψ) as specified
in ISDA SIMM v2.6, effective December 2, 2023.

References:
    - ISDA SIMM Methodology, Version 2.6, Section 1.4.2 (Inter-Risk-Class Correlations)
"""

import numpy as np


# ============================================================================
# INTER-RISK-CLASS CORRELATIONS (ψ)
# ============================================================================

# Inter-risk-class correlation matrix (6x6)
# Risk classes: IR, CreditQ, CreditNQ, Equity, Commodity, FX
# Source: ISDA SIMM v2.6, Table Global-1
INTER_RISK_CLASS_CORRELATIONS = np.array([
    #     IR    CreditQ CreditNQ Equity Comm  FX
    [1.00, 0.04,   0.04,    0.07, 0.37, 0.14],  # IR
    [0.04, 1.00,   0.54,    0.70, 0.27, 0.37],  # CreditQ
    [0.04, 0.54,   1.00,    0.46, 0.24, 0.15],  # CreditNQ
    [0.07, 0.70,   0.46,    1.00, 0.35, 0.39],  # Equity
    [0.37, 0.27,   0.24,    0.35, 1.00, 0.35],  # Commodity
    [0.14, 0.37,   0.15,    0.39, 0.35, 1.00],  # FX
])

# Labels for risk classes in the correlation matrix
INTER_RISK_CLASS_CORRELATION_LABELS = [
    "IR",
    "CreditQ",
    "CreditNQ",
    "Equity",
    "Commodity",
    "FX"
]
