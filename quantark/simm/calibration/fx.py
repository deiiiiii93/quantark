"""
Foreign Exchange calibration parameters for ISDA SIMM v2.6.

All values are transcribed from the ISDA SIMM Methodology, version 2.6,
Section I (Foreign Exchange risk) and Sections J.5/J.10 (concentration
thresholds). Paragraph references are given on each table.

FX risk weights and correlations depend on the FX volatility group
(paragraphs 67-68: high = BRL, RUB, TRY; regular = all others) of both
the currency concerned and the calculation currency.
"""

from quantark.simm.taxonomy import FXVolatilityGroup


# ============================================================================
# DELTA RISK WEIGHTS (paragraph 69)
# ============================================================================

# Risk weight by (volatility group of given currency, volatility group of
# calculation currency). There is no FX risk factor for the calculation
# currency itself.
FX_RISK_WEIGHTS = {
    (FXVolatilityGroup.REGULAR, FXVolatilityGroup.REGULAR): 7.4,
    (FXVolatilityGroup.REGULAR, FXVolatilityGroup.HIGH): 14.7,
    (FXVolatilityGroup.HIGH, FXVolatilityGroup.REGULAR): 14.7,
    (FXVolatilityGroup.HIGH, FXVolatilityGroup.HIGH): 21.4,
}

# Historical volatility ratio for the FX risk class (paragraph 70).
FX_HVR = 0.57

# Vega risk weight for FX volatility (paragraph 71).
FX_VRW = 0.48


# ============================================================================
# CORRELATIONS (paragraphs 72-73)
# ============================================================================

# Delta correlations rho_kl between two FX risk factors, keyed by the
# calculation currency's volatility group, then by the volatility groups
# of the two risk factor currencies (paragraph 72).
FX_DELTA_CORRELATIONS = {
    FXVolatilityGroup.REGULAR: {
        (FXVolatilityGroup.REGULAR, FXVolatilityGroup.REGULAR): 0.50,
        (FXVolatilityGroup.REGULAR, FXVolatilityGroup.HIGH): 0.25,
        (FXVolatilityGroup.HIGH, FXVolatilityGroup.REGULAR): 0.25,
        (FXVolatilityGroup.HIGH, FXVolatilityGroup.HIGH): -0.05,
    },
    FXVolatilityGroup.HIGH: {
        (FXVolatilityGroup.REGULAR, FXVolatilityGroup.REGULAR): 0.88,
        (FXVolatilityGroup.REGULAR, FXVolatilityGroup.HIGH): 0.72,
        (FXVolatilityGroup.HIGH, FXVolatilityGroup.REGULAR): 0.72,
        (FXVolatilityGroup.HIGH, FXVolatilityGroup.HIGH): 0.50,
    },
}

# Correlation for pairs of FX volatility and curvature risk factors
# (paragraph 73).
FX_VEGA_CORRELATION = 0.50


# ============================================================================
# CONCENTRATION THRESHOLDS (Sections J.5 and J.10)
# ============================================================================

# Delta concentration thresholds by FX category of the currency, USD mm/%
# (paragraph 79). Categories per paragraph 80.
FX_DELTA_CONCENTRATION_THRESHOLDS = {
    1: 3300.0,   # Category 1 - Significantly material
    2: 880.0,    # Category 2 - Frequently traded
    3: 170.0,    # Category 3 - Others
}

# Vega concentration thresholds by unordered category pair of the two
# currencies in the pair, USD mm (paragraph 86).
FX_VEGA_CONCENTRATION_THRESHOLDS = {
    frozenset((1,)): 2800.0,      # Category 1 - Category 1
    frozenset((1, 2)): 1400.0,    # Category 1 - Category 2
    frozenset((1, 3)): 590.0,     # Category 1 - Category 3
    frozenset((2,)): 520.0,       # Category 2 - Category 2
    frozenset((2, 3)): 340.0,     # Category 2 - Category 3
    frozenset((3,)): 210.0,       # Category 3 - Category 3
}


def get_fx_vega_concentration_threshold(category_1: int, category_2: int) -> float:
    """Vega concentration threshold for a currency pair (paragraph 86)."""
    return FX_VEGA_CONCENTRATION_THRESHOLDS[frozenset((category_1, category_2))]
