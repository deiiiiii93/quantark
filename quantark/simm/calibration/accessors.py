"""
Shared calibration accessors for ISDA SIMM v2.6.

Provides the cross-risk-class correlation lookup (Section K) and the
volatility scale factor used to construct implied volatilities from delta
risk weights for Equity, FX and Commodity vega (paragraph 10(b)).
"""

import math

from quantark.simm.taxonomy import RiskClass
from quantark.simm.calibration.cross_risk import INTER_RISK_CLASS_CORRELATIONS


# Order of risk classes in the Section K correlation matrix.
RISK_CLASS_ORDER = (
    RiskClass.INTEREST_RATE,
    RiskClass.CREDIT_QUALIFYING,
    RiskClass.CREDIT_NON_QUALIFYING,
    RiskClass.EQUITY,
    RiskClass.COMMODITY,
    RiskClass.FX,
)

_RISK_CLASS_INDEX = {rc: i for i, rc in enumerate(RISK_CLASS_ORDER)}

# 99th percentile of the standard normal distribution (paragraph 10(b)).
PHI_INV_99 = 2.3263478740408408

# 99.5th percentile of the standard normal distribution (paragraph 11(d)).
PHI_INV_995 = 2.5758293035489004

# sigma_kj = RW_k * VOL_SCALE for Equity, FX and Commodity instruments
# (paragraph 10(b)): sigma_kj = RW_k * sqrt(365/14) / PHI_INV(99%).
VOL_SCALE = math.sqrt(365.0 / 14.0) / PHI_INV_99


def get_inter_risk_class_correlation(
    risk_class_1: RiskClass, risk_class_2: RiskClass
) -> float:
    """Correlation psi_rs between two risk classes within a product class
    (paragraph 88, Section K).
    """
    i = _RISK_CLASS_INDEX[risk_class_1]
    j = _RISK_CLASS_INDEX[risk_class_2]
    return float(INTER_RISK_CLASS_CORRELATIONS[i, j])


def scaling_function(expiry_days: float) -> float:
    """Curvature scaling function SF(t) (paragraph 11(a)).

    SF(t) = 0.5 * min(1, 14 / t), with t the expiry time in calendar days.
    """
    if expiry_days <= 0:
        return 0.5
    return 0.5 * min(1.0, 14.0 / expiry_days)
