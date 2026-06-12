"""
SIMM Aggregation Engine Module.

Implements the ISDA SIMM v2.6 margin aggregation:
- Risk-factor netting and weighted sensitivities (WS = RW * s * CR)
- Concentration risk factors (CR, VCR, f_kl, g_bc)
- Intra-/inter-bucket correlations (Sections D-I)
- Bucket-level aggregation (K_b)
- Risk class margins (Delta / Vega / Curvature / BaseCorr)
- Product class aggregation (psi correlations, Section K)
- The SIMMCalculator orchestrator (per product class, paragraph 6)
"""

from quantark.simm.engines.aggregation.concentration import (
    ConcentrationCalculator,
    ConcentrationResult,
)
from quantark.simm.engines.aggregation.correlations import (
    intra_bucket_correlation,
    inter_bucket_correlation,
)
from quantark.simm.engines.aggregation.weighted_sensitivity import (
    NettedSensitivity,
    WeightedSensitivity,
    WeightedSensitivityCalculator,
    delta_risk_weight,
    vega_risk_weight,
    hvr_for_vega,
    net_by_risk_factor,
)
from quantark.simm.engines.aggregation.bucket_aggregator import (
    BucketAggregator,
    BucketResult,
)
from quantark.simm.engines.aggregation.risk_class_aggregator import (
    CurvatureExposure,
    RiskClassAggregator,
    RiskClassResult,
)
from quantark.simm.engines.aggregation.product_class_aggregator import (
    ProductClassAggregator,
    ProductClassResult,
)
from quantark.simm.engines.aggregation.addon import (
    AddOnCalculator,
    AddOnResult,
)
from quantark.simm.engines.aggregation.simm_calculator import (
    SIMMCalculator,
    SIMMAggregationResult,
)

__all__ = [
    # Concentration
    "ConcentrationCalculator",
    "ConcentrationResult",
    # Correlations
    "intra_bucket_correlation",
    "inter_bucket_correlation",
    # Netting / weighting
    "NettedSensitivity",
    "WeightedSensitivity",
    "WeightedSensitivityCalculator",
    "delta_risk_weight",
    "vega_risk_weight",
    "hvr_for_vega",
    "net_by_risk_factor",
    # Bucket aggregation
    "BucketAggregator",
    "BucketResult",
    # Risk class aggregation
    "CurvatureExposure",
    "RiskClassAggregator",
    "RiskClassResult",
    # Product class aggregation
    "ProductClassAggregator",
    "ProductClassResult",
    # Add-on
    "AddOnCalculator",
    "AddOnResult",
    # Main calculator
    "SIMMCalculator",
    "SIMMAggregationResult",
]
