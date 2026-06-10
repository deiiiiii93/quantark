"""
SIMM Aggregation Engine Module.

This module implements the ISDA SIMM margin calculation aggregation logic
following the SIMM specification Sections B and 5-13.

Components:
- Concentration risk factors (CR, VCR, g_bc)
- Weighted sensitivity calculation (WS = RW × s × CR)
- Bucket-level aggregation (K_b formula)
- Risk class aggregation (Delta/Vega/Curvature margins)
- Product class aggregation (SIMM_product)
- Main SIMMCalculator class
"""

from quantark.simm.engines.aggregation.concentration import (
    ConcentrationCalculator,
    ConcentrationResult,
)
from quantark.simm.engines.aggregation.weighted_sensitivity import (
    WeightedSensitivityCalculator,
    WeightedSensitivity,
)
from quantark.simm.engines.aggregation.bucket_aggregator import (
    BucketAggregator,
    BucketResult,
)
from quantark.simm.engines.aggregation.risk_class_aggregator import (
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
    # Weighted Sensitivity
    "WeightedSensitivityCalculator",
    "WeightedSensitivity",
    # Bucket Aggregation
    "BucketAggregator",
    "BucketResult",
    # Risk Class Aggregation
    "RiskClassAggregator",
    "RiskClassResult",
    # Product Class Aggregation
    "ProductClassAggregator",
    "ProductClassResult",
    # Add-On
    "AddOnCalculator",
    "AddOnResult",
    # Main Calculator
    "SIMMCalculator",
    "SIMMAggregationResult",
]
