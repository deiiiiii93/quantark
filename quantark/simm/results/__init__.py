"""
SIMM Results Module.

This module provides comprehensive result dataclasses for SIMM calculation
including hierarchical margin breakdown, attribution, and what-if analysis.
"""
from .simm_result import (
    SIMMResult,
    RiskClassMargin,
    BucketDetail,
    SensitivityContribution,
    AddonBreakdown,
)
from .attribution import (
    SIMMAttribution,
    PositionAttribution,
    ContributorInfo,
)
from .whatif import (
    SIMMWhatIf,
    WhatIfResult,
)

__all__ = [
    # Result dataclasses
    "SIMMResult",
    "RiskClassMargin",
    "BucketDetail",
    "SensitivityContribution",
    "AddonBreakdown",
    # Attribution
    "SIMMAttribution",
    "PositionAttribution",
    "ContributorInfo",
    # What-if analysis
    "SIMMWhatIf",
    "WhatIfResult",
]
