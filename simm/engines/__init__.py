"""
SIMM Sensitivity Engines

This module provides sensitivity calculation engines for ISDA SIMM (Standard Initial Margin Model) v2.6.

The engines follow a protocol-based architecture with support for:
- Interest Rate sensitivities (delta, vega, curvature)
- Equity sensitivities (delta, vega)
- Portfolio-to-sensitivity conversion
- CRIF format import/export
"""

from simm.engines.base import (
    SensitivityEngine,
    BaseSensitivityEngine,
)
from simm.engines.factory import create_engine
from simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from simm.engines.result import SIMMResult

__all__ = [
    "SensitivityEngine",
    "BaseSensitivityEngine",
    "create_engine",
    "SIMMPortfolioAdapter",
    "SIMMResult",
]
