"""
Portfolio management module.

This module provides comprehensive portfolio management capabilities for
tracking positions, calculating valuations, and aggregating risk metrics
across multiple assets and products.

Supports multiple asset classes:
- Equity: Options, futures, delta-one products
- Fixed Income: Bonds, bond futures, interest rate derivatives
- FX: Spot, forward, swap, vanilla/digital/quanto options

Main components:
- BasePosition, BasePortfolio: Asset-agnostic protocols
- EquityPosition, EquityPortfolio: Equity-specific implementations
- FIPosition, FIPortfolio: Fixed Income-specific implementations
- FXPosition, FXPortfolio: FX-specific implementations
- PortfolioSnapshot: Point-in-time snapshot of portfolio state
- PortfolioExporter: Export functionality for parquet and excel formats

Backward Compatibility:
- Position: Alias for EquityPosition
- Portfolio: Alias for EquityPortfolio
"""

# Base protocols
from .base import BasePosition, BasePortfolio

# Equity implementations (with backward-compatible aliases)
from .equity import (
    EquityPosition,
    EquityPortfolio,
    Position,  # Backward compatibility alias
    Portfolio,  # Backward compatibility alias
)

# Fixed Income implementations
from .fi import (
    FIPosition,
    FIPortfolio,
)

# FX implementations (FxPosition is a backward-compatible alias of FXPosition)
from .fx import (
    FXPosition,
    FxPosition,
    FXPortfolio,
)

# Utilities
from .portfolio_snapshot import PortfolioSnapshot
from .portfolio_storage import PortfolioExporter

__all__ = [
    # Base protocols
    "BasePosition",
    "BasePortfolio",
    # Equity
    "EquityPosition",
    "EquityPortfolio",
    # Fixed Income
    "FIPosition",
    "FIPortfolio",
    # FX
    "FXPosition",
    "FxPosition",
    "FXPortfolio",
    # Backward compatibility aliases
    "Position",
    "Portfolio",
    # Utilities
    "PortfolioSnapshot",
    "PortfolioExporter",
]
