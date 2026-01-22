"""
Pricing engines for equity derivatives.
"""

from .base_engine import BaseEngine
from .analytical import (
    BlackScholesEngine,
    BarrierAnalyticalEngine,
    OneTouchAnalyticalEngine,
    AsianOptionAnalyticalEngine,
)
from .pde_engine import PDEEngine
from .mc import (
    EuropeanMCEngine,
    AmericanOptionMCEngine,
    SnowballMCEngine,
    PhoenixMCEngine,
    DigitalOptionMCEngine,
    BarrierOptionMCEngine,
)
from .pde import (
    BasePDESolver,
    EuropeanPDESolver,
    AmericanPDESolver,
    BarrierPDESolver,
    DoubleBarrierPDESolver,
    OneTouchPDESolver,
    DoubleOneTouchPDESolver,
    PhoenixPDESolver,
    TimeGrid,
    SpatialGrid,
)
from .quad import (
    EuropeanQuadEngine,
    BarrierQuadEngine,
    OneTouchQuadEngine,
    SnowballQuadEngine,
    PhoenixQuadEngine,
)

__all__ = [
    # Base
    "BaseEngine",
    # Analytical
    "BlackScholesEngine",
    "BarrierAnalyticalEngine",
    "OneTouchAnalyticalEngine",
    "AsianOptionAnalyticalEngine",
    # Monte Carlo
    "EuropeanMCEngine",
    "AmericanOptionMCEngine",
    "SnowballMCEngine",
    "PhoenixMCEngine",
    "DigitalOptionMCEngine",
    "BarrierOptionMCEngine",
    # Unified PDE Engine
    "PDEEngine",
    # PDE Solvers
    "BasePDESolver",
    "EuropeanPDESolver",
    "AmericanPDESolver",
    "BarrierPDESolver",
    "DoubleBarrierPDESolver",
    "OneTouchPDESolver",
    "DoubleOneTouchPDESolver",
    "PhoenixPDESolver",
    # Grid utilities
    "TimeGrid",
    "SpatialGrid",
    # Quadrature
    "EuropeanQuadEngine",
    "BarrierQuadEngine",
    "OneTouchQuadEngine",
    "SnowballQuadEngine",
    "PhoenixQuadEngine",
]
